"""Deterministic 128-pixel crops and overlap-add inference utilities.

The helpers in this module are deliberately model agnostic.  Training keeps
the existing FloodCastBench tensors and only changes the spatial view; tiled
evaluation calls the public trainer model interface and reconstructs one full
grid before the autoregressive state is advanced.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_int_seed(*parts: object) -> int:
    """Return a process-independent 63-bit seed for arbitrary identifiers."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def tile_starts(length: int, window_size: int = 128, stride: int = 96) -> List[int]:
    """Generate fixed-stride starts and anchor the final tile to the far edge."""

    length = int(length)
    window_size = int(window_size)
    stride = int(stride)
    if length < 1 or window_size < 1 or stride < 1:
        raise ValueError(
            f"length/window/stride must be positive, got {length}/{window_size}/{stride}"
        )
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final = length - window_size
    if starts[-1] != final:
        starts.append(final)
    if starts[0] != 0 or starts[-1] != final:
        raise AssertionError("tile starts do not cover both physical boundaries")
    if any(right - left > window_size for left, right in zip(starts, starts[1:])):
        raise AssertionError("tile layout leaves an uncovered gap")
    return starts


def clipped_hann_weight(
    window_size: int,
    *,
    floor: float = 0.05,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """A positive 2-D Hann weight; unlike a raw Hann its boundary is nonzero."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError(f"blend floor must be in (0,1], got {floor}")
    one_dimensional = torch.hann_window(
        int(window_size), periodic=False, dtype=torch.float32, device=device
    )
    weight = torch.outer(one_dimensional, one_dimensional)
    weight = weight / weight.max().clamp_min(torch.finfo(weight.dtype).eps)
    weight = weight.clamp_min(float(floor)).unsqueeze(0).unsqueeze(0)
    if not torch.isfinite(weight).all() or not bool((weight > 0).all()):
        raise FloatingPointError("invalid clipped-Hann blending weight")
    return weight


def _repeat_metadata(
    metadata: Optional[Mapping[str, Any]],
    key: str,
    batch: int,
    repeats: int,
) -> Optional[List[Any]]:
    if metadata is None or metadata.get(key) is None:
        return None
    values = metadata[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = [values] * batch
    values = list(values)
    if len(values) != batch:
        raise ValueError(
            f"metadata[{key!r}] has {len(values)} values for batch size {batch}"
        )
    return values * repeats


def _pad_eval_inputs(
    x: torch.Tensor,
    static: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    height, width = (int(value) for value in x.shape[-2:])
    pad_h = max(0, int(window_size) - height)
    pad_w = max(0, int(window_size) - width)
    if pad_h == 0 and pad_w == 0:
        return x, static, (height, width)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    static = F.pad(static, (0, pad_w, 0, pad_h), mode="replicate")
    return x, static, (height, width)


def tiled_model_delta(
    model: nn.Module,
    *,
    x: torch.Tensor,
    static: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
    window_size: int = 128,
    stride: int = 96,
    tile_batch_size: int = 1,
    blend_floor: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Predict and stitch the trainer's normalized ``dh`` on one full grid."""

    if x.ndim != 4 or x.shape[1] != 2:
        raise AssertionError(f"x must be [B,2,H,W], got {tuple(x.shape)}")
    if static.ndim != 4 or static.shape[1] != 1:
        raise AssertionError(
            f"static must be [B,1,H,W], got {tuple(static.shape)}"
        )
    if x.shape[0] != static.shape[0] or x.shape[-2:] != static.shape[-2:]:
        raise AssertionError("x/static batch or spatial shape mismatch")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be positive")
    if not torch.isfinite(x).all() or not torch.isfinite(static).all():
        raise FloatingPointError("tiled inference input contains NaN/Inf")

    original_height, original_width = (int(value) for value in x.shape[-2:])
    x, static, original_shape = _pad_eval_inputs(x, static, int(window_size))
    padded_height, padded_width = (int(value) for value in x.shape[-2:])
    rows = tile_starts(padded_height, window_size, stride)
    columns = tile_starts(padded_width, window_size, stride)
    positions = [(top, left) for top in rows for left in columns]
    batch = int(x.shape[0])
    weight = clipped_hann_weight(
        int(window_size), floor=blend_floor, device=x.device
    )
    numerator = torch.zeros(
        (batch, 1, padded_height, padded_width),
        dtype=torch.float32,
        device=x.device,
    )
    denominator = torch.zeros(
        (1, 1, padded_height, padded_width),
        dtype=torch.float32,
        device=x.device,
    )

    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    for offset in range(0, len(positions), int(tile_batch_size)):
        chunk = positions[offset : offset + int(tile_batch_size)]
        x_tiles = torch.cat(
            [
                x[:, :, top : top + window_size, left : left + window_size]
                for top, left in chunk
            ],
            dim=0,
        )
        static_tiles = torch.cat(
            [
                static[:, :, top : top + window_size, left : left + window_size]
                for top, left in chunk
            ],
            dim=0,
        )
        tile_metadata: Dict[str, Any] = {
            "height": [int(window_size)] * (len(chunk) * batch),
            "width": [int(window_size)] * (len(chunk) * batch),
        }
        for key in ("dt_minutes", "dx", "dy"):
            repeated = _repeat_metadata(metadata, key, batch, len(chunk))
            if repeated is not None:
                tile_metadata[key] = repeated
        prediction = model(
            x=x_tiles,
            static=static_tiles,
            metadata=tile_metadata,
        )
        expected = (len(chunk) * batch, 1, window_size, window_size)
        if tuple(prediction.shape) != expected:
            raise AssertionError(
                f"tile prediction {tuple(prediction.shape)} != {expected}"
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("tile prediction contains NaN/Inf")
        prediction = prediction.float().reshape(
            len(chunk), batch, 1, window_size, window_size
        )
        for index, (top, left) in enumerate(chunk):
            numerator[
                :, :, top : top + window_size, left : left + window_size
            ] += prediction[index] * weight
            denominator[
                :, :, top : top + window_size, left : left + window_size
            ] += weight

    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    if not torch.isfinite(denominator).all() or not bool((denominator > 0).all()):
        raise FloatingPointError("tiled reconstruction has invalid/uncovered pixels")
    stitched = numerator / denominator
    stitched = stitched[:, :, :original_height, :original_width]
    if tuple(stitched.shape[-2:]) != original_shape:
        raise AssertionError("tiled output was not cropped to the original grid")
    if not torch.isfinite(stitched).all():
        raise FloatingPointError("stitched output contains NaN/Inf")
    return stitched, {
        "window_size": int(window_size),
        "stride": int(stride),
        "tile_batch_size": int(tile_batch_size),
        "row_starts": rows,
        "column_starts": columns,
        "tile_count": len(positions),
        "inference_seconds": float(elapsed),
        "blend": "clipped_hann",
        "blend_floor": float(blend_floor),
        "minimum_denominator": float(denominator.min().detach().cpu()),
    }
