"""Shared native-grid padding helpers for FloodCastBench baselines.

Only bottom and right padding is permitted.  These helpers deliberately do
not expose resize operations, so wrappers cannot accidentally resample the
physical fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialPad:
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int

    @property
    def pad_bottom(self) -> int:
        return self.padded_height - self.original_height

    @property
    def pad_right(self) -> int:
        return self.padded_width - self.original_width


def validate_flood_input(x: torch.Tensor) -> Tuple[int, int, int]:
    if x.ndim != 4 or x.shape[1] != 3:
        raise AssertionError(f"expected x [B,3,H,W], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"flood input must be floating point, got {x.dtype}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("flood input contains NaN/Inf")
    batch, _, height, width = (int(v) for v in x.shape)
    if min(batch, height, width) < 1:
        raise ValueError(f"empty flood input {tuple(x.shape)}")
    return batch, height, width


def minimal_pad_bottom_right(
    x: torch.Tensor,
    multiple_height: int,
    multiple_width: int | None = None,
    *,
    value: float = 0.0,
) -> tuple[torch.Tensor, SpatialPad]:
    """Pad a BCHW tensor to integer spatial multiples on bottom/right only."""
    _, height, width = validate_flood_input(x)
    multiple_width = multiple_height if multiple_width is None else multiple_width
    if multiple_height < 1 or multiple_width < 1:
        raise ValueError("padding multiples must be positive")
    pad_bottom = (-height) % int(multiple_height)
    pad_right = (-width) % int(multiple_width)
    padded = F.pad(x, (0, pad_right, 0, pad_bottom), value=float(value))
    spec = SpatialPad(
        original_height=height,
        original_width=width,
        padded_height=height + pad_bottom,
        padded_width=width + pad_right,
    )
    if tuple(padded.shape[-2:]) != (spec.padded_height, spec.padded_width):
        raise AssertionError("bottom/right padding produced the wrong shape")
    return padded, spec


def crop_to_original(x: torch.Tensor, spec: SpatialPad) -> torch.Tensor:
    if x.ndim < 2:
        raise AssertionError("cannot spatially crop a tensor with fewer than 2 dims")
    if x.shape[-2] < spec.original_height or x.shape[-1] < spec.original_width:
        raise AssertionError(
            f"output {tuple(x.shape[-2:])} is smaller than requested crop "
            f"{(spec.original_height, spec.original_width)}"
        )
    return x[..., : spec.original_height, : spec.original_width].contiguous()


def assert_flood_output(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    expected = (x.shape[0], 1, x.shape[-2], x.shape[-1])
    if tuple(y.shape) != expected:
        raise AssertionError(f"expected output {expected}, got {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise FloatingPointError("model output contains NaN/Inf")
    return y
