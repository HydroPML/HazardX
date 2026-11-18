"""Shared tensor contract for native-grid FloodSimBench context-6 models.

This module is intentionally separate from ``spatial_adapter.py`` because the
existing FloodCastBench wrappers have a public three-channel contract.  The
FloodSimBench experiment uses twelve physical channels and must not weaken or
change that older contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


INPUT_CHANNELS = 12
OUTPUT_CHANNELS = 1
CROP_SIZE = 1024


@dataclass(frozen=True)
class SpatialPad:
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int


def validate_context12_input(
    x: torch.Tensor, *, require_crop1024: bool = True
) -> Tuple[int, int, int]:
    if x.ndim != 4 or x.shape[1] != INPUT_CHANNELS:
        raise AssertionError(
            f"expected FloodSimBench input [B,{INPUT_CHANNELS},H,W], got {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise TypeError(f"FloodSimBench input must be floating point, got {x.dtype}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("FloodSimBench input contains NaN/Inf")
    batch, _, height, width = (int(value) for value in x.shape)
    if require_crop1024 and (height, width) != (CROP_SIZE, CROP_SIZE):
        raise AssertionError(
            f"native benchmark requires 1024x1024, got {(height, width)}"
        )
    return batch, height, width


def minimal_pad_bottom_right_context12(
    x: torch.Tensor,
    multiple: int,
    *,
    value: float = 0.0,
) -> tuple[torch.Tensor, SpatialPad]:
    _, height, width = validate_context12_input(x)
    if multiple < 1:
        raise ValueError("padding multiple must be positive")
    pad_bottom = (-height) % int(multiple)
    pad_right = (-width) % int(multiple)
    padded = F.pad(x, (0, pad_right, 0, pad_bottom), value=float(value))
    return padded, SpatialPad(
        original_height=height,
        original_width=width,
        padded_height=height + pad_bottom,
        padded_width=width + pad_right,
    )


def crop_to_original(x: torch.Tensor, pad: SpatialPad) -> torch.Tensor:
    if x.shape[-2] < pad.original_height or x.shape[-1] < pad.original_width:
        raise AssertionError(
            f"model output {tuple(x.shape[-2:])} is smaller than the requested crop"
        )
    return x[..., : pad.original_height, : pad.original_width].contiguous()


def validate_context12_output(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    expected = (x.shape[0], OUTPUT_CHANNELS, x.shape[-2], x.shape[-1])
    if tuple(y.shape) != expected:
        raise AssertionError(f"expected output {expected}, got {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise FloatingPointError("FloodSimBench model output contains NaN/Inf")
    return y
