"""Shape-dynamic wrapper adapters for official learned positional tables."""

from __future__ import annotations

from types import MethodType
from typing import Tuple

import torch
import torch.nn.functional as F


def _gphyt_dynamic_abs_forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or self.pe.ndim != 5 or x.shape[-1] != self.pe.shape[-1]:
        raise RuntimeError(
            f"unconfirmed GPhyT positional layout: x={tuple(x.shape)}, "
            f"pe={tuple(self.pe.shape)}"
        )
    position = self.pe
    if tuple(position.shape[1:4]) != tuple(x.shape[1:4]):
        position = F.interpolate(
            position.permute(0, 4, 1, 2, 3).float(),
            size=x.shape[1:4],
            mode="trilinear",
            align_corners=False,
        ).permute(0, 2, 3, 4, 1).to(dtype=x.dtype)
    return x + position


def install_gphyt_dynamic_absolute_position(
    backbone,
    *,
    checkpoint_grid: Tuple[int, int, int] = (4, 16, 8),
) -> None:
    """Keep the official module/table and interpolate it per current grid.

    The official GPhyT-S checkpoint table is ``[1,4,16,8,192]``.  Building
    that exact official parameter shape permits an exact checkpoint load; the
    wrapper changes only its addition-time spatial lookup, as required for
    FloodCastBench's heterogeneous native grids.
    """
    current = backbone.init_pos_encodings
    if current is None or not hasattr(current, "pe"):
        raise RuntimeError("official GPhyT absolute position module is missing")
    channels = int(current.pe.shape[-1])
    module_type = type(current)
    replacement = module_type(
        num_channels=channels,
        time=int(checkpoint_grid[0]),
        height=int(checkpoint_grid[1]),
        width=int(checkpoint_grid[2]),
    )
    replacement.forward = MethodType(_gphyt_dynamic_abs_forward, replacement)
    backbone.init_pos_encodings = replacement

