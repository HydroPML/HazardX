"""Direct-next FloodCastBench adapters for the repository UNet and FNO.

The actual backbones are constructed by ``floodcast_four_baselines`` so their
architecture and DDP-compatible FNO spectral layers stay identical to the
four-baseline experiment.  These wrappers expose the public benchmark API
``[B,3,H,W] -> [B,1,H,W]`` and return absolute normalized ``h_next``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import floodcast_four_baselines as four


BASELINE_NAMES = ("UNet", "FNO")


def _validate_fields(fields: torch.Tensor) -> Tuple[int, int]:
    if fields.ndim != 4 or fields.shape[1] != 3:
        raise AssertionError(f"fields must be [B,3,H,W], got {tuple(fields.shape)}")
    if not torch.isfinite(fields).all():
        raise FloatingPointError("UNet/FNO input contains NaN/Inf")
    return int(fields.shape[-2]), int(fields.shape[-1])


def _validate_prediction(
    prediction: torch.Tensor, fields: torch.Tensor
) -> torch.Tensor:
    expected = (fields.shape[0], 1, fields.shape[-2], fields.shape[-1])
    if tuple(prediction.shape) != expected:
        raise AssertionError(
            f"absolute h_next shape {tuple(prediction.shape)} != {expected}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("UNet/FNO h_next contains NaN/Inf")
    return prediction


class UNetDirectNext(nn.Module):
    """Exact four-baseline UNet core with an absolute-state public output."""

    experiment_name = "UNet"
    core_prediction = "h_next"
    spatial_resize = False

    def __init__(self, *, init_features: int = 32, grid_type: str = "cartesian"):
        super().__init__()
        source = four.NativeGridUNet(
            init_features=int(init_features),
            grid_type=str(grid_type),
            output_method=four.OUTPUT_DIRECT,
        )
        self.backbone = source.core
        self.init_features = int(init_features)
        self.grid_type = str(grid_type)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        height, width = _validate_fields(fields)
        pad_h, pad_w = four.NativeGridUNet.padding_for(height, width)
        padded = fields
        if pad_h or pad_w:
            padded = F.pad(padded, (0, pad_w, 0, pad_h), value=0.0)
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        official_output = self.backbone(official_input)
        prediction = (
            official_output[:, :height, :width, 0, 0]
            .unsqueeze(1)
            .contiguous()
        )
        return _validate_prediction(prediction, fields)


class FNODirectNext(nn.Module):
    """Exact four-baseline DDP-compatible FNO with absolute-state output."""

    experiment_name = "FNO"
    core_prediction = "h_next"
    spatial_resize = False

    def __init__(
        self,
        *,
        modes_height: int = 12,
        modes_width: int = 12,
        width: int = 20,
        grid_type: str = "cartesian",
    ) -> None:
        super().__init__()
        source = four.NativeGridFNO(
            modes_height=int(modes_height),
            modes_width=int(modes_width),
            width=int(width),
            grid_type=str(grid_type),
            output_method=four.OUTPUT_DIRECT,
        )
        # Includes the exact real/imaginary DDP reparameterization installed by
        # NativeGridFNO for each of the four spectral convolution blocks.
        self.backbone = source.core
        self.modes_height = int(modes_height)
        self.modes_width = int(modes_width)
        self.width = int(width)
        self.grid_type = str(grid_type)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        height, width = _validate_fields(fields)
        official_input = fields.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        # Preserve the four-baseline policy: native CUDA FFT executes in FP32.
        with torch.amp.autocast(device_type=fields.device.type, enabled=False):
            official_output = self.backbone(official_input.float())
        prediction = (
            official_output[:, :height, :width, 0, 0]
            .unsqueeze(1)
            .contiguous()
        )
        return _validate_prediction(prediction, fields)

