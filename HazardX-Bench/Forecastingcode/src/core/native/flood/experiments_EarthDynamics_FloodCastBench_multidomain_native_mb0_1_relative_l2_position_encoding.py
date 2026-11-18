"""Physical-position ablations of the native mb0_1 relative-L2 baseline.

Everything outside positional encoding is delegated to
``experiments_EarthDynamics_FloodCastBench_multidomain_native_mb0_1_relative_l2``.
This keeps its datasets, seeds, optimizer, loss, scheduler, DDP behavior,
training loop, validation, rollout, and checkpointing unchanged.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments_EarthDynamics_FloodCastBench_multidomain_native_mb0_1_relative_l2 as baseline
from gphyt.model import model_specs
from gphyt.model.tokenizer.tokenizer import Detokenizer
from gphyt.model.transformer.physical_position_models import (
    POSITION_ENCODINGS,
    SPACE_REF_METERS,
    build_physical_position_gphyt,
)


DOMAIN_RESOLUTION_METERS = {
    "australia": 60.0,
    "pakistan": 480.0,
    "uk_30m": 30.0,
    "uk": 60.0,
    "mozambique": 480.0,
}


class NativeGridPhysicalPositionEarthDynamics(nn.Module):
    """Native-grid GPhyT with physical RoPE or physical Fourier-coordinate APE."""

    def __init__(
        self,
        patch_height: int = 16,
        patch_width: int = 16,
        model_size: str = "GPT_S",
        attention_mode: str = "spatial",
        tokenizer_mode: str = "linear",
        detokenizer_mode: str = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        use_derivatives: bool = False,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        direct_derivative_output: bool = True,
        position_encoding: str = "physical_rope",
        space_ref: float = SPACE_REF_METERS,
    ) -> None:
        super().__init__()
        if not direct_derivative_output:
            raise ValueError(
                "NativeGridPhysicalPositionEarthDynamics requires "
                "direct_derivative_output=True"
            )
        if patch_height < 1 or patch_width < 1:
            raise ValueError("Patch dimensions must be positive")
        if attention_mode not in {"spatial", "full"}:
            raise ValueError(
                "One-step Markov mode supports dynamic spatial/full attention only"
            )
        if position_encoding not in POSITION_ENCODINGS:
            raise ValueError(
                f"--position-encoding must select exactly one of "
                f"{POSITION_ENCODINGS}, got {position_encoding!r}"
            )
        if not math.isfinite(space_ref) or space_ref <= 0:
            raise ValueError(f"space_ref must be finite and positive, got {space_ref}")
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.input_channels = len(baseline.CHANNEL_NAMES)
        self.position_encoding = position_encoding
        self.space_ref = float(space_ref)
        patch_size = (1, self.patch_height, self.patch_width)
        gpt_spec = getattr(model_specs, model_size)()
        model_config = {
            "architecture": "gphyt",
            # T=1 selects spatial attention. H/W are dynamic in every forward.
            "img_size": (1, self.patch_height, self.patch_width),
            "transformer": {
                "input_channels": self.input_channels,
                "model_size": model_size,
                "att_mode": attention_mode,
                "dropout": dropout,
                "position_encoding": position_encoding,
                "space_ref": self.space_ref,
                "patch_size": list(patch_size),
                "stochastic_depth_rate": stochastic_depth_rate,
                "use_derivatives": use_derivatives,
                "integrator": None,
            },
            "tokenizer": {
                "tokenizer_mode": tokenizer_mode,
                "detokenizer_mode": detokenizer_mode,
                "tokenizer_overlap": tokenizer_overlap,
                "detokenizer_overlap": detokenizer_overlap,
            },
        }
        self.backbone = build_physical_position_gphyt(
            model_config,
            hidden_dim=gpt_spec.hidden_dim,
            mlp_dim=gpt_spec.mlp_dim,
            num_heads=gpt_spec.num_heads,
            num_layers=gpt_spec.num_layers,
            conv_channels=gpt_spec.conv_channels,
        )
        # Match the baseline's direct one-channel derivative detokenizer.
        self.backbone.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=gpt_spec.hidden_dim,
            out_channels=1,
            mode=detokenizer_mode,
            conv_net_channels=gpt_spec.conv_channels,
            overlap=detokenizer_overlap,
            img_size=(1, self.patch_height, self.patch_width),
        )
        self.direct_derivative_output = True

    def padding_for(self, height: int, width: int) -> Tuple[int, int]:
        return (
            (-int(height)) % self.patch_height,
            (-int(width)) % self.patch_width,
        )

    @staticmethod
    def _metadata_values(
        metadata: Optional[Mapping[str, Any]],
        key: str,
        *,
        required: bool,
    ) -> Any:
        value = None if metadata is None else metadata.get(key)
        if required and value is None:
            raise ValueError(
                f"metadata[{key!r}] is required by physical position encoding"
            )
        return value

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise AssertionError(
                f"x must be [B,2,H,W] (h,rain), got {tuple(x.shape)}"
            )
        if static is None or static.ndim != 4 or static.shape[1] != 1:
            raise AssertionError(
                "static terrain must be present with shape [B,1,H,W]"
            )
        if x.shape[0] != static.shape[0] or x.shape[-2:] != static.shape[-2:]:
            raise AssertionError(
                f"x/static mismatch: {tuple(x.shape)} vs {tuple(static.shape)}"
            )
        if not torch.isfinite(x).all() or not torch.isfinite(static).all():
            raise FloatingPointError("Model input contains NaN or Inf")
        batch_size = int(x.shape[0])
        height, width = int(x.shape[-2]), int(x.shape[-1])

        meta_heights = self._metadata_values(metadata, "height", required=False)
        meta_widths = self._metadata_values(metadata, "width", required=False)
        if meta_heights is not None:
            values = torch.as_tensor(meta_heights).reshape(-1)
            if values.numel() not in {1, batch_size} or bool(
                (values != height).any()
            ):
                raise AssertionError("metadata height does not match x")
        if meta_widths is not None:
            values = torch.as_tensor(meta_widths).reshape(-1)
            if values.numel() not in {1, batch_size} or bool(
                (values != width).any()
            ):
                raise AssertionError("metadata width does not match x")
        dx = self._metadata_values(metadata, "dx", required=True)
        dy = self._metadata_values(metadata, "dy", required=True)

        fields = torch.cat((x, static), dim=1)
        pad_h, pad_w = self.padding_for(height, width)
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), value=0.0)
        # Flood GPhyT external layout is [B,T,H,W,C].
        fields = fields.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        derivative_fields = self.backbone(fields, dx=dx, dy=dy)
        if derivative_fields.shape[-1] != 1:
            raise AssertionError(
                "EarthDynamics backbone must directly return one derivative "
                f"channel, got {tuple(derivative_fields.shape)}"
            )
        dh = (
            derivative_fields[:, 0, :height, :width, :]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        expected = (batch_size, 1, height, width)
        if tuple(dh.shape) != expected:
            raise AssertionError(
                f"Unpadded dh shape {tuple(dh.shape)} != expected {expected}"
            )
        if not torch.isfinite(dh).all():
            raise FloatingPointError("Model output dh contains NaN or Inf")
        return dh


_BASE_BUILD_PARSER = baseline.build_parser
_BASE_MODEL_CONFIG_FROM_ARGS = baseline.model_config_from_args


def build_parser() -> argparse.ArgumentParser:
    """Extend only position metadata/config while retaining every base default."""
    parser = _BASE_BUILD_PARSER()
    parser.description = (
        "FloodCastBench native-grid exact mb0_1 relative-L2 physical-position "
        "ablation; all non-position training settings match the source baseline"
    )
    parser.add_argument(
        "--position-encoding",
        choices=POSITION_ENCODINGS,
        required=True,
        help=(
            "physical_rope rotates post-head Q/K only; fourier_ape adds a "
            "dynamic physical-coordinate embedding to tokenizer output only"
        ),
    )
    parser.set_defaults(
        output_dir=(
            "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
            "FloodCastBench_multidomain_native_mb0_1_relative_l2_position_encoding"
        ),
        australia_dx=DOMAIN_RESOLUTION_METERS["australia"],
        australia_dy=DOMAIN_RESOLUTION_METERS["australia"],
        pakistan_dx=DOMAIN_RESOLUTION_METERS["pakistan"],
        pakistan_dy=DOMAIN_RESOLUTION_METERS["pakistan"],
        uk30_dx=DOMAIN_RESOLUTION_METERS["uk_30m"],
        uk30_dy=DOMAIN_RESOLUTION_METERS["uk_30m"],
        uk_dx=DOMAIN_RESOLUTION_METERS["uk"],
        uk_dy=DOMAIN_RESOLUTION_METERS["uk"],
        mozambique_dx=DOMAIN_RESOLUTION_METERS["mozambique"],
        mozambique_dy=DOMAIN_RESOLUTION_METERS["mozambique"],
    )
    return parser


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = _BASE_MODEL_CONFIG_FROM_ARGS(args)
    config.update(
        {
            "position_encoding": args.position_encoding,
            "space_ref": SPACE_REF_METERS,
        }
    )
    return config


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = NativeGridPhysicalPositionEarthDynamics(**dict(config))
    if any(
        name.startswith("delta_projection")
        for name, _ in model.named_parameters()
    ):
        raise AssertionError("Direct-derivative model still has projection weights")
    if model.position_encoding == "physical_rope":
        if model.backbone.coordinate_embedding is not None:
            raise AssertionError("Physical RoPE must disable absolute token encoding")
    elif any(
        block.attention.rope is not None
        for block in model.backbone.attention_blocks
    ):
        raise AssertionError("Fourier-coordinate APE must disable RoPE completely")
    return model.to(device)


def main() -> None:
    # The baseline's functions resolve these factories from their own module
    # globals, so install the position-only variants before entering its main.
    baseline.build_parser = build_parser
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.main()


if __name__ == "__main__":
    main()
