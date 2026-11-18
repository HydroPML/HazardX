"""NaN-safe MSE baseline with post-head Physical 2D RoPE.

Training, data loading, normalization, optimizer, scheduler, DDP, rollout,
checkpointing, and the FP32 MSE loss are delegated unchanged to
``experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe``.
Only the model factory is replaced with the Physical 2D RoPE implementation
from the physical-position ablation.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
from experiments_EarthDynamics_FloodCastBench_multidomain_native_mb0_1_relative_l2_position_encoding import (
    DOMAIN_RESOLUTION_METERS,
    NativeGridPhysicalPositionEarthDynamics,
)
from gphyt.model.transformer.physical_position_models import SPACE_REF_METERS


POSITION_ENCODING = "physical_rope"

_BASE_BUILD_PARSER = baseline.build_parser
_BASE_MODEL_CONFIG_FROM_ARGS = baseline.model_config_from_args


def build_parser() -> argparse.ArgumentParser:
    """Keep every NaN-safe MSE default and add fixed physical-grid metadata."""
    parser = _BASE_BUILD_PARSER()
    parser.description = (
        "FloodCastBench native-grid NaN-safe MSE baseline with post-head "
        "Physical 2D RoPE; all non-position settings match the NaN-safe baseline"
    )
    parser.add_argument(
        "--position-encoding",
        choices=(POSITION_ENCODING,),
        default=POSITION_ENCODING,
        help=(
            "Fixed to physical_rope: rotate actual per-head Q/K with physical "
            "x/y coordinates and no absolute token position embedding"
        ),
    )
    parser.set_defaults(
        output_dir="results/FloodCastBench_native_nan_safe_physical_rope_2gpu_all",
        loss="mse",
        normalization="train_standard",
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
    """Extend the NaN-safe model config with Physical RoPE metadata only."""
    config = _BASE_MODEL_CONFIG_FROM_ARGS(args)
    config.update(
        {
            "position_encoding": args.position_encoding,
            "space_ref": SPACE_REF_METERS,
        }
    )
    return config


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Build the post-head Physical RoPE model used by the MSE training loop."""
    model = NativeGridPhysicalPositionEarthDynamics(**dict(config))
    if model.position_encoding != POSITION_ENCODING:
        raise AssertionError(
            f"MSE Physical RoPE baseline received {model.position_encoding!r}"
        )
    if model.backbone.coordinate_embedding is not None:
        raise AssertionError("Physical RoPE must disable absolute token encoding")
    if any(
        block.attention.rope is None
        or block.attention.rope.head_dim != block.attention.head_dim
        for block in model.backbone.attention_blocks
    ):
        raise AssertionError(
            "Physical RoPE must rotate actual Q/K after attention-head splitting"
        )
    if any(
        name.startswith("delta_projection")
        for name, _ in model.named_parameters()
    ):
        raise AssertionError("Direct-derivative model still has projection weights")
    return model.to(device)


def main() -> None:
    # The NaN-safe functions resolve these factories through their module
    # globals, so install the physical-position variants before entering main.
    baseline.build_parser = build_parser
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.main()


if __name__ == "__main__":
    main()
