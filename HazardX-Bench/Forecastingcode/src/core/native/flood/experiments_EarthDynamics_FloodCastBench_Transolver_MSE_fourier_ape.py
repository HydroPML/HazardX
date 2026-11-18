"""FloodCastBench residual-conditioned Transolver with Physical Fourier APE.

The model predicts only a normalized single-channel ``delta_h``:

    h[t-1], rainfall[t-1], terrain -> delta_h

The imported NaN-safe FloodCastBench loop is solely responsible for the FP32
Euler update, non-negative water constraint, and masked MSE.  This file does
not copy data loading or training code.

Model path:

    shared dynamic-grid patch tokenizer
      -> one PhysicalFourierCoordinateAPE
      -> residual-conditioned Transolver physics-attention blocks (M=64)
      -> direct one-channel delta decoder

There is no internal integrator, learned fixed-grid position table, or RoPE.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
from experiments_EarthDynamics_FloodCastBench_multidomain_native_mb0_1_relative_l2_position_encoding import (
    DOMAIN_RESOLUTION_METERS,
    NativeGridPhysicalPositionEarthDynamics,
)
from gphyt.model import model_specs
from gphyt.model.tokenizer.tokenizer import Detokenizer, Tokenizer
from gphyt.model.transformer.attention_transolver_slice_residual_conditioned import (
    ResidualConditionedTransolverSliceAttentionBlock2d,
)
from gphyt.model.transformer.physical_position_models import (
    SPACE_REF_METERS,
    Physical2DRotaryEmbedding,
    PhysicalFourierCoordinateAPE,
    _as_batch_spacing,
)
from gphyt.model.transformer.pos_encodings import AbsPositionalEmbedding


EXPERIMENT_NAME = "FloodCastBench_Transolver_MSE_PhysicalFourierAPE"
POSITION_ENCODING = "fourier_ape"
SLICE_NUM = 64

_BASE_BUILD_PARSER = baseline.build_parser
_BASE_MODEL_CONFIG_FROM_ARGS = baseline.model_config_from_args


def _is_primary_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class PhysicalFourierResidualTransolverCore(nn.Module):
    """Dynamic-grid tokenizer/APE/Transolver/direct-delta backbone."""

    def __init__(
        self,
        *,
        num_fields: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: Tuple[int, int, int],
        img_size: Tuple[int, int, int],
        tokenizer_mode: str,
        detokenizer_mode: str,
        tokenizer_overlap: int,
        detokenizer_overlap: int,
        tokenizer_net_channels: Sequence[int],
        detokenizer_net_channels: Sequence[int],
        dropout: float,
        stochastic_depth_rate: float,
        slice_num: int,
        state_dim: Optional[int],
        state_reduction: int,
        fno_modes_height: int,
        fno_modes_width: int,
        fno_dropout: float,
        local_kernel: int,
        init_local_gamma: float,
        init_global_gamma: float,
        space_ref: float,
    ) -> None:
        super().__init__()
        if slice_num != SLICE_NUM:
            raise ValueError(f"Physical slice count is fixed at M={SLICE_NUM}")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not math.isfinite(space_ref) or space_ref <= 0:
            raise ValueError("space_ref must be finite and positive")

        self.num_fields = int(num_fields)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.position_encoding = POSITION_ENCODING
        self.slice_num = SLICE_NUM
        # Explicit public state used by acceptance tests/checkpoint inspection.
        self.integrator = None
        self.returns_delta = True

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=num_fields,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
            conv_net_channels=list(tokenizer_net_channels),
            overlap=tokenizer_overlap,
        )
        self.physical_fourier_ape = PhysicalFourierCoordinateAPE(
            hidden_dim=hidden_dim,
            space_ref=space_ref,
        )
        self.attention_blocks = nn.ModuleList(
            [
                ResidualConditionedTransolverSliceAttentionBlock2d(
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    slice_num=SLICE_NUM,
                    state_dim=state_dim,
                    state_reduction=state_reduction,
                    dropout=dropout,
                    stochastic_depth_rate=stochastic_depth_rate,
                    fno_modes_height=fno_modes_height,
                    fno_modes_width=fno_modes_width,
                    fno_dropout=fno_dropout,
                    local_kernel=local_kernel,
                    init_local_gamma=init_local_gamma,
                    init_global_gamma=init_global_gamma,
                )
                for _ in range(num_layers)
            ]
        )
        self.delta_decoder = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=1,
            mode=detokenizer_mode,
            conv_net_channels=list(detokenizer_net_channels),
            overlap=detokenizer_overlap,
            img_size=img_size,
        )

    def forward(
        self,
        fields: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        del step_size
        if self.integrator is not None:
            raise RuntimeError("Internal integration is forbidden")
        if fields.ndim != 5 or fields.shape[1] != 1:
            raise ValueError(
                f"Expected fields [B,1,H,W,C], got {tuple(fields.shape)}"
            )
        if fields.shape[-1] != self.num_fields:
            raise ValueError(
                f"Expected {self.num_fields} fields, got {fields.shape[-1]}"
            )

        tokens = self.tokenizer(fields)
        token_dx = _as_batch_spacing(
            dx,
            batch_size=tokens.shape[0],
            device=tokens.device,
            name="dx",
        ) * self.patch_size[2]
        token_dy = _as_batch_spacing(
            dy,
            batch_size=tokens.shape[0],
            device=tokens.device,
            name="dy",
        ) * self.patch_size[1]
        # Exactly one physical coordinate encoding, before the first slice
        # assignment.  No slice-index coordinates are introduced.
        tokens = self.physical_fourier_ape(
            tokens,
            dx=token_dx,
            dy=token_dy,
        )
        for block in self.attention_blocks:
            tokens = block(tokens)
        delta = self.delta_decoder(tokens)
        return delta[:, -1:, ...]


class NativeGridPhysicalFourierResidualTransolver(
    NativeGridPhysicalPositionEarthDynamics
):
    """Baseline-compatible dynamic native-grid ``forward(x, static, metadata)``."""

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
        position_encoding: str = POSITION_ENCODING,
        space_ref: float = SPACE_REF_METERS,
        slice_num: int = SLICE_NUM,
        transolver_state_dim: int = 0,
        transolver_state_reduction: int = 4,
        fno_modes_height: int = 8,
        fno_modes_width: int = 8,
        fno_dropout: float = 0.0,
        local_kernel: int = 3,
        init_local_gamma: float = 0.5,
        init_global_gamma: float = 0.1,
        experiment: str = EXPERIMENT_NAME,
    ) -> None:
        # Reuse the tested parent forward/padding/crop implementation without
        # constructing its ordinary PhysicalPositionPhysicsTransformer.
        nn.Module.__init__(self)
        del experiment
        if not direct_derivative_output:
            raise ValueError("Direct single-channel delta output is required")
        if use_derivatives:
            raise ValueError("Input must be exactly h, rainfall, and terrain")
        if attention_mode != "spatial":
            raise ValueError("Transolver experiment fixes spatial physics attention")
        if position_encoding != POSITION_ENCODING:
            raise ValueError("Transolver experiment fixes Physical Fourier APE")
        if slice_num != SLICE_NUM:
            raise ValueError(f"--slice-num is fixed at {SLICE_NUM}")
        if patch_height < 1 or patch_width < 1:
            raise ValueError("Patch dimensions must be positive")

        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.input_channels = len(baseline.CHANNEL_NAMES)
        self.position_encoding = POSITION_ENCODING
        self.space_ref = float(space_ref)
        self.direct_derivative_output = True
        self.returns_delta = True
        patch_size = (1, self.patch_height, self.patch_width)
        spec = getattr(model_specs, model_size)()
        state_dim = (
            None if int(transolver_state_dim) <= 0 else int(transolver_state_dim)
        )
        self.backbone = PhysicalFourierResidualTransolverCore(
            num_fields=self.input_channels,
            hidden_dim=spec.hidden_dim,
            mlp_dim=spec.mlp_dim,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            patch_size=patch_size,
            img_size=(1, self.patch_height, self.patch_width),
            tokenizer_mode=tokenizer_mode,
            detokenizer_mode=detokenizer_mode,
            tokenizer_overlap=tokenizer_overlap,
            detokenizer_overlap=detokenizer_overlap,
            tokenizer_net_channels=spec.conv_channels,
            detokenizer_net_channels=spec.conv_channels,
            dropout=dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            slice_num=SLICE_NUM,
            state_dim=state_dim,
            state_reduction=transolver_state_reduction,
            fno_modes_height=fno_modes_height,
            fno_modes_width=fno_modes_width,
            fno_dropout=fno_dropout,
            local_kernel=local_kernel,
            init_local_gamma=init_local_gamma,
            init_global_gamma=init_global_gamma,
            space_ref=self.space_ref,
        )


def model_summary(
    model: NativeGridPhysicalFourierResidualTransolver,
) -> Dict[str, Any]:
    backbone = model.backbone
    return {
        "experiment": EXPERIMENT_NAME,
        "parameters_total": _parameter_count(model),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameters_by_component": {
            "shared_patch_tokenizer": _parameter_count(backbone.tokenizer),
            "physical_fourier_ape": _parameter_count(
                backbone.physical_fourier_ape
            ),
            "residual_conditioned_transolver": _parameter_count(
                backbone.attention_blocks
            ),
            "one_channel_delta_decoder": _parameter_count(
                backbone.delta_decoder
            ),
        },
        "transformer_layers": len(backbone.attention_blocks),
        "slice_num_shared": backbone.slice_num,
        "position_encoding": POSITION_ENCODING,
        "rope": False,
        "learned_fixed_grid_position": False,
        "integrator": None,
        "returns": "delta_h [B,1,H,W]",
        "external_update": "clamp(h[t-1] + delta_h, min=0)",
        "loss": "FP32 masked MSE",
        "normalization": "train_standard",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _BASE_BUILD_PARSER()
    parser.description = (
        "FloodCastBench residual-conditioned Transolver + MSE + "
        "Physical Fourier coordinate APE"
    )
    parser.add_argument(
        "--position-encoding",
        choices=(POSITION_ENCODING,),
        default=POSITION_ENCODING,
    )
    parser.add_argument("--slice-num", type=int, choices=(SLICE_NUM,), default=SLICE_NUM)
    parser.add_argument("--transolver-state-dim", type=int, default=0)
    parser.add_argument("--transolver-state-reduction", type=int, default=4)
    parser.add_argument("--fno-modes-height", type=int, default=8)
    parser.add_argument("--fno-modes-width", type=int, default=8)
    parser.add_argument("--fno-dropout", type=float, default=0.0)
    parser.add_argument("--local-kernel", type=int, default=3)
    parser.add_argument("--init-local-gamma", type=float, default=0.5)
    parser.add_argument("--init-global-gamma", type=float, default=0.1)
    parser.set_defaults(
        output_dir="results/FloodCastBench_Transolver_MSE_PhysicalFourierAPE",
        attention_mode="spatial",
        use_derivatives=False,
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
    if args.position_encoding != POSITION_ENCODING:
        raise ValueError("Physical Fourier APE is required")
    if args.attention_mode != "spatial":
        raise ValueError("Spatial attention mode is required")
    if args.use_derivatives:
        raise ValueError("Input must remain h, rainfall, terrain")
    if args.slice_num != SLICE_NUM:
        raise ValueError(f"slice_num must remain {SLICE_NUM}")
    config = _BASE_MODEL_CONFIG_FROM_ARGS(args)
    config.update(
        {
            "position_encoding": POSITION_ENCODING,
            "space_ref": SPACE_REF_METERS,
            "slice_num": SLICE_NUM,
            "transolver_state_dim": args.transolver_state_dim,
            "transolver_state_reduction": args.transolver_state_reduction,
            "fno_modes_height": args.fno_modes_height,
            "fno_modes_width": args.fno_modes_width,
            "fno_dropout": args.fno_dropout,
            "local_kernel": args.local_kernel,
            "init_local_gamma": args.init_local_gamma,
            "init_global_gamma": args.init_global_gamma,
            "experiment": EXPERIMENT_NAME,
        }
    )
    return config


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = NativeGridPhysicalFourierResidualTransolver(**dict(config)).to(device)
    backbone = model.backbone
    if backbone.integrator is not None or not model.returns_delta:
        raise AssertionError("Model must return delta_h with integrator=None")
    if any(
        block.attention.slice_num != SLICE_NUM
        for block in backbone.attention_blocks
    ):
        raise AssertionError(f"Every Transolver layer must use M={SLICE_NUM}")
    if sum(
        isinstance(module, PhysicalFourierCoordinateAPE)
        for module in model.modules()
    ) != 1:
        raise AssertionError("Exactly one Physical Fourier APE is required")
    if any(
        isinstance(module, (Physical2DRotaryEmbedding, AbsPositionalEmbedding))
        for module in model.modules()
    ):
        raise AssertionError("Transolver Fourier experiment forbids RoPE/absolute table")
    if _is_primary_process():
        print("MODEL_INFO", json.dumps(model_summary(model), ensure_ascii=False))
    return model


def main() -> None:
    baseline.build_parser = build_parser
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.main()


if __name__ == "__main__":
    main()
