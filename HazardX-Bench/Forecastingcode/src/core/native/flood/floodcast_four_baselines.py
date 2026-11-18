"""FloodCastBench UNet/FNO output-formulation and loss baselines.

This module installs model and optimizer hooks into the validated native-grid
FloodCastBench lifecycle.  The shared lifecycle remains responsible for data
splits, task sampling, homogeneous micro-batches, macro-batch accumulation,
Markov pairing, normalized physical-state reconstruction, masks, validation,
testing, checkpoint selection and strict autoregressive rollout.  UNet and FNO
can independently select their core output (direct ``h_next`` or residual
``dh``) and training loss (masked relative L2 or masked MSE-matched Huber).
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
import floodcast_transolver_centered_dem_common as centered
from floodcast_official_transolver_dynamic import (
    OfficialTransolverStructuredMesh2D,
)
from gphyt.model import model_specs
from gphyt.model.tokenizer.tokenizer import Detokenizer
from gphyt.model.transformer.model import get_model as build_gphyt_model
from models.FNO import FNO2d
from models.Unet import UNet2d


MODEL_TYPES = ("UNet", "FNO", "Transolver", "GPhyT")
LOSS_RELATIVE_L2 = "relative_l2"
LOSS_HUBER = centered.LOSS_HUBER
LOSS_KINDS = (LOSS_RELATIVE_L2, LOSS_HUBER)
OUTPUT_NATIVE = "native"
OUTPUT_DIRECT = "direct"
OUTPUT_RESIDUAL = "residual"
OUTPUT_METHODS = (OUTPUT_NATIVE, OUTPUT_DIRECT, OUTPUT_RESIDUAL)
EXPERIMENT_NAME = "FloodCastBench_Four_Fair_Markov_Baselines"
DEFAULT_OUTPUT_DIR = (
    "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
    "FloodCastBench_four_fair_baselines"
)

_BASE_BUILD_PARSER = baseline.build_parser
_BASE_TRAIN = baseline.train
_BASE_RUN_SMOKE_TESTS = baseline.run_smoke_tests
_BASE_MASKED_LOSS = baseline.masked_loss
_BASE_CHECKPOINT_PAYLOAD = baseline.checkpoint_payload
_REAL_TORCH_ADAM = torch.optim.Adam

_ACTIVE_LOSS_KIND = LOSS_HUBER
_ACTIVE_OUTPUT_METHOD = OUTPUT_DIRECT
_ACTIVE_MODEL_TYPE = "UNet"


def _validate_common_inputs(
    x: torch.Tensor,
    static: Optional[torch.Tensor],
    metadata: Optional[Mapping[str, Any]],
) -> Tuple[int, int]:
    if x.ndim != 4 or x.shape[1] != 2:
        raise AssertionError(f"x must be [B,2,H,W], got {tuple(x.shape)}")
    if static is None or static.ndim != 4 or static.shape[1] != 1:
        raise AssertionError(
            f"static DEM must be [B,1,H,W], got "
            f"{None if static is None else tuple(static.shape)}"
        )
    if x.shape[0] != static.shape[0] or x.shape[-2:] != static.shape[-2:]:
        raise AssertionError(f"x/static mismatch: {x.shape} vs {static.shape}")
    if not torch.isfinite(x).all() or not torch.isfinite(static).all():
        raise FloatingPointError("Model input contains NaN/Inf")
    height, width = (int(value) for value in x.shape[-2:])
    if metadata is not None:
        for value in metadata.get("height", [height]):
            if int(value) != height:
                raise AssertionError("metadata height does not match x")
        for value in metadata.get("width", [width]):
            if int(value) != width:
                raise AssertionError("metadata width does not match x")
    return height, width


def _direct_next_to_trainer_delta(
    predicted_next: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    expected = (x.shape[0], 1, x.shape[-2], x.shape[-1])
    if tuple(predicted_next.shape) != expected:
        raise AssertionError(
            f"direct h_next shape {tuple(predicted_next.shape)} != {expected}"
        )
    if not torch.isfinite(predicted_next).all():
        raise FloatingPointError("Direct h_next prediction contains NaN/Inf")
    # The shared, already-validated trainer reconstructs h_next as h_t + dh.
    # Returning this algebraic adapter retains direct-next model semantics.
    return predicted_next - x[:, :1]


def _core_output_to_trainer_delta(
    core_output: torch.Tensor,
    x: torch.Tensor,
    output_method: str,
) -> torch.Tensor:
    expected = (x.shape[0], 1, x.shape[-2], x.shape[-1])
    if tuple(core_output.shape) != expected:
        raise AssertionError(
            f"core output shape {tuple(core_output.shape)} != {expected}"
        )
    if not torch.isfinite(core_output).all():
        raise FloatingPointError("Core output contains NaN/Inf")
    if output_method == OUTPUT_DIRECT:
        return _direct_next_to_trainer_delta(core_output, x)
    if output_method == OUTPUT_RESIDUAL:
        return core_output
    raise ValueError(f"Unsupported resolved output method {output_method!r}")


class NativeGridUNet(nn.Module):
    """Repository UNet2d with minimal multiple-of-16 pad/crop."""

    def __init__(
        self,
        init_features: int = 32,
        grid_type: str = "cartesian",
        output_method: str = OUTPUT_DIRECT,
    ) -> None:
        super().__init__()
        if output_method not in (OUTPUT_DIRECT, OUTPUT_RESIDUAL):
            raise ValueError(f"Invalid UNet output method {output_method!r}")
        self.output_method = output_method
        self.core = UNet2d(
            in_channels=3,
            out_channels=1,
            init_features=int(init_features),
            grid_type=grid_type,
        )
        self.direct_derivative_output = True
        self.core_prediction = (
            "h_next" if output_method == OUTPUT_DIRECT else "dh"
        )
        self.trainer_adapter = (
            "h_next_minus_h_t" if output_method == OUTPUT_DIRECT else "identity"
        )

    @staticmethod
    def padding_for(height: int, width: int) -> Tuple[int, int]:
        return (-int(height)) % 16, (-int(width)) % 16

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        height, width = _validate_common_inputs(x, static, metadata)
        fields = torch.cat((x, static), dim=1)
        pad_h, pad_w = self.padding_for(height, width)
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), value=0.0)
        # UNet2d's public layout is [B,H,W,T,C].
        fields = fields.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        predicted = self.core(fields)
        predicted = (
            predicted[:, :height, :width, 0, 0].unsqueeze(1).contiguous()
        )
        return _core_output_to_trainer_delta(predicted, x, self.output_method)


class DDPCompatibleSpectralConv2d(nn.Module):
    """Exact SpectralConv2d operation with real parameters for NCCL DDP.

    PyTorch 2.0 NCCL cannot broadcast or all-reduce ComplexFloat parameters.
    Splitting every complex coefficient into real and imaginary parameters is
    an exact reparameterization; the Fourier operator itself is unchanged.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.in_channels = int(source.in_channels)
        self.out_channels = int(source.out_channels)
        self.modes1 = int(source.modes1)
        self.modes2 = int(source.modes2)

        self.weights1_real = nn.Parameter(source.weights1.detach().real.clone())
        self.weights1_imag = nn.Parameter(source.weights1.detach().imag.clone())
        self.weights2_real = nn.Parameter(source.weights2.detach().real.clone())
        self.weights2_imag = nn.Parameter(source.weights2.detach().imag.clone())

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        height, width = inputs.shape[-2:]
        inputs_ft = torch.fft.rfft2(inputs)
        outputs_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=inputs.device,
        )
        weights1 = torch.complex(self.weights1_real, self.weights1_imag)
        weights2 = torch.complex(self.weights2_real, self.weights2_imag)
        outputs_ft[:, :, : self.modes1, : self.modes2] = self._multiply(
            inputs_ft[:, :, : self.modes1, : self.modes2], weights1
        )
        outputs_ft[:, :, -self.modes1 :, : self.modes2] = self._multiply(
            inputs_ft[:, :, -self.modes1 :, : self.modes2], weights2
        )
        return torch.fft.irfft2(outputs_ft, s=(height, width))


class NativeGridFNO(nn.Module):
    """Repository FNO2d evaluated directly at every native H/W."""

    def __init__(
        self,
        modes_height: int = 12,
        modes_width: int = 12,
        width: int = 20,
        grid_type: str = "cartesian",
        output_method: str = OUTPUT_DIRECT,
    ) -> None:
        super().__init__()
        if output_method not in (OUTPUT_DIRECT, OUTPUT_RESIDUAL):
            raise ValueError(f"Invalid FNO output method {output_method!r}")
        self.output_method = output_method
        self.core = FNO2d(
            num_channels=3,
            modes1=int(modes_height),
            modes2=int(modes_width),
            width=int(width),
            initial_step=1,
            grid_type=grid_type,
        )
        for name in ("conv0", "conv1", "conv2", "conv3"):
            setattr(
                self.core,
                name,
                DDPCompatibleSpectralConv2d(getattr(self.core, name)),
            )
        self.direct_derivative_output = True
        self.core_prediction = (
            "h_next" if output_method == OUTPUT_DIRECT else "dh"
        )
        self.trainer_adapter = (
            "h_next_minus_h_t" if output_method == OUTPUT_DIRECT else "identity"
        )

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        height, width = _validate_common_inputs(x, static, metadata)
        fields = torch.cat((x, static), dim=1)
        fields = fields.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        # Native non-power-of-two cuFFT is unsupported in FP16 on CUDA.
        # Keep the repository FNO core in FP32; no spatial resize is applied.
        with torch.cuda.amp.autocast(enabled=False):
            predicted = self.core(fields.float())
        predicted = predicted[:, :height, :width, 0, 0].unsqueeze(1).contiguous()
        return _core_output_to_trainer_delta(predicted, x, self.output_method)


class NativeGridOfficialTransolver(nn.Module):
    """Official Structured-Mesh-2D Transolver with FloodCastBench I/O."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        slice_num: int = 32,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        ref: int = 8,
        unified_pos: bool = True,
    ) -> None:
        super().__init__()
        self.core = OfficialTransolverStructuredMesh2D(
            space_dim=2,
            n_layers=int(num_layers),
            n_hidden=int(hidden_dim),
            dropout=float(dropout),
            n_head=int(num_heads),
            mlp_ratio=int(mlp_ratio),
            fun_dim=3,
            out_dim=1,
            slice_num=int(slice_num),
            ref=int(ref),
            unified_pos=bool(unified_pos),
        )
        self.direct_derivative_output = True
        self.core_prediction = "h_next"
        self.trainer_adapter = "h_next_minus_h_t"

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        height, width = _validate_common_inputs(x, static, metadata)
        fields = torch.cat((x, static), dim=1)
        fields = fields.permute(0, 2, 3, 1).reshape(
            x.shape[0], height * width, 3
        )
        predicted = self.core(fields, height=height, width=width)
        predicted = predicted.reshape(x.shape[0], height, width, 1)
        predicted = predicted.permute(0, 3, 1, 2).contiguous()
        return _direct_next_to_trainer_delta(predicted, x)


class CroppedAbsolutePositionEmbedding(nn.Module):
    """Use one learned GPhyT absolute table on every smaller native patch grid."""

    def __init__(self, position_parameter: nn.Parameter) -> None:
        super().__init__()
        self.pe = position_parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, height, width, _ = x.shape
        if (
            time > self.pe.shape[1]
            or height > self.pe.shape[2]
            or width > self.pe.shape[3]
        ):
            raise ValueError(
                "Native GPhyT patch grid exceeds configured absolute table: "
                f"input={(time, height, width)}, max={tuple(self.pe.shape[1:4])}"
            )
        return x + self.pe[:, :time, :height, :width, :]


class NativeGridGPhyT(nn.Module):
    """Original GPhyT core, spatial attention only, preserving direct dh."""

    def __init__(
        self,
        patch_height: int = 16,
        patch_width: int = 16,
        model_size: str = "GPT_S",
        tokenizer_mode: str = "linear",
        detokenizer_mode: str = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        use_derivatives: bool = True,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        max_patches_height: int = 51,
        max_patches_width: int = 34,
    ) -> None:
        super().__init__()
        if min(patch_height, patch_width, max_patches_height, max_patches_width) < 1:
            raise ValueError("GPhyT patch/table dimensions must be positive")
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.max_patches_height = int(max_patches_height)
        self.max_patches_width = int(max_patches_width)
        patch_size = (1, self.patch_height, self.patch_width)
        img_size = (
            1,
            self.patch_height * self.max_patches_height,
            self.patch_width * self.max_patches_width,
        )
        model_config = {
            "architecture": "gphyt",
            "img_size": img_size,
            "transformer": {
                "input_channels": 3,
                "model_size": model_size,
                "att_mode": "spatial",
                "dropout": float(dropout),
                "pos_enc_mode": "absolute",
                "patch_size": list(patch_size),
                "stochastic_depth_rate": float(stochastic_depth_rate),
                "use_derivatives": bool(use_derivatives),
                # Preserve current FloodCastBench GPhyT dh formulation.
                "integrator": None,
            },
            "tokenizer": {
                "tokenizer_mode": tokenizer_mode,
                "detokenizer_mode": detokenizer_mode,
                "tokenizer_overlap": int(tokenizer_overlap),
                "detokenizer_overlap": int(detokenizer_overlap),
            },
        }
        self.backbone = build_gphyt_model(model_config)
        if self.backbone.init_pos_encodings is None:
            raise AssertionError("GPhyT must use its absolute position encoding")
        original_position = self.backbone.init_pos_encodings.pe
        self.backbone.init_pos_encodings = CroppedAbsolutePositionEmbedding(
            original_position
        )
        spec = getattr(model_specs, model_size)()
        self.backbone.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=spec.hidden_dim,
            out_channels=1,
            mode=detokenizer_mode,
            conv_net_channels=spec.conv_channels,
            overlap=int(detokenizer_overlap),
            img_size=img_size,
        )
        self.direct_derivative_output = True
        self.core_prediction = "dh"
        self.trainer_adapter = "identity"

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        height, width = _validate_common_inputs(x, static, metadata)
        fields = torch.cat((x, static), dim=1)
        pad_h = (-height) % self.patch_height
        pad_w = (-width) % self.patch_width
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), value=0.0)
        padded_height, padded_width = (int(value) for value in fields.shape[-2:])
        if padded_height // self.patch_height > self.max_patches_height:
            raise ValueError("GPhyT native height exceeds absolute-position table")
        if padded_width // self.patch_width > self.max_patches_width:
            raise ValueError("GPhyT native width exceeds absolute-position table")
        fields = fields.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        derivative = self.backbone(fields)
        dh = derivative[:, 0, :height, :width, :].permute(0, 3, 1, 2)
        expected = (x.shape[0], 1, height, width)
        if tuple(dh.shape) != expected:
            raise AssertionError(f"GPhyT dh shape {dh.shape} != {expected}")
        if not torch.isfinite(dh).all():
            raise FloatingPointError("GPhyT dh contains NaN/Inf")
        return dh.contiguous()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _parse_output_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": OUTPUT_DIRECT,
        "h_next": OUTPUT_DIRECT,
        "ht+1": OUTPUT_DIRECT,
        "direct": OUTPUT_DIRECT,
        "dh": OUTPUT_RESIDUAL,
        "residual": OUTPUT_RESIDUAL,
        "native": OUTPUT_NATIVE,
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "output_method must be native/direct/residual; None aliases direct"
        )
    return aliases[normalized]


def _native_output_method(model_type: str) -> str:
    return OUTPUT_RESIDUAL if model_type == "GPhyT" else OUTPUT_DIRECT


def _resolve_output_method(model_type: str, requested: str) -> str:
    native = _native_output_method(model_type)
    if requested == OUTPUT_NATIVE:
        return native
    if model_type in ("UNet", "FNO"):
        return requested
    if requested != native:
        raise ValueError(
            f"{model_type} keeps its native {native!r} output in this entry point"
        )
    return requested


def masked_relative_l2_lploss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked NaN-safe counterpart of utils2.LpLoss.rel for one channel."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise AssertionError(
            f"loss shape mismatch pred={prediction.shape}, "
            f"target={target.shape}, mask={mask.shape}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"relative-L2 eps must be finite and positive, got {eps}")
    prediction_f = prediction.float()
    target_f = target.float()
    valid = (
        mask.bool()
        & torch.isfinite(prediction_f)
        & torch.isfinite(target_f)
    )
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("At least one sample has zero valid target pixels")
    difference = torch.where(
        valid, prediction_f - target_f, torch.zeros_like(prediction_f)
    )
    target_valid = torch.where(valid, target_f, torch.zeros_like(target_f))
    difference_norm = torch.linalg.vector_norm(difference.flatten(1), dim=1)
    target_norm = torch.linalg.vector_norm(target_valid.flatten(1), dim=1)
    per_sample = difference_norm / target_norm.clamp_min(float(eps))
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked relative-L2 loss")
    return loss


def _selected_masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    if kind != "mse":
        raise ValueError(
            "The shared trainer must request its reserved MSE loss slot, "
            f"got {kind!r}"
        )
    if _ACTIVE_LOSS_KIND == LOSS_RELATIVE_L2:
        return masked_relative_l2_lploss(
            prediction, target, mask, eps=eps
        )
    if _ACTIVE_LOSS_KIND == LOSS_HUBER:
        return centered._huber_masked_loss(
            prediction, target, mask, kind=kind, eps=eps
        )
    raise RuntimeError(f"Unsupported active loss {_ACTIVE_LOSS_KIND!r}")


def _selected_checkpoint_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    if _ACTIVE_LOSS_KIND == LOSS_HUBER:
        payload = centered._huber_checkpoint_payload(*args, **kwargs)
    elif _ACTIVE_LOSS_KIND == LOSS_RELATIVE_L2:
        payload = _BASE_CHECKPOINT_PAYLOAD(*args, **kwargs)
        payload["loss_kind"] = LOSS_RELATIVE_L2
        payload["relative_l2_definition"] = (
            "per-sample masked ||prediction-target||_2 / "
            "max(||target||_2, 1e-8), followed by sample mean"
        )
    else:
        raise RuntimeError(f"Unsupported active loss {_ACTIVE_LOSS_KIND!r}")
    model_config = payload["model_config"]
    model_type = str(model_config["model_type"])
    payload["output_method"] = model_config.get(
        "output_method", _native_output_method(model_type)
    )
    return payload


def _selected_load_checkpoint_file(
    path: Any,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = centered._load_checkpoint_file(path, device)
    model_config = checkpoint["model_config"]
    saved_model_type = str(model_config["model_type"])
    if saved_model_type != _ACTIVE_MODEL_TYPE:
        raise RuntimeError(
            f"Checkpoint model {saved_model_type!r} != running model "
            f"{_ACTIVE_MODEL_TYPE!r}"
        )
    saved_output = model_config.get(
        "output_method", _native_output_method(saved_model_type)
    )
    if saved_output != _ACTIVE_OUTPUT_METHOD:
        raise RuntimeError(
            f"Checkpoint output method {saved_output!r} != running output "
            f"method {_ACTIVE_OUTPUT_METHOD!r}"
        )
    # The legacy lifecycle internally reserves the name ``mse`` for the loss
    # hook.  Preserve real metadata above, but present that compatibility view
    # to resume logic without mutating the checkpoint file on disk.
    checkpoint = dict(checkpoint)
    checkpoint["run_config"] = dict(checkpoint.get("run_config", {}))
    checkpoint["run_config"]["loss"] = "mse"
    return checkpoint


def _selected_load_eval_model(
    checkpoint_path: Any,
    device: torch.device,
) -> Tuple[nn.Module, baseline.ChannelNormalizer, Dict[str, Any]]:
    checkpoint = baseline.load_checkpoint_file(checkpoint_path, device)
    model = baseline.build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = baseline.ChannelNormalizer.from_state_dict(
        checkpoint["normalization"]
    )
    if normalizer.mode != "train_standard":
        raise ValueError("Evaluation requires train_standard normalization")
    saved_loss = checkpoint.get(
        "loss_kind", checkpoint.get("run_config", {}).get("loss")
    )
    if saved_loss != _ACTIVE_LOSS_KIND:
        raise ValueError(
            f"Checkpoint loss {saved_loss!r} != running loss "
            f"{_ACTIVE_LOSS_KIND!r}"
        )
    return model, normalizer, checkpoint


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common = {
        "model_type": args.model_type,
        # Kept for the shared checkpoint compatibility guard.  Every adapter
        # returns the trainer's required single-channel dh interface.
        "direct_derivative_output": True,
    }
    if args.model_type == "UNet":
        common.update(
            init_features=args.unet_init_features,
            grid_type=args.unet_grid_type,
            output_method=args.output_method,
        )
    elif args.model_type == "FNO":
        common.update(
            modes_height=args.fno_modes_height,
            modes_width=args.fno_modes_width,
            width=args.fno_width,
            grid_type=args.fno_grid_type,
            output_method=args.output_method,
        )
    elif args.model_type == "Transolver":
        common.update(
            hidden_dim=args.transolver_hidden_dim,
            num_layers=args.transolver_num_layers,
            num_heads=args.transolver_num_heads,
            slice_num=args.transolver_slice_num,
            mlp_ratio=args.transolver_mlp_ratio,
            dropout=args.transolver_dropout,
            ref=args.transolver_ref,
            unified_pos=True,
        )
    elif args.model_type == "GPhyT":
        common.update(
            patch_height=args.gphyt_patch_height,
            patch_width=args.gphyt_patch_width,
            model_size=args.gphyt_model_size,
            tokenizer_mode=args.gphyt_tokenizer_mode,
            detokenizer_mode=args.gphyt_detokenizer_mode,
            tokenizer_overlap=args.gphyt_tokenizer_overlap,
            detokenizer_overlap=args.gphyt_detokenizer_overlap,
            use_derivatives=args.gphyt_use_derivatives,
            dropout=args.gphyt_dropout,
            stochastic_depth_rate=args.gphyt_stochastic_depth_rate,
            max_patches_height=args.gphyt_max_patches_height,
            max_patches_width=args.gphyt_max_patches_width,
        )
    else:
        raise ValueError(args.model_type)
    return common


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    values = dict(config)
    model_type = values.pop("model_type")
    values.pop("direct_derivative_output", None)
    constructors = {
        "UNet": NativeGridUNet,
        "FNO": NativeGridFNO,
        "Transolver": NativeGridOfficialTransolver,
        "GPhyT": NativeGridGPhyT,
    }
    if model_type not in constructors:
        raise ValueError(f"Unknown model_type {model_type!r}")
    model = constructors[model_type](**values).to(device)
    info = {
        "experiment": EXPERIMENT_NAME,
        "model_type": model_type,
        "loss": _ACTIVE_LOSS_KIND,
        "output_method": getattr(
            model, "output_method", _native_output_method(model_type)
        ),
        "parameters_total": parameter_count(model),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "core_prediction": getattr(model, "core_prediction", None),
        "trainer_adapter": getattr(model, "trainer_adapter", None),
        "input": ["h_norm", "rainfall_norm", "terrain_centered_global_scaled"],
        "trainer_output": "dh [B,1,H,W]",
    }
    print("MODEL_INFO", json.dumps(info, ensure_ascii=False))
    return model


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(dest)


def build_parser() -> argparse.ArgumentParser:
    parser = _BASE_BUILD_PARSER()
    parser.description = (
        "FloodCastBench native-grid Markov baselines: same data/splits/"
        "sampling/preprocessing/evaluation, with selectable UNet/FNO loss "
        "and output formulation"
    )
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="UNet")
    parser.add_argument(
        "--output_method",
        "--output-method",
        type=_parse_output_method,
        choices=OUTPUT_METHODS,
        default=OUTPUT_NATIVE,
        help=(
            "Model core output: direct h_next or residual dh; native preserves "
            "the previous per-architecture behavior. None aliases direct."
        ),
    )
    parser.add_argument("--dem-root", default=dem_common.DEM_ROOT_DEFAULT)
    parser.add_argument(
        "--huber-delta-m",
        type=float,
        default=centered.HUBER_DELTA_M_DEFAULT,
        help="Physical Mask Huber transition in metres",
    )

    parser.add_argument("--unet-init-features", type=int, default=32)
    parser.add_argument("--unet-grid-type", default="cartesian")
    parser.add_argument("--fno-modes-height", type=int, default=12)
    parser.add_argument("--fno-modes-width", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=20)
    parser.add_argument("--fno-grid-type", default="cartesian")

    # Official Transolver_NS.sh / exp_ns.py defaults.
    parser.add_argument("--transolver-hidden-dim", type=int, default=256)
    parser.add_argument("--transolver-num-layers", type=int, default=8)
    parser.add_argument("--transolver-num-heads", type=int, default=8)
    parser.add_argument("--transolver-slice-num", type=int, default=32)
    parser.add_argument("--transolver-mlp-ratio", type=int, default=1)
    parser.add_argument("--transolver-dropout", type=float, default=0.0)
    parser.add_argument("--transolver-ref", type=int, default=8)

    # GPhyT config.yaml architecture defaults, with spatial attention fixed.
    parser.add_argument(
        "--gphyt-model-size",
        choices=("GPT_S", "GPT_M", "GPT_L", "GPT_XL"),
        default="GPT_S",
    )
    parser.add_argument("--gphyt-patch-height", type=int, default=16)
    parser.add_argument("--gphyt-patch-width", type=int, default=16)
    parser.add_argument("--gphyt-tokenizer-mode", default="linear")
    parser.add_argument("--gphyt-detokenizer-mode", default="linear")
    parser.add_argument("--gphyt-tokenizer-overlap", type=int, default=0)
    parser.add_argument("--gphyt-detokenizer-overlap", type=int, default=0)
    parser.add_argument(
        "--gphyt-use-derivatives", type=baseline.str2bool, default=True
    )
    parser.add_argument("--gphyt-dropout", type=float, default=0.0)
    parser.add_argument("--gphyt-stochastic-depth-rate", type=float, default=0.0)
    parser.add_argument("--gphyt-max-patches-height", type=int, default=51)
    parser.add_argument("--gphyt-max-patches-width", type=int, default=34)

    loss_action = _action(parser, "loss")
    loss_action.choices = LOSS_KINDS
    loss_action.default = LOSS_HUBER
    lr_action = _action(parser, "learning_rate")
    lr_action.default = None
    wd_action = _action(parser, "weight_decay")
    wd_action.default = None
    scheduler_action = _action(parser, "scheduler")
    scheduler_action.choices = (
        "auto",
        "cosine",
        "step",
        "none",
        "onecycle",
        "gphyt_warmup_cosine",
    )
    scheduler_action.default = "auto"
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR,
        normalization="train_standard",
        loss=LOSS_HUBER,
        attention_mode="spatial",
    )
    return parser


def _optimizer_defaults(model_type: str) -> Dict[str, Any]:
    if model_type == "Transolver":
        return {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "scheduler": "onecycle",
        }
    if model_type == "GPhyT":
        return {
            "name": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "scheduler": "gphyt_warmup_cosine",
        }
    return {
        "name": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    }


def resolve_optimization_args(args: argparse.Namespace) -> None:
    defaults = _optimizer_defaults(args.model_type)
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.scheduler == "auto":
        args.scheduler = defaults["scheduler"]
    args.optimizer_name = defaults["name"]
    if args.model_type == "FNO" and args.amp:
        args.amp = False
        args.amp_disabled_reason = (
            "Native non-power-of-two CUDA FFTs are unsupported in FP16"
        )
    elif args.model_type == "Transolver" and args.amp:
        args.amp = False
        args.amp_disabled_reason = (
            "Official Transolver uses FP32; native-grid slice reductions can "
            "overflow under FP16 autocast"
        )


def validate_args(args: argparse.Namespace) -> None:
    global _ACTIVE_LOSS_KIND, _ACTIVE_OUTPUT_METHOD, _ACTIVE_MODEL_TYPE
    selected_loss = args.loss
    if selected_loss not in LOSS_KINDS:
        raise ValueError(f"Unsupported loss {selected_loss!r}")
    resolved_output = _resolve_output_method(args.model_type, args.output_method)
    if args.model_type not in ("UNet", "FNO") and selected_loss != LOSS_HUBER:
        raise ValueError(
            "relative_l2 is defined here only for the requested UNet/FNO baselines"
        )
    if selected_loss == LOSS_HUBER and (
        not math.isfinite(args.huber_delta_m) or args.huber_delta_m <= 0
    ):
        raise ValueError("--huber-delta-m must be finite and positive")

    # The validated lifecycle hard-codes its internal loss hook name as MSE.
    # Present that compatibility value only during its argument checks.
    args.output_method = resolved_output
    args.loss = "mse"
    try:
        centered._BASE_VALIDATE_ARGS(args)
    finally:
        args.loss = selected_loss
    resolve_optimization_args(args)

    _ACTIVE_LOSS_KIND = selected_loss
    _ACTIVE_OUTPUT_METHOD = resolved_output
    _ACTIVE_MODEL_TYPE = args.model_type
    centered._ACTIVE_LOSS_KIND = selected_loss
    centered._ACTIVE_HUBER_DELTA_M = (
        float(args.huber_delta_m) if selected_loss == LOSS_HUBER else None
    )
    centered._ACTIVE_TRAIN_WATER_STD = None
    centered._ACTIVE_HUBER_DELTA_NORM = None
    store = centered._configure_store(args.dem_root)
    if store.static_channels != 1:
        raise AssertionError("Centered-global-scaled DEM must have one channel")

    positive = {
        "unet_init_features": args.unet_init_features,
        "fno_modes_height": args.fno_modes_height,
        "fno_modes_width": args.fno_modes_width,
        "fno_width": args.fno_width,
        "transolver_hidden_dim": args.transolver_hidden_dim,
        "transolver_num_layers": args.transolver_num_layers,
        "transolver_num_heads": args.transolver_num_heads,
        "transolver_slice_num": args.transolver_slice_num,
        "transolver_mlp_ratio": args.transolver_mlp_ratio,
        "transolver_ref": args.transolver_ref,
        "gphyt_patch_height": args.gphyt_patch_height,
        "gphyt_patch_width": args.gphyt_patch_width,
        "gphyt_max_patches_height": args.gphyt_max_patches_height,
        "gphyt_max_patches_width": args.gphyt_max_patches_width,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Architecture arguments must be positive: {invalid}")
    if args.transolver_hidden_dim % args.transolver_num_heads:
        raise ValueError("Transolver hidden dim must be divisible by heads")
    if args.model_type == "GPhyT" and not args.gphyt_use_derivatives:
        raise ValueError("GPhyT config.yaml requires --gphyt-use-derivatives true")


def build_optimizer(
    parameters: Iterable[nn.Parameter], args: argparse.Namespace
) -> torch.optim.Optimizer:
    if args.optimizer_name == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    if args.optimizer_name == "Adam":
        return _REAL_TORCH_ADAM(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            # PyTorch 2.0 foreach Adam cannot combine FNO complex parameters
            # with weight decay; the scalar path uses the same Adam update.
            foreach=False,
        )
    raise ValueError(args.optimizer_name)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    total_updates = max(1, int(args.epochs) * int(args.updates_per_epoch))
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=args.updates_per_epoch,
        )
    if args.scheduler == "gphyt_warmup_cosine":
        warmup_updates = min(5000, total_updates)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        if total_updates <= warmup_updates:
            return warmup
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_updates - warmup_updates,
            eta_min=args.learning_rate * 0.01,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_updates],
        )
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_updates,
            gamma=args.scheduler_gamma,
        )
    if args.scheduler == "none":
        return None
    raise ValueError(args.scheduler)


def optimizer_summary(args: argparse.Namespace) -> Dict[str, Any]:
    summary = {
        "model_type": args.model_type,
        "loss": args.loss,
        "output_method": args.output_method,
        "optimizer": args.optimizer_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
    }
    if args.scheduler == "onecycle":
        summary["max_lr"] = args.learning_rate
    elif args.scheduler == "gphyt_warmup_cosine":
        summary.update(warmup_updates=5000, end_factor=0.01)
    return summary


def _with_architecture_optimizer(function: Any, args: argparse.Namespace, *call: Any):
    original_adam = torch.optim.Adam

    def optimizer_factory(parameters: Iterable[nn.Parameter], *unused: Any, **ignored: Any):
        del unused, ignored
        optimizer = build_optimizer(parameters, args)
        print("OPTIMIZER_INFO", json.dumps(optimizer_summary(args)))
        return optimizer

    torch.optim.Adam = optimizer_factory
    try:
        return function(*call)
    finally:
        torch.optim.Adam = original_adam


def train_with_architecture_optimizer(
    args: argparse.Namespace,
    specs: Mapping[str, baseline.DomainSpec],
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
):
    return _with_architecture_optimizer(
        _BASE_TRAIN, args, args, specs, device, distributed, rank, world_size
    )


def smoke_with_architecture_optimizer(
    args: argparse.Namespace,
    specs: Mapping[str, baseline.DomainSpec],
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    print("SMOKE optimizer", json.dumps(optimizer_summary(args)))
    _with_architecture_optimizer(
        _BASE_RUN_SMOKE_TESTS, args, args, specs, device, rank, world_size
    )


def run_experiment() -> None:
    """Install isolated hooks and enter the original benchmark lifecycle."""
    centered._ACTIVE_LOSS_KIND = LOSS_HUBER
    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.compute_training_normalizer = centered._compute_training_normalizer
    baseline.FloodDomainDataset = dem_common.DemFeatureFloodDomainDataset
    baseline.rollout_sequence = dem_common.rollout_sequence
    baseline.describe_dataset = centered._describe_dataset
    baseline.load_checkpoint_file = _selected_load_checkpoint_file
    baseline.checkpoint_payload = _selected_checkpoint_payload
    baseline.load_eval_model = _selected_load_eval_model
    baseline.masked_loss = _selected_masked_loss
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.make_scheduler = make_scheduler
    baseline.train = train_with_architecture_optimizer
    baseline.run_smoke_tests = smoke_with_architecture_optimizer
    baseline.main()

