"""Official GPhyT-S pretrained/fine-tuned FloodCastBench wrapper."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    load_pretrained_with_report,
    load_torch_checkpoint_state,
)
from gphyt.model.official_import import sha256_file, verify_source_file
from gphyt.model.dynamic_position_adapter import install_gphyt_dynamic_absolute_position
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/FloWsnr/General-Physics-Transformer"
COMMIT = "df1527c4773ababc6f8af0d117fb5803fa3bf74b"
SOURCE_RELATIVE_PATH = "gphyt/models/transformer/model.py"
SOURCE_SHA256 = "088a681177c224fabdebe38718ec07fd1ac1baea3c9ddadf6c761921d8c1169d"
CHECKPOINT_REPOSITORY = "flwi/Physics-Foundation-Model"
CHECKPOINT_FILENAME = "gphyt-S.pth"
CHECKPOINT_SHA256 = "a7beab39e6e9527d44a7a43a7f2e95a5a22aaa69f5932cc9809f9dabb0b2e6c0"


def _import_exact_gphyt(official_root: str):
    import gphyt as local_gphyt

    package_path = str(Path(official_root).expanduser().resolve() / "gphyt")
    if package_path not in local_gphyt.__path__:
        local_gphyt.__path__.append(package_path)
    module = importlib.import_module("gphyt.models.transformer.model")
    actual = Path(module.__file__).resolve()
    expected = Path(official_root).expanduser().resolve() / SOURCE_RELATIVE_PATH
    if actual != expected:
        raise ImportError(f"GPhyT module collision: imported {actual}, expected {expected}")
    return module


def _convert_derivative_filter(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.shape[0] != 5 or target.shape[0] != 3 or tuple(source.shape[1:]) != tuple(target.shape[1:]):
        raise RuntimeError(
            f"unconfirmed GPhyT derivative filter layout for {key}: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )
    reference = source[:1].expand_as(source)
    if not torch.equal(source, reference):
        raise RuntimeError(f"GPhyT derivative filter channels are not identical for {key}")
    return source[:3]


def _convert_tokenizer(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.ndim != 5 or target.ndim != 5 or source.shape[1] != 20 or target.shape[1] != 12:
        raise RuntimeError(
            f"unconfirmed GPhyT tokenizer layout for {key}: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )
    if tuple(source.shape[0:1] + source.shape[2:]) != tuple(target.shape[0:1] + target.shape[2:]):
        raise RuntimeError("GPhyT tokenizer differs outside the physical-channel axis")
    # differentiate() concatenates [u, dt, dh, dw].  Average the five source
    # variables independently within each derivative group, then replicate to
    # FloodCastBench's [h, rainfall, DEM].
    grouped = source.reshape(source.shape[0], 4, 5, *source.shape[2:])
    grouped = grouped.mean(dim=2, keepdim=True).repeat(1, 1, 3, 1, 1, 1)
    return grouped.reshape_as(target)


def _convert_detokenizer_weight(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.ndim != 5 or target.ndim != 5 or source.shape[1] != 5 or target.shape[1] != 3:
        raise RuntimeError(
            f"unconfirmed GPhyT detokenizer weight layout for {key}: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )
    mean = source.mean(dim=1, keepdim=True)
    return mean.repeat(1, 3, 1, 1, 1)


def _convert_detokenizer_bias(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if tuple(source.shape) != (5,) or tuple(target.shape) != (3,):
        raise RuntimeError(f"unconfirmed GPhyT detokenizer bias layout for {key}")
    return source.mean().expand_as(target).clone()


def _interpolate_absolute_position(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.ndim != 5 or target.ndim != 5 or source.shape[0] != 1 or target.shape[0] != 1:
        raise RuntimeError(f"unconfirmed GPhyT absolute-position tensor for {key}")
    if source.shape[-1] != target.shape[-1]:
        raise RuntimeError("GPhyT position embedding width changed")
    channels_first = source.permute(0, 4, 1, 2, 3).float()
    resized = F.interpolate(
        channels_first,
        size=target.shape[1:4],
        mode="trilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 4, 1).to(dtype=target.dtype)


class GPhyTSPTFT(nn.Module):
    experiment_name = "GPhyT-S-PT-FT"
    source_repository = REPOSITORY
    source_commit = COMMIT
    checkpoint_repository = CHECKPOINT_REPOSITORY
    checkpoint_sha256 = CHECKPOINT_SHA256
    patch_size = (1, 16, 16)
    dt = 1.0

    def __init__(
        self,
        runtime_height: int,
        runtime_width: int,
        *,
        official_root: str,
        checkpoint_path: str,
        report_path: str,
    ) -> None:
        super().__init__()
        verify_source_file(
            official_root,
            SOURCE_RELATIVE_PATH,
            SOURCE_SHA256,
            repository=REPOSITORY,
            commit=COMMIT,
        )
        module = _import_exact_gphyt(official_root)
        padded_height = int(runtime_height) + (-int(runtime_height)) % 16
        padded_width = int(runtime_width) + (-int(runtime_width)) % 16
        self.runtime_padded_shape = (padded_height, padded_width)
        self.official_config = {
            "architecture": "gphyt",
            "img_size": (1, padded_height, padded_width),
            "transformer": {
                "input_channels": 3,
                "model_size": "GPT_S",
                "att_mode": "full",
                "dropout": 0.0,
                "pos_enc_mode": "absolute",
                "patch_size": [1, 16, 16],
                "stochastic_depth_rate": 0.0,
                "use_derivatives": True,
                "integrator": "Euler",
            },
            "tokenizer": {
                "tokenizer_mode": "linear",
                "detokenizer_mode": "linear",
                "tokenizer_overlap": 0,
                "detokenizer_overlap": 0,
            },
        }
        self.backbone = module.get_model(self.official_config)
        install_gphyt_dynamic_absolute_position(self.backbone)

        digest = sha256_file(checkpoint_path)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"GPhyT-S checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}"
            )
        state, loaded_digest = load_torch_checkpoint_state(checkpoint_path)
        if digest != loaded_digest:
            raise AssertionError("GPhyT checkpoint changed while loading")
        converters = {
            "derivatives.dt_filter": _convert_derivative_filter,
            "derivatives.dh_filter": _convert_derivative_filter,
            "derivatives.dw_filter": _convert_derivative_filter,
            "tokenizer.tokenizer.to_patch_embedding.1.weight": _convert_tokenizer,
            "detokenizer.detokenizer.from_patch_embedding.1.weight": _convert_detokenizer_weight,
            "detokenizer.detokenizer.from_patch_embedding.1.bias": _convert_detokenizer_bias,
        }
        self.loading_report: PretrainedLoadingReport = load_pretrained_with_report(
            self.backbone,
            state,
            checkpoint_sha256=digest,
            report_path=report_path,
            converters=converters,
            newly_initialized_prefixes=(),
            backbone_excluded_prefixes=("tokenizer", "detokenizer"),
            minimum_backbone_ratio=0.90,
        )
        assert_full_finetuning(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, 16)
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        # The cited commit's forward implements exactly u_next = u + derivative.
        predicted_fields = self.backbone(official_input, step_size=self.dt)
        if predicted_fields.ndim != 5 or predicted_fields.shape[-1] != 3:
            raise AssertionError(
                f"GPhyT returned unexpected shape {tuple(predicted_fields.shape)}"
            )
        prediction = predicted_fields[:, 0, :, :, 0].unsqueeze(1)
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
