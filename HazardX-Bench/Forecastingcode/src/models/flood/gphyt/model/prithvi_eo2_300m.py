"""TerraTorch Prithvi-EO-2.0-300M PT-FT FloodCastBench wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    extract_state_dict,
    load_pretrained_with_report,
)
from gphyt.model.official_import import sha256_file
from gphyt.model.prithvi_eo2_registry_adapter import (
    REGISTRY_NAME,
    convert_official_position_embedding,
    register_official_prithvi_backbone,
)
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/NASA-IMPACT/Prithvi-EO-2.0"
COMMIT = "758311280f935d4775c05b26d792768bf1fda841"
CHECKPOINT_REPOSITORY = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
CHECKPOINT_FILENAME = "Prithvi_EO_V2_300M.pt"
CHECKPOINT_SHA256 = "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7"


def _convert_patch_embed(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    required_source = (1024, 6, 1, 16, 16)
    required_target = (1024, 3, 1, 16, 16)
    if key != "patch_embed.proj.weight":
        raise RuntimeError(f"unexpected Prithvi patch converter key {key}")
    if tuple(source.shape) != required_source or tuple(target.shape) != required_target:
        raise RuntimeError(
            "Prithvi patch embedding semantic check failed: "
            f"source={tuple(source.shape)} target={tuple(target.shape)}"
        )
    mean = source.mean(dim=1, keepdim=True)
    return mean.repeat(1, 3, 1, 1, 1) * 2.0


def _normalize_prithvi_checkpoint(checkpoint: object) -> Dict[str, torch.Tensor]:
    state = extract_state_dict(checkpoint)
    normalized: Dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = str(original_key)
        for prefix in ("module.", "_orig_mod.", "_timm_module."):
            while key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("encoder."):
            key = key[len("encoder.") :]
        if key in normalized:
            raise KeyError(f"Prithvi checkpoint key collision after normalization: {key}")
        normalized[key] = value
    return normalized


class PrithviEO2300MPTFT(nn.Module):
    experiment_name = "Prithvi-EO2-300M-PT-FT"
    source_repository = REPOSITORY
    source_commit = COMMIT
    checkpoint_repository = CHECKPOINT_REPOSITORY
    checkpoint_sha256 = CHECKPOINT_SHA256
    patch_size = (1, 16, 16)

    def __init__(
        self,
        runtime_height: int,
        runtime_width: int,
        *,
        checkpoint_path: str,
        report_path: str,
    ) -> None:
        super().__init__()
        padded_height = int(runtime_height) + (-int(runtime_height)) % 16
        padded_width = int(runtime_width) + (-int(runtime_width)) % 16
        self.runtime_padded_shape = (padded_height, padded_width)
        registry, official_module = register_official_prithvi_backbone(
            checkpoint_path
        )
        if REGISTRY_NAME not in registry:
            raise AssertionError(f"missing registered backbone {REGISTRY_NAME}")
        # Raw physical channels deliberately avoid HLS band names and spectral
        # normalization. FloodCastBench normalization remains authoritative.
        self.backbone = registry.build(
            REGISTRY_NAME,
            pretrained=False,
            in_chans=3,
            num_frames=1,
            img_size=self.runtime_padded_shape,
            patch_size=(1, 16, 16),
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            coords_encoding=[],
        )
        if len(self.backbone.blocks) != 24 or self.backbone.embed_dim != 1024:
            raise RuntimeError("registry did not build standard Prithvi-EO2 300M")
        patch_weight = self.backbone.patch_embed.proj.weight
        if tuple(patch_weight.shape) != (1024, 3, 1, 16, 16):
            raise RuntimeError(
                f"unexpected Prithvi registry patch embed {tuple(patch_weight.shape)}"
            )
        self.recovery = nn.Sequential(
            nn.Conv2d(1024, 16 * 16, kernel_size=1),
            nn.PixelShuffle(16),
        )

        digest = sha256_file(checkpoint_path)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Prithvi checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}"
            )
        # The known official file is a PyTorch checkpoint; verify its published
        # hash before invoking PyTorch's restricted weights-only loader.
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "this torch version lacks weights_only checkpoint loading; use "
                "the Prithvi-supported torch 2.5 environment"
            ) from exc
        state = _normalize_prithvi_checkpoint(checkpoint)

        def convert_position(
            key: str, source: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            if key != "pos_embed":
                raise RuntimeError(f"unexpected Prithvi position converter key {key}")
            return convert_official_position_embedding(
                official_module,
                source,
                target,
                runtime_height=padded_height,
                runtime_width=padded_width,
            )

        converters = {
            "patch_embed.proj.weight": _convert_patch_embed,
            "pos_embed": convert_position,
        }
        self.loading_report: PretrainedLoadingReport = load_pretrained_with_report(
            self.backbone,
            state,
            checkpoint_sha256=digest,
            report_path=report_path,
            converters=converters,
            newly_initialized_prefixes=(),
            backbone_excluded_prefixes=(),
            minimum_backbone_ratio=0.99,
        )
        invalid_unexpected = [
            key
            for key in self.loading_report.unexpected
            if not key.startswith("decoder.")
        ]
        if invalid_unexpected:
            raise RuntimeError(
                f"unexpected non-decoder Prithvi checkpoint keys: {invalid_unexpected}"
            )
        if self.loading_report.missing or self.loading_report.shape_mismatch:
            raise RuntimeError(
                "Prithvi backbone load left missing/mismatched tensors; "
                f"report={report_path}"
            )
        block_keys = {
            key for key in self.backbone.state_dict() if key.startswith("blocks.")
        }
        if not block_keys.issubset(set(self.loading_report.exact_loaded)):
            missing_blocks = sorted(
                block_keys - set(self.loading_report.exact_loaded)
            )
            raise RuntimeError(
                f"Prithvi Transformer blocks were not loaded exactly: {missing_blocks}"
            )
        self.loading_report.newly_initialized.extend(
            ("recovery.0.bias", "recovery.0.weight")
        )
        self.loading_report.newly_initialized.sort()
        self.loading_report.write(report_path)

        buffers = dict(self.backbone.named_buffers())
        parameters = dict(self.backbone.named_parameters())
        if "pos_embed" not in buffers or "pos_embed" in parameters:
            raise AssertionError("Prithvi pos_embed must be a fixed buffer")
        pos_embed = buffers["pos_embed"]
        expected_pos_shape = (
            1,
            1 + (padded_height // 16) * (padded_width // 16),
            1024,
        )
        if tuple(pos_embed.shape) != expected_pos_shape:
            raise AssertionError(
                f"Prithvi runtime position shape {tuple(pos_embed.shape)} != "
                f"{expected_pos_shape}"
            )
        if not torch.isfinite(pos_embed).all() or not bool((pos_embed != 0).any()):
            raise FloatingPointError("Prithvi runtime position buffer is invalid")
        self.registry_source = "local_official_adapter"
        self.official_source_sha256 = (
            "9cee9a8cb17c3859d6365a7e58f723e0f4c08e3e29aa4b8e6523506db70321da"
        )
        assert_full_finetuning(self)

    def enable_activation_checkpointing(self) -> None:
        if not hasattr(self.backbone, "set_grad_checkpointing"):
            raise RuntimeError("installed TerraTorch backbone lacks activation checkpointing")
        self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, 16)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(
                f"Prithvi was built for {self.runtime_padded_shape}, got "
                f"{tuple(padded.shape[-2:])}; build per native grid, no resize is allowed"
            )
        official_input = padded.unsqueeze(2)  # [B,3,1,H,W]
        features = self.backbone.forward_features(
            official_input,
            temporal_coords=None,
            location_coords=None,
        )
        tokens = features[-1]
        # Official PrithviViT.forward_features explicitly prepends cls_token and
        # normalizes the final sequence before returning the last feature.
        patch_tokens = tokens[:, 1:, :]
        patch_height = padded.shape[-2] // 16
        patch_width = padded.shape[-1] // 16
        if patch_tokens.shape[1] != patch_height * patch_width:
            raise AssertionError(
                f"Prithvi special-token removal left {patch_tokens.shape[1]} tokens, "
                f"expected {patch_height * patch_width}"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            x.shape[0], 1024, patch_height, patch_width
        )
        prediction = self.recovery(feature_map)
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
