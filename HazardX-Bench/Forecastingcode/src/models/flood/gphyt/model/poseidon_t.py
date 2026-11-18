"""Official Poseidon-T PT-FT wrapper with native rectangular-grid metadata."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    load_pretrained_with_report,
    load_torch_checkpoint_state,
)
from gphyt.model.official_import import (
    import_official_module,
    sha256_file,
    verify_source_file,
)
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/camlab-ethz/poseidon"
COMMIT = "b8fa28f59bd7f7673323f28d11a12c6f3a215c61"
SOURCE_RELATIVE_PATH = "scOT/model.py"
SOURCE_SHA256 = "10a3dcd29e26f65e953ee7d69af022347d4ce25565f9959b82f24b3c92c3a2a1"
CHECKPOINT_REPOSITORY = "camlab-ethz/Poseidon-T"
CHECKPOINT_SHA256 = "a363f7317fbc3a900a318fc63cc53197705d95fce0e0ce28dd3c8844a89112e2"
LICENSE = "CC-BY-NC-4.0"


def _rectangular_unmerge(self, input_feature, output_dimensions, time):
    output_height, output_width = output_dimensions
    batch, sequence_length, hidden_size = input_feature.shape
    input_height = output_height // 2
    input_width = output_width // 2
    if sequence_length != input_height * input_width:
        raise AssertionError(
            f"Poseidon decoder grid mismatch: N={sequence_length}, "
            f"expected {input_height}*{input_width}"
        )
    input_feature = self.upsample(input_feature)
    input_feature = input_feature.reshape(
        batch, input_height, input_width, 2, 2, hidden_size // 2
    )
    input_feature = input_feature.permute(0, 1, 3, 2, 4, 5)
    input_feature = input_feature.reshape(
        batch, 2 * input_height, 2 * input_width, hidden_size // 2
    )
    input_feature = self.maybe_crop(input_feature, output_height, output_width)
    input_feature = input_feature.reshape(batch, -1, hidden_size // 2)
    input_feature = self.norm(input_feature, time)
    return self.mixup(input_feature)


def _rectangular_convnext(self, x, time):
    batch, sequence_length, hidden_size = x.shape
    height, width = self._flood_hw
    if sequence_length != height * width:
        raise AssertionError(
            f"Poseidon residual grid mismatch: N={sequence_length}, H/W={self._flood_hw}"
        )
    residual = x
    x = x.reshape(batch, height, width, hidden_size).permute(0, 3, 1, 2)
    x = self.dwconv(x).permute(0, 2, 3, 1)
    x = self.norm(x, time)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)
    if self.weight is not None:
        x = self.weight * x
    x = x.reshape(batch, sequence_length, hidden_size)
    return residual + self.drop_path(x)


class PoseidonTPTFT(nn.Module):
    experiment_name = "Poseidon-T-PT-FT"
    source_repository = REPOSITORY
    source_commit = COMMIT
    checkpoint_repository = CHECKPOINT_REPOSITORY
    checkpoint_sha256 = CHECKPOINT_SHA256
    license = LICENSE
    time_condition = 1.0
    spatial_multiple = 32  # patch 4 and three factor-2 merges

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
        module = import_official_module(
            "scOT.model",
            root=official_root,
            import_subdirs=("",),
            install_hint="install transformers==4.35.2 as used by the official repository",
        )
        padded_height = int(runtime_height) + (-int(runtime_height)) % self.spatial_multiple
        padded_width = int(runtime_width) + (-int(runtime_width)) % self.spatial_multiple
        self.runtime_padded_shape = (padded_height, padded_width)
        self.official_config = {
            "image_size": self.runtime_padded_shape,
            "patch_size": 4,
            "num_channels": 3,
            "num_out_channels": 1,
            "embed_dim": 48,
            "depths": [4, 4, 4, 4],
            "num_heads": [3, 6, 12, 24],
            "window_size": 16,
            "mlp_ratio": 4.0,
            "skip_connections": [2, 2, 2, 0],
            "residual_model": "convnext",
            "use_conditioning": True,
            "learn_residual": False,
            "drop_path_rate": 0.0,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        }
        config = module.ScOTConfig(**self.official_config)
        self.backbone = module.ScOT(config)
        self._install_rectangular_metadata(module)

        digest = sha256_file(checkpoint_path)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Poseidon-T checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}"
            )
        state, loaded_digest = load_torch_checkpoint_state(checkpoint_path)
        if digest != loaded_digest:
            raise AssertionError("Poseidon checkpoint changed while loading")
        self.loading_report: PretrainedLoadingReport = load_pretrained_with_report(
            self.backbone,
            state,
            checkpoint_sha256=digest,
            report_path=report_path,
            converters={},
            newly_initialized_prefixes=(
                "embeddings.patch_embeddings.projection",
                "patch_recovery",
            ),
            backbone_excluded_prefixes=(
                "embeddings.patch_embeddings.projection",
                "patch_recovery",
            ),
            minimum_backbone_ratio=0.90,
        )
        assert_full_finetuning(self)

    def _install_rectangular_metadata(self, module) -> None:
        patch_height = self.runtime_padded_shape[0] // 4
        patch_width = self.runtime_padded_shape[1] // 4
        for layer in self.backbone.decoder.layers:
            if layer.upsample is not None:
                layer.upsample.forward = MethodType(_rectangular_unmerge, layer.upsample)
        for stage, blocks in enumerate(self.backbone.residual_blocks):
            stage_hw = (patch_height // (2**stage), patch_width // (2**stage))
            for block in blocks:
                if isinstance(block, module.ConvNeXtBlock):
                    block._flood_hw = stage_hw
                    block.forward = MethodType(_rectangular_convnext, block)

    def parameter_groups(self, backbone_lr: float = 5e-5, adapter_lr: float = 5e-4):
        adapter_prefixes = (
            "backbone.embeddings.patch_embeddings",
            "backbone.patch_recovery",
        )
        adapter, backbone = [], []
        for name, parameter in self.named_parameters():
            (adapter if name.startswith(adapter_prefixes) else backbone).append(parameter)
        return [
            {"params": backbone, "lr": float(backbone_lr), "name": "backbone"},
            {"params": adapter, "lr": float(adapter_lr), "name": "new_io_adapters"},
        ]

    def _forward_rectangular(self, pixel_values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        head_mask = backbone.get_head_mask(
            None, backbone.num_layers_encoder + backbone.num_layers_decoder
        )
        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: backbone.num_layers_encoder]
            head_mask_decoder = head_mask[backbone.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [backbone.num_layers_encoder, backbone.num_layers_decoder]
            )
        embedding_output, input_dimensions = backbone.embeddings(
            pixel_values, bool_masked_pos=None, time=time
        )
        encoder_outputs = backbone.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask_encoder,
            output_attentions=False,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )
        skip_states = list(encoder_outputs.hidden_states[1:])
        for index, blocks in enumerate(backbone.residual_blocks):
            for block in blocks:
                skip_states[index] = (
                    block(skip_states[index])
                    if isinstance(block, nn.Identity)
                    else block(skip_states[index], time)
                )
        deepest_dimensions = (
            input_dimensions[0] // 8,
            input_dimensions[1] // 8,
        )
        decoder_output = backbone.decoder(
            skip_states[-1],
            deepest_dimensions,
            time=time,
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return backbone.patch_recovery(decoder_output[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, self.spatial_multiple)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(
                f"Poseidon was built for {self.runtime_padded_shape}, got "
                f"{tuple(padded.shape[-2:])}; build per native grid, no resize is allowed"
            )
        time = torch.full(
            (batch,), self.time_condition, device=x.device, dtype=x.dtype
        )
        prediction = self._forward_rectangular(padded, time)
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
