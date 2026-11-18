"""Native 12-channel/1024 wrappers for the seven audited flood baselines.

The original FloodCastBench wrappers deliberately retain their three-channel
API.  These new wrappers adapt only model I/O and checkpoint input/output
tensors for FloodSimBench context-6 forecasting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import gphyt.model.dpot_s as dpot_base
import gphyt.model.earthformer_official as earth_base
import gphyt.model.gphyt_s as gphyt_base
import gphyt.model.pde_transformer_sc_s as pde_base
import gphyt.model.poseidon_t as poseidon_base
import gphyt.model.prithvi_eo2_300m as prithvi_base
from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    load_pretrained_with_report,
    load_torch_checkpoint_state,
)
from gphyt.model.convlstm_openstl import ConvLSTMCell
from gphyt.model.dynamic_position_adapter import (
    install_gphyt_dynamic_absolute_position,
)
from gphyt.model.floodsimbench_context12_common import (
    crop_to_original,
    minimal_pad_bottom_right_context12,
    validate_context12_input,
    validate_context12_output,
)
from gphyt.model.official_import import sha256_file, verify_source_file


CONTEXT_CHANNELS = 12
LATEST_DEPTH_CHANNEL = 11


def _append_report_conversion(
    report: PretrainedLoadingReport, marker: str, report_path: str
) -> None:
    if marker not in report.converted:
        report.converted.append(marker)
        report.converted.sort()
    report.write(report_path)


def _enable_official_checkpointing(module: nn.Module) -> bool:
    """Enable an official checkpoint API when one is exposed."""

    for name in (
        "gradient_checkpointing_enable",
        "enable_gradient_checkpointing",
        "set_grad_checkpointing",
    ):
        method = getattr(module, name, None)
        if callable(method):
            try:
                method(True) if name == "set_grad_checkpointing" else method()
            except TypeError:
                method()
            return True
    return False


class ConvLSTMContext12Scratch(nn.Module):
    experiment_name = "ConvLSTM-Scratch"
    architecture_variant = "ConvLSTM-OpenSTL-Context12"
    patch_size = 2

    def __init__(
        self,
        hidden_channels: Tuple[int, ...] = (128, 128, 128, 128),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if tuple(hidden_channels) != (128, 128, 128, 128):
            raise ValueError("ConvLSTM benchmark fixes four 128-channel layers")
        self.hidden_channels = tuple(hidden_channels)
        self.activation_checkpointing = False
        cells: List[nn.Module] = []
        for layer, channels in enumerate(self.hidden_channels):
            in_channels = (
                CONTEXT_CHANNELS * self.patch_size**2
                if layer == 0
                else self.hidden_channels[layer - 1]
            )
            cells.append(ConvLSTMCell(in_channels, channels, kernel_size))
        self.cells = nn.ModuleList(cells)
        self.recovery = nn.Conv2d(self.hidden_channels[-1], self.patch_size**2, 1)
        self.pixel_shuffle = nn.PixelShuffle(self.patch_size)

    def enable_activation_checkpointing(self) -> None:
        self.activation_checkpointing = True

    def _zero_state(
        self, batch: int, height: int, width: int, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(batch, channels, height, width, device=x.device, dtype=x.dtype),
                torch.zeros(batch, channels, height, width, device=x.device, dtype=x.dtype),
            )
            for channels in self.hidden_channels
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, self.patch_size)
        hidden = F.pixel_unshuffle(padded, self.patch_size)
        expected_channels = CONTEXT_CHANNELS * self.patch_size**2
        if hidden.shape[1] != expected_channels:
            raise AssertionError(f"ConvLSTM patch channels {hidden.shape[1]} != {expected_channels}")
        state = self._zero_state(batch, hidden.shape[-2], hidden.shape[-1], hidden)
        for cell, (h, c) in zip(self.cells, state):
            if self.activation_checkpointing and self.training:
                hidden, _ = checkpoint(
                    lambda value, h0, c0, layer=cell: layer(value, (h0, c0)),
                    hidden,
                    h,
                    c,
                    use_reentrant=False,
                )
            else:
                hidden, _ = cell(hidden, (h, c))
        prediction = self.pixel_shuffle(self.recovery(hidden))
        prediction = crop_to_original(prediction, pad)
        return validate_context12_output(prediction, x)


class EarthFormerContext12Scratch(earth_base.EarthFormerScratch):
    experiment_name = "EarthFormer-Scratch"

    def __init__(self, runtime_height: int, runtime_width: int, *, official_root: str) -> None:
        nn.Module.__init__(self)
        if not official_root:
            raise ValueError("EarthFormer-Scratch requires --official-root")
        earth_base.verify_source_file(
            official_root,
            earth_base.SOURCE_RELATIVE_PATH,
            earth_base.SOURCE_SHA256,
            repository=earth_base.REPOSITORY,
            commit=earth_base.COMMIT,
        )
        module = earth_base.import_official_module(
            "earthformer.cuboid_transformer.cuboid_transformer",
            root=official_root,
            import_subdirs=("src",),
            install_hint=f"checkout {earth_base.REPOSITORY} at {earth_base.COMMIT}",
        )
        padded_height = int(runtime_height) + (-int(runtime_height)) % self.spatial_multiple
        padded_width = int(runtime_width) + (-int(runtime_width)) % self.spatial_multiple
        self.runtime_padded_shape = (padded_height, padded_width)
        self.official_config = {
            "input_shape": (1, padded_height, padded_width, CONTEXT_CHANNELS),
            "target_shape": (1, padded_height, padded_width, 1),
            "base_units": 128,
            "scale_alpha": 1.0,
            "enc_depth": [1, 1],
            "dec_depth": [1, 1],
            "enc_use_inter_ffn": True,
            "dec_use_inter_ffn": True,
            "dec_hierarchical_pos_embed": False,
            "downsample": 2,
            "downsample_type": "patch_merge",
            "upsample_type": "upsample",
            "num_global_vectors": 8,
            "use_dec_self_global": False,
            "dec_self_update_global": True,
            "use_dec_cross_global": False,
            "use_global_vector_ffn": False,
            "use_global_self_attn": True,
            "separate_global_qkv": True,
            "global_dim_ratio": 1,
            "enc_attn_patterns": ["axial", "axial"],
            "dec_self_attn_patterns": ["axial", "axial"],
            "dec_cross_attn_patterns": ["cross_1x1", "cross_1x1"],
            "dec_cross_start": 0,
            "dec_cross_last_n_frames": None,
            "attn_drop": 0.1,
            "proj_drop": 0.1,
            "ffn_drop": 0.1,
            "num_heads": 4,
            "ffn_activation": "gelu",
            "gated_ffn": False,
            "norm_layer": "layer_norm",
            "padding_type": "zeros",
            "pos_embed_type": "t+h+w",
            "use_relative_pos": True,
            "self_attn_use_final_proj": True,
            "dec_use_first_self_attn": False,
            "z_init_method": "zeros",
            # Official EarthFormer activation checkpointing; no architecture change.
            "checkpoint_level": 2,
            "initial_downsample_type": "stack_conv",
            "initial_downsample_activation": "leaky",
            "initial_downsample_stack_conv_num_layers": 3,
            "initial_downsample_stack_conv_dim_list": [16, 64, 128],
            "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
            "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],
            "attn_linear_init_mode": "0",
            "ffn_linear_init_mode": "0",
            "conv_init_mode": "0",
            "down_up_linear_init_mode": "0",
            "norm_init_mode": "0",
        }
        self.backbone = module.CuboidTransformerModel(**self.official_config)
        self._freeze_terminal_unused_global_update()

    def enable_activation_checkpointing(self) -> None:
        # checkpoint_level is fixed before official model construction.
        if self.official_config.get("checkpoint_level", 0) <= 0:
            raise RuntimeError("EarthFormer checkpoint_level is not enabled")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, self.spatial_multiple)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(
                f"EarthFormer configured for {self.runtime_padded_shape}, got {tuple(padded.shape[-2:])}"
            )
        output = self.backbone(padded.permute(0, 2, 3, 1).unsqueeze(1).contiguous())
        prediction = output[:, 0].permute(0, 3, 1, 2).contiguous()
        prediction = crop_to_original(prediction, pad)
        return validate_context12_output(prediction, x)


class PDETransformerContext12PTFT(pde_base.PDETransformerSCSPTFT):
    experiment_name = "PDE-T-SC-S-PT-FT"
    channel_type_ids_context12 = (1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.official_config["physical_fields"] = CONTEXT_CHANNELS
        self.activation_checkpointing_enabled = False
        self.activation_checkpointed_blocks = 0
        self.enable_activation_checkpointing()

    def enable_activation_checkpointing(self) -> None:
        if self.activation_checkpointing_enabled:
            return
        implementation = self.backbone.model
        stages = [
            getattr(implementation, f"encoder_level_{index}")
            for index in range(implementation.num_encoder_layers)
        ]
        stages.append(implementation.latent)
        stages.extend(
            getattr(implementation, f"decoder_level_{index}")
            for index in range(implementation.num_encoder_layers)
        )
        count = 0
        for stage in stages:
            for block in stage.blocks:
                original_forward = block.forward
                owner = block

                def checkpointed_forward(*args, _forward=original_forward, _owner=owner, **kwargs):
                    if _owner.training and torch.is_grad_enabled():
                        return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                    return _forward(*args, **kwargs)

                block.forward = checkpointed_forward
                count += 1
        if count != 22:
            raise RuntimeError(f"PDE-T expected 22 official HAT blocks, got {count}")
        self.activation_checkpointed_blocks = count
        self.activation_checkpointing_enabled = True

    @staticmethod
    def _metadata(batch: int, device: torch.device, dtype: torch.dtype):
        count = CONTEXT_CHANNELS
        zeros = [torch.zeros(batch, device=device, dtype=dtype) for _ in range(count)]
        channel_type = [
            torch.full((batch,), value, device=device, dtype=torch.long)
            for value in PDETransformerContext12PTFT.channel_type_ids_context12
        ]
        labels = [torch.zeros(batch, device=device, dtype=torch.long) for _ in range(count)]
        parameters = [torch.zeros(batch, 1, device=device, dtype=dtype) for _ in range(count)]
        parameter_classes = [
            torch.zeros(batch, 1, device=device, dtype=torch.long) for _ in range(count)
        ]
        tasks = [torch.zeros(batch, device=device, dtype=torch.long) for _ in range(count)]
        return zeros, channel_type, labels, parameters, parameter_classes, tasks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, self.spatial_multiple)
        fields = [
            torch.cat((padded[:, index : index + 1], torch.zeros_like(padded[:, index : index + 1])), dim=1)
            for index in range(CONTEXT_CHANNELS)
        ]
        zeros, channel_type, labels, parameters, parameter_classes, tasks = self._metadata(
            batch, padded.device, padded.dtype
        )
        result = self.backbone(
            x=fields,
            simulation_time=zeros,
            channel_type=channel_type,
            pde_type=labels,
            pde_parameters=parameters,
            pde_parameters_class=parameter_classes,
            simulation_dt=[torch.ones_like(value) for value in zeros],
            task=tasks,
            t=zeros,
            return_dict=True,
        )
        outputs = result.sample
        if not isinstance(outputs, list) or len(outputs) != CONTEXT_CHANNELS:
            raise AssertionError(f"PDE-T returned {type(outputs)} with {len(outputs) if isinstance(outputs, list) else 'n/a'} fields")
        latest_depth = outputs[LATEST_DEPTH_CHANNEL]
        if latest_depth.ndim != 4 or latest_depth.shape[1] != 2:
            raise AssertionError(f"unexpected PDE-T depth field {tuple(latest_depth.shape)}")
        prediction = crop_to_original(latest_depth[:, 1:2], pad)
        return validate_context12_output(prediction, x)


class DPOTContext12PTFT(dpot_base.DPOTSPTFT):
    experiment_name = "DPOT-S-PT-FT"

    def __init__(self, *args, **kwargs) -> None:
        report_path = str(kwargs["report_path"])
        super().__init__(*args, **kwargs)
        projection = self.backbone.patch_embed.proj[0]
        if not isinstance(projection, nn.Conv2d) or projection.in_channels != 6:
            raise RuntimeError("unexpected loaded DPOT input projection")
        expanded = nn.Conv2d(
            15,
            projection.out_channels,
            projection.kernel_size,
            projection.stride,
            projection.padding,
            projection.dilation,
            projection.groups,
            projection.bias is not None,
            projection.padding_mode,
        )
        with torch.no_grad():
            physical = projection.weight[:, :3].mean(dim=1, keepdim=True)
            # The official DPOT-S checkpoint has four physical input channels.
            # Preserve its equal-channel convolution response when expanding
            # those physical kernels to the 12 FloodSimBench context channels.
            expanded.weight[:, :12].copy_(physical.repeat(1, 12, 1, 1) * (4.0 / 12.0))
            expanded.weight[:, 12:15].copy_(projection.weight[:, 3:6])
            if projection.bias is not None:
                expanded.bias.copy_(projection.bias)
        self.backbone.patch_embed.proj[0] = expanded
        self.backbone.in_channels = CONTEXT_CHANNELS
        self.official_config["in_channels"] = CONTEXT_CHANNELS
        self.activation_checkpointing = True
        _append_report_conversion(
            self.loading_report,
            "context12::patch_embed.proj.0.weight[4+3->12+3]",
            report_path,
        )

    def enable_activation_checkpointing(self) -> None:
        self.activation_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, self.patch_size)
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        backbone = self.backbone
        batch, height, width, timesteps, _ = official_input.shape
        grid = backbone.get_grid_3d(official_input)
        hidden = torch.cat((official_input, grid), dim=-1).contiguous()
        hidden = dpot_base.rearrange(hidden, "b x y t c -> (b t) c x y")
        hidden = backbone.patch_embed.proj(hidden)
        position = F.interpolate(
            backbone.pos_embed, size=hidden.shape[-2:], mode="bicubic", align_corners=False
        )
        hidden = hidden + position
        hidden = dpot_base.rearrange(hidden, "(b t) c x y -> b x y t c", b=batch, t=timesteps)
        hidden = backbone.time_agg_layer(hidden)
        hidden = dpot_base.rearrange(hidden, "b x y c -> b c x y")
        for block in backbone.blocks:
            hidden = (
                checkpoint(block, hidden, use_reentrant=False)
                if self.activation_checkpointing and self.training
                else block(hidden)
            )
        prediction = backbone.out_layer(hidden).permute(0, 2, 3, 1)
        prediction = prediction.reshape(batch, height, width, 1, 1)[..., 0, 0].unsqueeze(1)
        prediction = crop_to_original(prediction, pad)
        return validate_context12_output(prediction, x)


class PoseidonContext12PTFT(poseidon_base.PoseidonTPTFT):
    experiment_name = "Poseidon-T-PT-FT"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        projection = self.backbone.embeddings.patch_embeddings.projection
        if not isinstance(projection, nn.Conv2d) or projection.in_channels != 3:
            raise RuntimeError("unexpected Poseidon patch projection")
        self.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(
            CONTEXT_CHANNELS,
            projection.out_channels,
            projection.kernel_size,
            projection.stride,
            projection.padding,
            projection.dilation,
            projection.groups,
            projection.bias is not None,
            projection.padding_mode,
        )
        self.official_config["num_channels"] = CONTEXT_CHANNELS
        self.backbone.embeddings.patch_embeddings.num_channels = CONTEXT_CHANNELS
        self.activation_checkpointing_enabled = _enable_official_checkpointing(self.backbone)

    def enable_activation_checkpointing(self) -> None:
        if not self.activation_checkpointing_enabled:
            self.activation_checkpointing_enabled = _enable_official_checkpointing(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, self.spatial_multiple)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(f"Poseidon configured for {self.runtime_padded_shape}, got {tuple(padded.shape[-2:])}")
        time = torch.full((batch,), self.time_condition, device=x.device, dtype=x.dtype)
        prediction = self._forward_rectangular(padded, time)
        prediction = crop_to_original(prediction, pad)
        return validate_context12_output(prediction, x)


def _convert_gphyt_derivative_context12(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.shape[0] != 5 or target.shape[0] != CONTEXT_CHANNELS or tuple(source.shape[1:]) != tuple(target.shape[1:]):
        raise RuntimeError(f"unconfirmed GPhyT derivative layout {key}: {tuple(source.shape)} -> {tuple(target.shape)}")
    if not torch.equal(source, source[:1].expand_as(source)):
        raise RuntimeError(f"GPhyT derivative filters differ by source variable: {key}")
    return source[:1].repeat(CONTEXT_CHANNELS, *([1] * (source.ndim - 1)))


def _convert_gphyt_tokenizer_context12(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.ndim != 5 or source.shape[1] != 20 or target.shape[1] != 48:
        raise RuntimeError(f"unconfirmed GPhyT tokenizer {key}: {tuple(source.shape)} -> {tuple(target.shape)}")
    grouped = source.reshape(source.shape[0], 4, 5, *source.shape[2:]).mean(dim=2, keepdim=True)
    grouped = grouped.repeat(1, 1, CONTEXT_CHANNELS, 1, 1, 1) * (5.0 / CONTEXT_CHANNELS)
    return grouped.reshape_as(target)


def _convert_gphyt_detokenizer_weight_context12(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.ndim != 5 or source.shape[1] != 5 or target.shape[1] != CONTEXT_CHANNELS:
        raise RuntimeError(f"unconfirmed GPhyT detokenizer {key}: {tuple(source.shape)} -> {tuple(target.shape)}")
    return source.mean(dim=1, keepdim=True).repeat(1, CONTEXT_CHANNELS, 1, 1, 1)


def _convert_gphyt_detokenizer_bias_context12(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if tuple(source.shape) != (5,) or tuple(target.shape) != (CONTEXT_CHANNELS,):
        raise RuntimeError(f"unconfirmed GPhyT detokenizer bias {key}")
    return source.mean().expand_as(target).clone()


class GPhyTContext12PTFT(nn.Module):
    experiment_name = "GPhyT-S-PT-FT"
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
            gphyt_base.SOURCE_RELATIVE_PATH,
            gphyt_base.SOURCE_SHA256,
            repository=gphyt_base.REPOSITORY,
            commit=gphyt_base.COMMIT,
        )
        module = gphyt_base._import_exact_gphyt(official_root)
        padded_height = int(runtime_height) + (-int(runtime_height)) % 16
        padded_width = int(runtime_width) + (-int(runtime_width)) % 16
        self.runtime_padded_shape = (padded_height, padded_width)
        self.official_config = {
            "architecture": "gphyt",
            "img_size": (1, padded_height, padded_width),
            "transformer": {
                "input_channels": CONTEXT_CHANNELS,
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
        if digest != gphyt_base.CHECKPOINT_SHA256:
            raise RuntimeError(f"GPhyT checkpoint SHA256 mismatch: {digest}")
        state, loaded_digest = load_torch_checkpoint_state(checkpoint_path)
        if loaded_digest != digest:
            raise AssertionError("GPhyT checkpoint changed while loading")
        converters = {
            "derivatives.dt_filter": _convert_gphyt_derivative_context12,
            "derivatives.dh_filter": _convert_gphyt_derivative_context12,
            "derivatives.dw_filter": _convert_gphyt_derivative_context12,
            "tokenizer.tokenizer.to_patch_embedding.1.weight": _convert_gphyt_tokenizer_context12,
            "detokenizer.detokenizer.from_patch_embedding.1.weight": _convert_gphyt_detokenizer_weight_context12,
            "detokenizer.detokenizer.from_patch_embedding.1.bias": _convert_gphyt_detokenizer_bias_context12,
        }
        self.loading_report = load_pretrained_with_report(
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
        blocks = getattr(self.backbone, "attention_blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) == 0:
            raise RuntimeError("GPhyT activation checkpointing requires official attention_blocks")
        self.activation_checkpointing_enabled = True

    def enable_activation_checkpointing(self) -> None:
        blocks = getattr(self.backbone, "attention_blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) == 0:
            raise RuntimeError("GPhyT activation checkpointing requires official attention_blocks")
        self.activation_checkpointing_enabled = True

    def _differentiate(self, value: torch.Tensor) -> torch.Tensor:
        if self.backbone.use_derivatives:
            dt, dh, dw = self.backbone.derivatives(value)
            value = torch.cat([value, dt, dh, dw], dim=-1)
        value = self.backbone.tokenizer(value)
        if self.backbone.init_pos_encodings is not None:
            value = self.backbone.init_pos_encodings(value)
        for block in self.backbone.attention_blocks:
            if self.activation_checkpointing_enabled and self.training:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return self.backbone.detokenizer(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, 16)
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        predicted_fields = (self._differentiate(official_input) + official_input)[:, -1, ...].unsqueeze(1)
        if predicted_fields.ndim != 5 or predicted_fields.shape[-1] != CONTEXT_CHANNELS:
            raise AssertionError(f"GPhyT returned {tuple(predicted_fields.shape)}")
        prediction = predicted_fields[:, 0, :, :, LATEST_DEPTH_CHANNEL].unsqueeze(1)
        prediction = crop_to_original(prediction, pad)
        return validate_context12_output(prediction, x)


class PrithviContext12PTFT(prithvi_base.PrithviEO2300MPTFT):
    experiment_name = "Prithvi-EO2-300M-PT-FT"

    def __init__(self, *args, **kwargs) -> None:
        report_path = str(kwargs["report_path"])
        super().__init__(*args, **kwargs)
        projection = self.backbone.patch_embed.proj
        if not isinstance(projection, nn.Conv3d) or projection.in_channels != 3:
            raise RuntimeError("unexpected Prithvi patch projection")
        expanded = nn.Conv3d(
            CONTEXT_CHANNELS,
            projection.out_channels,
            projection.kernel_size,
            projection.stride,
            projection.padding,
            projection.dilation,
            projection.groups,
            projection.bias is not None,
            projection.padding_mode,
        )
        with torch.no_grad():
            # Existing audited adapter is official 6 -> 3 with factor 2.  A
            # further factor 3/12 gives the direct official 6/12 response scale.
            weight = projection.weight.mean(dim=1, keepdim=True)
            expanded.weight.copy_(weight.repeat(1, CONTEXT_CHANNELS, 1, 1, 1) * (3.0 / CONTEXT_CHANNELS))
            if projection.bias is not None:
                expanded.bias.copy_(projection.bias)
        self.backbone.patch_embed.proj = expanded
        _append_report_conversion(
            self.loading_report,
            "context12::patch_embed.proj.weight[6->12,response_scale=6/12]",
            report_path,
        )
        self.activation_checkpointing = False
        self.enable_activation_checkpointing()

    def enable_activation_checkpointing(self) -> None:
        """Checkpoint every official Prithvi transformer block."""
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) != 24:
            raise RuntimeError(
                "Prithvi activation checkpointing requires exactly 24 official "
                f"transformer blocks, got {None if blocks is None else len(blocks)}"
            )
        self.activation_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_context12_input(x)
        padded, pad = minimal_pad_bottom_right_context12(x, 16)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(f"Prithvi configured for {self.runtime_padded_shape}, got {tuple(padded.shape[-2:])}")
        official_input = padded.unsqueeze(2)
        sample_shape = tuple(int(v) for v in official_input.shape[-3:])
        tokens = self.backbone.patch_embed(official_input)
        position = self.backbone.interpolate_pos_encoding(sample_shape)
        tokens = tokens + position[:, 1:, :]
        cls_token = self.backbone.cls_token + position[:, :1, :]
        tokens = torch.cat(
            (cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1
        )
        for block in self.backbone.blocks:
            if self.activation_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.backbone.norm(tokens)[:, 1:, :]
        patch_height = padded.shape[-2] // 16
        patch_width = padded.shape[-1] // 16
        if tokens.shape[1] != patch_height * patch_width:
            raise AssertionError(f"Prithvi patch tokens {tokens.shape[1]} != {patch_height * patch_width}")
        feature_map = tokens.transpose(1, 2).reshape(
            x.shape[0], 1024, patch_height, patch_width
        )
        prediction = crop_to_original(self.recovery(feature_map), pad)
        return validate_context12_output(prediction, x)


MODEL_CLASSES = {
    "ConvLSTM-Scratch": ConvLSTMContext12Scratch,
    "EarthFormer-Scratch": EarthFormerContext12Scratch,
    "PDE-T-SC-S-PT-FT": PDETransformerContext12PTFT,
    "DPOT-S-PT-FT": DPOTContext12PTFT,
    "Poseidon-T-PT-FT": PoseidonContext12PTFT,
    "GPhyT-S-PT-FT": GPhyTContext12PTFT,
    "Prithvi-EO2-300M-PT-FT": PrithviContext12PTFT,
}
