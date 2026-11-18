"""Official EarthFormer CuboidTransformerModel FloodCastBench wrapper.

Backbone source: amazon-science/earth-forecasting-transformer at commit
7732b03bdb366110563516c3502315deab4c2026.  The wrapper verifies the exact
official source file before importing it and never spatially resizes inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gphyt.model.official_import import (
    import_official_module,
    verify_source_file,
)
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/amazon-science/earth-forecasting-transformer"
COMMIT = "7732b03bdb366110563516c3502315deab4c2026"
SOURCE_RELATIVE_PATH = "src/earthformer/cuboid_transformer/cuboid_transformer.py"
SOURCE_SHA256 = "672206cf0d5daa3847a401846da6aadf88309c290684c8a24ec63d3deb7c67b7"

# The official SEVIR-v1 graph computes one final encoder global-vector update
# after producing its last local output.  Decoder global self/cross attention
# is disabled, so that final global vector has no consumer.  Keeping these
# parameters trainable makes DDP fail because they can never receive a gradient.
_TERMINAL_UNUSED_GLOBAL_UPDATE_PARAMETERS = (
    "encoder.blocks.1.0.attn_l.2.g2l_global_q_net.weight",
    "encoder.blocks.1.0.attn_l.2.g2l_k_net.weight",
    "encoder.blocks.1.0.attn_l.2.g2l_v_net.weight",
    "encoder.blocks.1.0.attn_l.2.g2g_global_qkv_net.weight",
    "encoder.blocks.1.0.attn_l.2.global_proj.weight",
    "encoder.blocks.1.0.attn_l.2.global_proj.bias",
)


class EarthFormerScratch(nn.Module):
    experiment_name = "EarthFormer-Scratch"
    source_repository = REPOSITORY
    source_commit = COMMIT
    # Stack-conv factor 3*2*2 followed by one hierarchy merge of factor 2.
    spatial_multiple = 24

    def __init__(
        self,
        runtime_height: int,
        runtime_width: int,
        *,
        official_root: str,
    ) -> None:
        super().__init__()
        if not official_root:
            raise ValueError("EarthFormer-Scratch requires --official-root for commit verification")
        verify_source_file(
            official_root,
            SOURCE_RELATIVE_PATH,
            SOURCE_SHA256,
            repository=REPOSITORY,
            commit=COMMIT,
        )
        module = import_official_module(
            "earthformer.cuboid_transformer.cuboid_transformer",
            root=official_root,
            import_subdirs=("src",),
            install_hint=f"checkout {REPOSITORY} at {COMMIT}",
        )
        model_class = module.CuboidTransformerModel
        padded_height = int(runtime_height) + (-int(runtime_height)) % self.spatial_multiple
        padded_width = int(runtime_width) + (-int(runtime_width)) % self.spatial_multiple
        self.runtime_padded_shape = (padded_height, padded_width)

        # Field names and values are taken directly from
        # earthformer_sevir_v1.yaml.  cross_self_pattern is axial in the YAML;
        # cross_1x1 is the separate cross_pattern field.
        self.official_config = {
            "input_shape": (1, padded_height, padded_width, 3),
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
            "checkpoint_level": 0,
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
        self.backbone = model_class(**self.official_config)
        self._freeze_terminal_unused_global_update()

    def _freeze_terminal_unused_global_update(self) -> None:
        """Freeze parameters belonging only to the terminal dead branch."""
        if self.official_config["use_dec_self_global"]:
            raise AssertionError(
                "terminal global update assumes decoder globals are disabled"
            )
        if self.official_config["use_dec_cross_global"]:
            raise AssertionError(
                "terminal global update assumes decoder globals are disabled"
            )

        parameters = dict(self.backbone.named_parameters())
        expected = set(_TERMINAL_UNUSED_GLOBAL_UPDATE_PARAMETERS)
        missing = sorted(expected.difference(parameters))
        if missing:
            raise RuntimeError(
                "verified official EarthFormer no longer has the expected terminal "
                f"global-update parameters: {missing}"
            )
        for name in _TERMINAL_UNUSED_GLOBAL_UPDATE_PARAMETERS:
            parameters[name].requires_grad_(False)

        self.structurally_inactive_parameter_names = tuple(
            f"backbone.{name}"
            for name in _TERMINAL_UNUSED_GLOBAL_UPDATE_PARAMETERS
        )
        self.structurally_inactive_parameter_count = sum(
            parameters[name].numel()
            for name in _TERMINAL_UNUSED_GLOBAL_UPDATE_PARAMETERS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, self.spatial_multiple)
        if tuple(padded.shape[-2:]) != self.runtime_padded_shape:
            raise ValueError(
                "official CuboidTransformerModel binds its encoder/decoder shape at "
                f"construction: configured {self.runtime_padded_shape}, received "
                f"{tuple(padded.shape[-2:])}. Build one experiment per native grid; "
                "the wrapper will not resize or over-pad to a different fixed grid."
            )
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        official_output = self.backbone(official_input)
        if official_output.ndim != 5 or official_output.shape[1] != 1:
            raise AssertionError(
                f"EarthFormer returned unexpected shape {tuple(official_output.shape)}"
            )
        prediction = official_output[:, 0].permute(0, 3, 1, 2).contiguous()
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
