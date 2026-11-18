"""Task adapters for WildfireSpreadTS scratch baselines.

Flood architecture widths/depths are retained.  Only the input/output shapes
and temporal extent differ from the FloodCastBench wrappers.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from floodcast_four_baselines import DDPCompatibleSpectralConv2d
from gphyt.model.convlstm_openstl import ConvLSTMCell
from gphyt.model.official_import import import_official_module
from models.FNO import FNO2d
from models.Unet import UNet2d


def validate_wildfire_input(x: torch.Tensor) -> None:
    if x.ndim != 5 or tuple(x.shape[1:3]) != (5, 40):
        raise AssertionError(f"expected [B,5,40,H,W], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("wildfire input has NaN/Inf")


def validate_logits(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    expected = (x.shape[0], 1, x.shape[-2], x.shape[-1])
    if tuple(y.shape) != expected or not torch.isfinite(y).all():
        raise AssertionError(f"invalid wildfire logits {tuple(y.shape)}, expected {expected}")
    return y


class WildfireUNet2D(nn.Module):
    architecture_source = "Flood08/models/Unet.py:UNet2d"
    architecture = {"init_features": 32, "levels": 4, "grid_type": "cartesian"}

    def __init__(self) -> None:
        super().__init__()
        self.backbone = UNet2d(200, 1, 32, "cartesian")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_wildfire_input(x)
        b, _, _, h, w = x.shape
        fields = x.reshape(b, 200, h, w)
        pad_h, pad_w = (-h) % 16, (-w) % 16
        fields = F.pad(fields, (0, pad_w, 0, pad_h))
        out = self.backbone(fields.permute(0, 2, 3, 1).unsqueeze(3))
        return validate_logits(out[:, :h, :w, 0, 0].unsqueeze(1), x)


class WildfireFNO2D(nn.Module):
    architecture_source = "Flood08/models/FNO.py:FNO2d"
    architecture = {"modes": (12, 12), "width": 20, "blocks": 4, "grid_type": "cartesian"}

    def __init__(self) -> None:
        super().__init__()
        self.backbone = FNO2d(200, 12, 12, 20, 1, "cartesian")
        for name in ("conv0", "conv1", "conv2", "conv3"):
            setattr(self.backbone, name, DDPCompatibleSpectralConv2d(getattr(self.backbone, name)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_wildfire_input(x)
        b, _, _, h, w = x.shape
        fields = x.reshape(b, 200, h, w).permute(0, 2, 3, 1).unsqueeze(3)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            out = self.backbone(fields.float())
        return validate_logits(out[..., 0, 0].unsqueeze(1), x)


class WildfireConvLSTM(nn.Module):
    architecture_source = "Flood08/gphyt/model/convlstm_openstl.py"
    architecture = {"hidden_channels": (128, 128, 128, 128), "kernel_size": 5, "patch_size": 2}

    def __init__(self) -> None:
        super().__init__()
        hidden = self.architecture["hidden_channels"]
        self.cells = nn.ModuleList([
            ConvLSTMCell(40 * 4 if i == 0 else hidden[i - 1], c, 5)
            for i, c in enumerate(hidden)
        ])
        self.recovery = nn.Conv2d(hidden[-1], 4, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_wildfire_input(x)
        b, t, c, h, w = x.shape
        if h % 2 or w % 2:
            raise AssertionError("128x128 protocol must be divisible by ConvLSTM patch size")
        seq = F.pixel_unshuffle(x.reshape(b * t, c, h, w), 2).reshape(b, t, c * 4, h // 2, w // 2)
        states = [(
            torch.zeros(b, 128, h // 2, w // 2, device=x.device, dtype=x.dtype),
            torch.zeros(b, 128, h // 2, w // 2, device=x.device, dtype=x.dtype),
        ) for _ in self.cells]
        for step in range(t):
            hidden = seq[:, step]
            new_states = []
            for cell, state in zip(self.cells, states):
                hidden, cell_state = cell(hidden, state)
                new_states.append((hidden, cell_state))
            states = new_states
        return validate_logits(self.pixel_shuffle(self.recovery(hidden)), x)


class WildfireEarthFormer(nn.Module):
    architecture_source = "official CuboidTransformerModel; Flood SEVIR-v1 configuration"
    spatial_multiple = 24

    def __init__(self, official_root: str) -> None:
        super().__init__()
        module = import_official_module(
            "earthformer.cuboid_transformer.cuboid_transformer", root=official_root,
            import_subdirs=("src",), install_hint="install official EarthFormer dependencies",
        )
        self.official_config = {
            "input_shape": (5, 144, 144, 40), "target_shape": (1, 144, 144, 1),
            "base_units": 128, "scale_alpha": 1.0, "enc_depth": [1, 1], "dec_depth": [1, 1],
            "enc_use_inter_ffn": True, "dec_use_inter_ffn": True,
            "dec_hierarchical_pos_embed": False, "downsample": 2, "downsample_type": "patch_merge",
            "upsample_type": "upsample", "num_global_vectors": 8,
            "use_dec_self_global": False, "dec_self_update_global": True,
            "use_dec_cross_global": False, "use_global_vector_ffn": False,
            "use_global_self_attn": True, "separate_global_qkv": True, "global_dim_ratio": 1,
            "enc_attn_patterns": ["axial", "axial"], "dec_self_attn_patterns": ["axial", "axial"],
            "dec_cross_attn_patterns": ["cross_1x1", "cross_1x1"], "dec_cross_start": 0,
            "dec_cross_last_n_frames": None, "attn_drop": .1, "proj_drop": .1, "ffn_drop": .1,
            "num_heads": 4, "ffn_activation": "gelu", "gated_ffn": False,
            "norm_layer": "layer_norm", "padding_type": "zeros", "pos_embed_type": "t+h+w",
            "use_relative_pos": True, "self_attn_use_final_proj": True,
            "dec_use_first_self_attn": False, "z_init_method": "zeros", "checkpoint_level": 0,
            "initial_downsample_type": "stack_conv", "initial_downsample_activation": "leaky",
            "initial_downsample_stack_conv_num_layers": 3,
            "initial_downsample_stack_conv_dim_list": [16, 64, 128],
            "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
            "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],
            "attn_linear_init_mode": "0", "ffn_linear_init_mode": "0", "conv_init_mode": "0",
            "down_up_linear_init_mode": "0", "norm_init_mode": "0",
        }
        self.backbone = module.CuboidTransformerModel(**self.official_config)
        # Reuse the validated Flood wrapper's treatment of the official SEVIR
        # graph's terminal global update: these tensors have no downstream
        # consumer when decoder global attention is disabled.
        inactive = (
            "encoder.blocks.1.0.attn_l.2.g2l_global_q_net.weight",
            "encoder.blocks.1.0.attn_l.2.g2l_k_net.weight",
            "encoder.blocks.1.0.attn_l.2.g2l_v_net.weight",
            "encoder.blocks.1.0.attn_l.2.g2g_global_qkv_net.weight",
            "encoder.blocks.1.0.attn_l.2.global_proj.weight",
            "encoder.blocks.1.0.attn_l.2.global_proj.bias",
            # With wildfire T_out=1 this temporal attention group contains one
            # token, so its softmax is identically one and this bias cannot
            # affect the output.
            "decoder.self_blocks.0.0.attn_l.0.relative_position_bias_table",
        )
        parameters = dict(self.backbone.named_parameters())
        missing = sorted(set(inactive) - set(parameters))
        if missing:
            raise RuntimeError(f"EarthFormer official dead-branch layout changed: {missing}")
        for name in inactive:
            parameters[name].requires_grad_(False)
        self.structurally_inactive_parameter_names = tuple("backbone." + name for name in inactive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_wildfire_input(x)
        padded = F.pad(x, (0, 16, 0, 16))
        out = self.backbone(padded.permute(0, 1, 3, 4, 2).contiguous())
        return validate_logits(out[:, 0, :128, :128].permute(0, 3, 1, 2).contiguous(), x)


class WildfireUTAE(nn.Module):
    architecture_source = "WildfireSpreadTS vendored official UTAE-PAPS implementation"

    def __init__(self, official_root: str) -> None:
        super().__init__()
        root = Path(official_root).resolve()
        # Load only the official UTAE package. Importing WSTS ``src.models`` also
        # imports its Lightning/W&B training shell, which is unrelated to UTAE.
        import types
        package_name = "_wsts_official_utae"
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "src/models/utae_paps_models")]
        sys.modules[package_name] = package
        cls = importlib.import_module(package_name + ".utae").UTAE
        self.backbone = cls(
            input_dim=40, encoder_widths=[64, 64, 64, 128], decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1], str_conv_k=4, str_conv_s=2, str_conv_p=1,
            agg_mode="att_group", encoder_norm="group", n_head=16, d_model=256, d_k=4,
            encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        validate_wildfire_input(x)
        if positions is None:
            raise ValueError("UTAE requires real day-of-year positions from the dataset")
        return validate_logits(self.backbone(x, batch_positions=positions.to(x.device), return_att=False), x)
