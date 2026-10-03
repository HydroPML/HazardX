from __future__ import annotations

import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer


_DOFA_ROOT = Path("/root/workcsc/DOFA")
if _DOFA_ROOT.exists() and str(_DOFA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOFA_ROOT))

from wave_dynamic_layer import Dynamic_MLP_OFA  # type: ignore  # noqa: E402


class DOFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=768,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=14, embed_dim=embed_dim
        )
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches
        model_args = dict(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            init_values=1e-5,
            num_classes=0,
        )
        self.model = VisionTransformer(**model_args)
        del self.model.patch_embed
        self.waves = None
        self.norm = norm_layer(embed_dim)

    def forward_features(self, x, wave_list):
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        patch_tokens = x.shape[1]
        hw = int(round(math.sqrt(patch_tokens)))
        if hw * hw != patch_tokens:
            raise RuntimeError(f"Expected a square patch grid, got {patch_tokens} tokens.")
        hw_shape = (hw, hw)

        pos_embed = self.model.pos_embed
        if pos_embed is None:
            raise RuntimeError("DOFA ViT backbone is missing positional embeddings.")

        num_prefix_tokens = 1 if self.model.cls_token is not None else 0
        if pos_embed.shape[1] != patch_tokens + num_prefix_tokens:
            old_tokens = pos_embed.shape[1] - num_prefix_tokens
            old_hw = int(round(math.sqrt(old_tokens)))
            pos_embed = resample_abs_pos_embed(
                pos_embed,
                new_size=hw_shape,
                old_size=(old_hw, old_hw),
                num_prefix_tokens=num_prefix_tokens,
            )

        if self.model.cls_token is not None:
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + pos_embed.to(device=x.device, dtype=x.dtype)
        x = self.model.pos_drop(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        out_features = []

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_indices:
                out_features.append(x[:, 1:])

        x = self.model.norm(x)
        return out_features

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        return x


def vit_base_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[4, 6, 10, 11],
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[5, 11, 17, 23],
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
