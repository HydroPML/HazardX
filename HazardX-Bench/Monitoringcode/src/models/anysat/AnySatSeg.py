from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anysat_backbone import AnySatBackbone


class AnySatSeg(nn.Module):
    """AnySat segmentation wrapper for AnyDisasterMapping.

    This class takes AnySat patch-level features and decodes them with a light
    convolutional head to produce per-pixel logits.
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 2,
        model_size: str = "base",
        pretrained_backbone: bool = False,
        checkpoint_path: Optional[str] = None,
        decoder_channels: int = 256,
        patch_size: int = 10,
        patch_scale: int = 1,
        spatial_resolution: Optional[float] = None,
        flash_attn: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = AnySatBackbone(
            in_channels=in_channels,
            model_size=model_size,
            patch_size=patch_size,
            patch_scale=patch_scale,
            spatial_resolution=spatial_resolution,
            flash_attn=flash_attn,
            pretrained=pretrained_backbone,
            checkpoint_path=checkpoint_path,
        )

        self._freeze_backbone = bool(freeze_backbone)
        if freeze_backbone:
            missing_projector_names = self.backbone.freeze_pretrained_encoder()
            self.backbone.eval()
            if missing_projector_names:
                print(
                    "[AnySat] Frozen pretrained encoder; kept projector_aerial "
                    "trainable because checkpoint parameters were missing: "
                    f"{missing_projector_names}"
                )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, int(decoder_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(decoder_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(int(decoder_channels), num_classes, kernel_size=1),
        )

    def train(self, mode: bool = True):
        """Keep a frozen AnySat encoder deterministic during head training."""
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        patch_tokens = self.backbone(x, output="patch")
        if patch_tokens.dim() != 4:
            raise RuntimeError(f"AnySat patch output must be 4D, got shape {tuple(patch_tokens.shape)}")

        # AnySat patch output is (B, H_patch, W_patch, C)
        feat = patch_tokens.permute(0, 3, 1, 2).contiguous()
        logits = self.decoder(feat)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)
        return logits
