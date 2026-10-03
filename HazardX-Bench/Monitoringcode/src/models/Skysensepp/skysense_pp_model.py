# coding: utf-8
"""SkySense++ core model — standalone nn.Module (no antmmf dependency)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head


class SkySensePPModel(nn.Module):
    """SkySense++ segmentation model adapted for AnyDisasterMapping.

    Supports:
    - ``sources=['hr']``: single high-resolution input (compatible with existing seg datasets)
    - ``sources=['hr', 's2', 's1']``: full multi-modal fusion pipeline
    """

    def __init__(
        self,
        *,
        sources: Sequence[str] = ('hr',),
        num_classes: int = 2,
        vocabulary_size: int = 64,
        use_modal_vae: bool = False,
        use_ctpe: bool = False,
        calendar_time: int = 365,
        upscale_results: bool = True,
        backbone_hr: Optional[Dict] = None,
        backbone_s2: Optional[Dict] = None,
        backbone_s1: Optional[Dict] = None,
        head_s2: Optional[Dict] = None,
        head_s1: Optional[Dict] = None,
        rec_head_hr: Optional[Dict] = None,
        necks: Optional[Dict] = None,
        modality_vae: Optional[Dict] = None,
        cls_token_channels: int = 1024,
    ) -> None:
        super().__init__()
        self.sources = list(sources)
        self.vocabulary_size = vocabulary_size
        self.use_modal_vae = use_modal_vae
        self.use_ctpe = use_ctpe
        self.calendar_time = calendar_time
        self.upscale_results = upscale_results
        self.cls_token_channels = cls_token_channels

        if 'hr' in self.sources:
            hr_cfg = dict(backbone_hr or _default_backbone_hr(vocabulary_size))
            self.backbone_hr = build_backbone(hr_cfg.pop('type'), **hr_cfg)
        if 's2' in self.sources:
            s2_cfg = dict(backbone_s2 or _default_backbone_s2(vocabulary_size))
            self.backbone_s2 = build_backbone(s2_cfg.pop('type'), **s2_cfg)
            if use_ctpe:
                neck_cfg = necks or _default_necks()
                self.ctpe = nn.Parameter(
                    torch.zeros(1, calendar_time, neck_cfg['input_dims']))
            if head_s2 is not None:
                h2 = dict(head_s2)
                self.head_s2 = build_head(h2.pop('type'), **h2)
            elif 's2' in self.sources:
                h2 = _default_head_s2()
                self.head_s2 = build_head(h2.pop('type'), **h2)
            neck_cfg = dict(necks or _default_necks())
            self.fusion = build_neck(neck_cfg.pop('type'), **neck_cfg)
        if 's1' in self.sources:
            s1_cfg = dict(backbone_s1 or _default_backbone_s1(vocabulary_size))
            self.backbone_s1 = build_backbone(s1_cfg.pop('type'), **s1_cfg)
            if head_s1 is not None:
                h1 = dict(head_s1)
                self.head_s1 = build_head(h1.pop('type'), **h1)
            elif 's1' in self.sources:
                h1 = _default_head_s1()
                self.head_s1 = build_head(h1.pop('type'), **h1)

        rec_cfg = dict(rec_head_hr or _default_rec_head_hr(num_classes))
        self.head_rec_hr = build_head(rec_cfg.pop('type'), **rec_cfg)

        if use_modal_vae and len(self.sources) > 1:
            vae_cfg = dict(modality_vae or _default_modality_vae())
            self.modality_vae = build_neck(vae_cfg.pop('type'), **vae_cfg)

        # HR-only path: project stage-3 features to cls_token channels expected by UPerHead
        if 'hr' in self.sources and 's2' not in self.sources:
            self.cls_token_proj = nn.Conv2d(2816, cls_token_channels, kernel_size=1)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _prepare_anno(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build annotation token map expected by SwinTransformerV2MSL."""
        b, _, h, w = x.shape
        if label is not None:
            anno = label
            if anno.shape[-2:] != (h, w):
                anno = F.interpolate(
                    anno.unsqueeze(1).float(),
                    size=(h, w),
                    mode='nearest',
                ).squeeze(1).long()
            anno = anno.clamp(0, self.vocabulary_size)
            return anno

        return torch.zeros(b, h, w, dtype=torch.long, device=x.device)

    @staticmethod
    def _prepare_anno_mask(
        hr_img: torch.Tensor,
        anno_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _, h, w = hr_img.shape
        if anno_mask is None:
            return torch.zeros(b, h, w, dtype=hr_img.dtype, device=hr_img.device)
        if anno_mask.shape[-2:] != (h, w):
            anno_mask = F.interpolate(
                anno_mask.unsqueeze(1).float(),
                size=(h, w),
                mode='nearest',
            ).squeeze(1)
        return anno_mask

    def _split_stage1(self, hr_features: List[torch.Tensor]) -> List[torch.Tensor]:
        feat_stage1 = hr_features[0]
        if feat_stage1.shape[-1] % 2 == 0:
            left, right = torch.split(feat_stage1, feat_stage1.shape[-1] // 2, dim=-1)
            merged = torch.cat((left, right), dim=1)
            out = list(hr_features)
            out[0] = merged
            return out
        return hr_features

    def _forward_hr_only(
        self,
        hr_img: torch.Tensor,
        label: Optional[torch.Tensor],
        anno_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        anno = self._prepare_anno(hr_img, label)
        mask = self._prepare_anno_mask(hr_img, anno_mask)
        hr_features = self.backbone_hr(hr_img, anno, mask)
        # print(hr_img.shape)
        # print(mask.shape)

        cls_token = self.cls_token_proj(hr_features[-1])
        hr_rec_inputs = self._split_stage1(hr_features)
        logits = self.head_rec_hr([*hr_rec_inputs, cls_token])
        if self.upscale_results:
            logits = F.interpolate(
                logits.float(), scale_factor=4, mode='bilinear', align_corners=True)
        return logits

    def _forward_multimodal(
        self,
        hr_img: torch.Tensor,
        s2_img: torch.Tensor,
        s1_img: torch.Tensor,
        label: Optional[torch.Tensor],
        anno_mask: Optional[torch.Tensor],
        s2_ct: Optional[torch.Tensor] = None,
        s2_ct2: Optional[torch.Tensor] = None,
        modality_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        anno = self._prepare_anno(hr_img, label)
        if anno_mask is None:
            h_mask = max(1, hr_img.shape[-2] // 32)
            w_mask = max(1, hr_img.shape[-1] // 32)
            anno_mask = torch.zeros(hr_img.shape[0], h_mask, w_mask,
                                    dtype=hr_img.dtype, device=hr_img.device)
        block_size = 32
        b_mask, h_mask, w_mask = anno_mask.shape
        anno_mask_hr = anno_mask.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, 1, block_size, block_size)
        anno_mask_hr = anno_mask_hr.permute(0, 1, 3, 2, 4).reshape(
            b_mask, h_mask * block_size, w_mask * block_size).contiguous()

        anno_s2 = F.interpolate(
            anno.unsqueeze(1).float(),
            size=anno_mask.shape[-2:],
            mode='nearest',
        ).squeeze(1).long()
        anno_s1 = anno_s2

        hr_features = self.backbone_hr(hr_img, anno, anno_mask_hr)

        b, c_s2, s_s2, h_s2, w_s2 = s2_img.shape
        s2_flat = s2_img.permute(0, 2, 1, 3, 4).reshape(
            b * s_s2, c_s2, h_s2, w_s2).contiguous()
        s2_features = self.backbone_s2(s2_flat, anno_s2, anno_mask)
        if hasattr(self, 'head_s2'):
            s2_features = self.head_s2(s2_features[-1])
            s2_features = [s2_features]

        b, c_s1, s_s1, h_s1, w_s1 = s1_img.shape
        s1_flat = s1_img.permute(0, 2, 1, 3, 4).reshape(
            b * s_s1, c_s1, h_s1, w_s1).contiguous()
        s1_features = self.backbone_s1(s1_flat, anno_s1, anno_mask)
        if hasattr(self, 'head_s1'):
            s1_features = self.head_s1(s1_features[-1])
            s1_features = [s1_features]

        hr_features_stage3 = hr_features[-1]
        s2_features_stage3 = s2_features[-1]
        s1_features_stage3 = s1_features[-1]

        if modality_flags is None:
            modality_flags = torch.ones(b, 3, device=hr_img.device)

        if self.use_modal_vae:
            vae_out = self.modality_vae(
                hr_features_stage3, s2_features_stage3, s1_features_stage3, modality_flags)
            hr_features_stage3 = vae_out['hr_out']
            s2_features_stage3 = vae_out['s2_out']
            s1_features_stage3 = vae_out['s1_out']

        _, c3_g, h3_g, w3_g = hr_features_stage3.shape
        hr_stage3 = hr_features_stage3.permute(0, 2, 3, 1).reshape(
            b * h3_g * w3_g, 1, c3_g).contiguous()
        features_stage3 = hr_stage3

        _, c3_s2, h3_s2, w3_s2 = s2_features_stage3.shape
        s2_stage3 = s2_features_stage3.reshape(
            b, s_s2, c3_s2, h3_s2, w3_s2).permute(0, 3, 4, 1, 2).reshape(
            b, h3_s2 * w3_s2, s_s2, c3_s2).contiguous()
        if self.use_ctpe and s2_ct is not None and s2_ct2 is not None:
            ctpe = self.ctpe[:, s2_ct, :].permute(1, 0, 2, 3).expand(-1, 256, -1, -1)
            ctpe2 = self.ctpe[:, s2_ct2, :].permute(1, 0, 2, 3).expand(-1, 256, -1, -1)
            s2_stage3 = (s2_stage3 + torch.cat([ctpe, ctpe2], dim=1)).reshape(
                b * h3_s2 * w3_s2, s_s2, c3_s2).contiguous()
        else:
            s2_stage3 = s2_stage3.reshape(
                b * h3_s2 * w3_s2, s_s2, c3_s2).contiguous()
        features_stage3 = torch.cat((features_stage3, s2_stage3), dim=1)

        _, c3_s1, h3_s1, w3_s1 = s1_features_stage3.shape
        s1_stage3 = s1_features_stage3.reshape(
            b, s_s1, c3_s1, h3_s1, w3_s1).permute(0, 3, 4, 1, 2).reshape(
            b, h3_s1 * w3_s1, s_s1, c3_s1).contiguous()
        s1_stage3 = s1_stage3.reshape(
            b * h3_s1 * w3_s1, s_s1, c3_s1).contiguous()
        features_stage3 = torch.cat((features_stage3, s1_stage3), dim=1)

        cls_token = self.fusion(features_stage3)
        _, c_cls = cls_token.shape
        cls_token = cls_token.reshape(b, h3_g, w3_g, c_cls).permute(0, 3, 1, 2).contiguous()

        hr_rec_inputs = self._split_stage1(hr_features)
        logits = self.head_rec_hr([*hr_rec_inputs, cls_token])
        if self.upscale_results:
            logits = F.interpolate(
                logits.float(), scale_factor=4, mode='bilinear', align_corners=True)
        return logits

    def forward(
        self,
        hr_img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        *,
        s2_img: Optional[torch.Tensor] = None,
        s1_img: Optional[torch.Tensor] = None,
        anno_mask: Optional[torch.Tensor] = None,
        s2_ct: Optional[torch.Tensor] = None,
        s2_ct2: Optional[torch.Tensor] = None,
        modality_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(self.sources) == 1 and self.sources[0] == 'hr':
            return self._forward_hr_only(hr_img, label, anno_mask)
        if s2_img is None or s1_img is None:
            raise ValueError(
                'Multi-modal SkySensePP requires s2_img and s1_img when sources include s2/s1.')
        return self._forward_multimodal(
            hr_img, s2_img, s1_img, label, anno_mask, s2_ct, s2_ct2, modality_flags)


# ------------------------------------------------------------------
# Default architecture configs (from SkySense++ pretrain/eval yaml)
# ------------------------------------------------------------------

def _default_backbone_hr(vocabulary_size: int) -> Dict:
    return dict(
        type='SwinTransformerV2MSL',
        arch='huge',
        use_attn=True,
        merge_stage=2,
        vocabulary_size=vocabulary_size,
        img_size=128,
        patch_size=4,
        in_channels=14,
        window_size=2,
        drop_rate=0.0,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        use_abs_pos_embed=False,
        interpolate_mode='bicubic',
        with_cp=True,
        frozen_stages=-1,
        norm_eval=False,
        pad_small_map=False,
        pretrained_window_sizes=[0, 0, 0, 0],
    )


def _default_backbone_s2(vocabulary_size: int) -> Dict:
    return dict(
        type='VisionTransformerMSL',
        img_size=(16, 16),
        use_attn=False,
        merge_stage=4,
        vocabulary_size=vocabulary_size,
        patch_size=4,
        in_channels=10,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(5, 11, 17, 23),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        with_cls_token=False,
        output_cls_token=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        with_cp=True,
        interpolate_mode='bicubic',
    )


def _default_backbone_s1(vocabulary_size: int) -> Dict:
    cfg = _default_backbone_s2(vocabulary_size)
    cfg = dict(cfg)
    cfg['in_channels'] = 2
    return cfg


def _default_head_s2() -> Dict:
    return dict(type='UPHead', in_dim=1024, out_dim=2816, up_scale=4)


def _default_head_s1() -> Dict:
    return _default_head_s2()


def _default_necks() -> Dict:
    return dict(
        type='TransformerEncoder',
        input_dims=2816,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        with_cls_token=True,
        output_cls_token=True,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        num_fcs=2,
        norm_eval=False,
        with_cp=True,
    )


def _default_modality_vae() -> Dict:
    return dict(
        type='ModalityCompletion',
        input_shape_hr=[2816, 32, 16],
        input_shape_s2=[2816, 32, 16],
        input_shape_s1=[2816, 32, 16],
        conv_dim=256,
        z_dim=256,
        n_codebook=8192,
    )


def _default_rec_head_hr(num_classes: int) -> Dict:
    return dict(
        type='UPerHead',
        in_channels=[704, 704, 1408, 2816, 1024],
        in_index=[0, 1, 2, 3, 4],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        # GroupNorm remains well-defined for the batch-size-1 regime required
        # by full SkySense++ fine-tuning on 48 GB GPUs. BatchNorm fails in the
        # PPM 1x1 pooling branch because it sees only one value per channel.
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
    )
