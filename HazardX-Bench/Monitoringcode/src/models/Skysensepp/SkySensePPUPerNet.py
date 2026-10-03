"""SkySense++ wrapper for AnyDisasterMapping — mirrors SkySenseUPerNet interface."""

from __future__ import annotations

import re
import warnings
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skysense_pp_model import SkySensePPModel


class SkySensePPUPerNet(nn.Module):
    """SkySense++ segmentation model with optional multi-modal inputs.

    Default ``sources=['hr']`` works with standard AnyDisasterMapping seg datasets
    via ``forward(x) -> logits``.  Set ``sources=['hr', 's2', 's1']`` and pass
    extra tensors through ``forward_multimodal`` for the full SkySense++ pipeline.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 512,
        sources: Sequence[str] = ('hr', 's2', 's1'),
        vocabulary_size: int = 64,
        use_modal_vae: bool = False,
        use_ctpe: bool = False,
        pretrained: bool = True,
        pretrained_backbone_path: Optional[Union[str, Path]] = None,
        freeze_backbone: bool = False,
        upscale_results: bool = True,
        encoder_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.sources = list(sources)

        backbone_hr = _build_backbone_hr_cfg(in_channels, vocabulary_size, encoder_kwargs)
        extra = dict(model_kwargs or {})
        self.model = SkySensePPModel(
            sources=sources,
            num_classes=num_classes,
            vocabulary_size=vocabulary_size,
            use_modal_vae=use_modal_vae,
            use_ctpe=use_ctpe,
            upscale_results=upscale_results,
            backbone_hr=backbone_hr,
            **extra,
        )

        if freeze_backbone and hasattr(self.model, 'backbone_hr'):
            for param in self.model.backbone_hr.parameters():
                param.requires_grad = False

        if pretrained:
            ckpt_path = self._resolve_pretrained_path(pretrained_backbone_path)
            if ckpt_path.is_file():
                self._load_pretrained(ckpt_path)
            else:
                warnings.warn(
                    f'Pretrained weights requested but checkpoint not found at {ckpt_path}. '
                    'Training will proceed without loading weights.',
                    stacklevel=2,
                )

    @staticmethod
    def _resolve_pretrained_path(path: Optional[Union[str, Path]]) -> Path:
        if path is not None:
            return Path(path).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / 'pretrained_weight' / 'skysensepp_release_hr.pth'

    def _load_pretrained(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f'[SkySensePP] Loaded checkpoint keys from {checkpoint_path.name}: '
                  f'{list(checkpoint.keys())[:8]}')
            for key in ('model', 'state_dict', 'SkySensePP'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        state_dict = self._remap_checkpoint_keys(dict(checkpoint))
        model_state = self.model.state_dict()
        loadable: Dict[str, torch.Tensor] = {}
        skipped = []
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None:
                skipped.append(key)
                continue
            if value.shape != target.shape:
                # The released HR checkpoint uses RGB, while disaster change
                # detection concatenates the two acquisition dates.  Repeating
                # and averaging preserves the pretrained filter response scale.
                if (key.endswith('patch_embed.projection.weight')
                        and value.ndim == target.ndim == 4
                        and value.shape[0] == target.shape[0]
                        and value.shape[2:] == target.shape[2:]):
                    repeats = math.ceil(target.shape[1] / value.shape[1])
                    value = value.repeat(1, repeats, 1, 1)[:, :target.shape[1]] / repeats
                elif (key.endswith('vocabulary_token')
                      and value.ndim == target.ndim == 2
                      and value.shape[1] == target.shape[1]
                      and value.shape[0] >= target.shape[0]):
                    value = value[:target.shape[0]]
                else:
                    skipped.append(key)
                    continue
            loadable[key] = value

        incompatible = self.model.load_state_dict(loadable, strict=False)
        missing = getattr(incompatible, 'missing_keys', ())
        unexpected = getattr(incompatible, 'unexpected_keys', ())
        print(f'[SkySensePP] Loaded {len(loadable)}/{len(state_dict)} compatible '
              f'checkpoint tensors; skipped {len(skipped)} shape-incompatible tensors.')
        if unexpected:
            warnings.warn(
                f'[SkySensePP] Loaded with missing keys ({len(missing)}) and '
                f'unexpected keys ({len(unexpected)}). '
                f'Sample missing: {missing[:5]}',
                stacklevel=2,
            )

    @staticmethod
    def _remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remap official SkySense++ keys to this module layout."""
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = re.sub(r'^(?:module\.|model\.)+', '', key)
            known_roots = (
                'backbone_hr.', 'backbone_s2.', 'backbone_s1.',
                'head_s2.', 'head_s1.', 'head_rec_hr.', 'fusion.',
                'modality_vae.', 'ctpe', 'decode_head.', 'auxiliary_head.',
            )
            # The official HR-only checkpoint stores bare backbone keys such
            # as ``patch_embed.*`` and ``stages.*``.
            if not new_key.startswith(known_roots):
                new_key = 'backbone_hr.' + new_key

            remapped[new_key] = value
        return remapped

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Standard AnyDisasterMapping interface: ``model(image) -> logits``."""
        logits = self.model(x, label=labels, **kwargs)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

    def forward_multimodal(
        self,
        hr_img: torch.Tensor,
        s2_img: torch.Tensor,
        s1_img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Full SkySense++ forward with HR + Sentinel-2 + Sentinel-1 inputs."""
        logits = self.model(
            hr_img, label=label, s2_img=s2_img, s1_img=s1_img, **kwargs)
        if logits.shape[-2:] != hr_img.shape[-2:]:
            logits = F.interpolate(
                logits, size=hr_img.shape[2:], mode='bilinear', align_corners=False)
        return logits


def _build_backbone_hr_cfg(
    in_channels: int,
    vocabulary_size: int,
    encoder_kwargs: Optional[Dict],
) -> Dict:
    from .skysense_pp_model import _default_backbone_hr

    cfg = _default_backbone_hr(vocabulary_size)
    cfg['in_channels'] = in_channels
    if encoder_kwargs:
        cfg.update(encoder_kwargs)
    return cfg
