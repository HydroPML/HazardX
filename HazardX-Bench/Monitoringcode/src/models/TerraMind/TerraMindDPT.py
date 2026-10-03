from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels
from .tm_utils import LayerNorm

_TERRAMIND_ROOT = Path('/root/workcsc/terratorch')
if _TERRAMIND_ROOT.exists() and str(_TERRAMIND_ROOT) not in sys.path:
    sys.path.insert(0, str(_TERRAMIND_ROOT))

try:
    from .terramind_vit import TerraMindViT
except Exception as exc:  # pragma: no cover - dependency-specific failure
    TerraMindViT = None
    _TERRAMIND_IMPORT_ERROR = exc
else:
    _TERRAMIND_IMPORT_ERROR = None


_TERRAMIND_VARIANTS = {
    # These must match the published TerraMind v1 checkpoint architectures.
    'terramind-v1-tiny': dict(dim=192, encoder_depth=12, num_heads=3, patch_size=16),
    'terramind-v1-small': dict(dim=384, encoder_depth=12, num_heads=6, patch_size=16),
    'terramind-v1-base': dict(
        dim=768, encoder_depth=12, num_heads=12, patch_size=16,
        qkv_bias=False, proj_bias=False, mlp_bias=False,
        act_layer=nn.SiLU, gated_mlp=True,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
    ),
    'terramind-v1-large': dict(
        dim=1024, encoder_depth=24, num_heads=16, patch_size=16,
        qkv_bias=False, proj_bias=False, mlp_bias=False,
        act_layer=nn.SiLU, gated_mlp=True,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
    ),
}


class TerraMindFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = 'terramind-v1-base',
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        if TerraMindViT is None:
            raise ImportError(
                'TerraMind requires terratorch.models.backbones.terramind to be available.'
            ) from _TERRAMIND_IMPORT_ERROR

        key = backbone.lower()
        if key not in _TERRAMIND_VARIANTS:
            raise ValueError(f'Unsupported TerraMind backbone "{backbone}".')

        cfg = dict(_TERRAMIND_VARIANTS[key])
        cfg.update(backbone_kwargs)
        cfg.setdefault('img_size', 224)
        cfg.setdefault('patch_size', 16)
        cfg.setdefault('in_chans', in_channels)
        cfg.setdefault('modalities', {'image': in_channels})
        cfg.setdefault('merge_method', 'mean')
        cfg.setdefault('pretrained', False)

        self.model = TerraMindViT(**cfg)
        self.embed_dim = int(getattr(self.model, 'dim', cfg['dim']))
        self.patch_size = int(cfg['patch_size'])

        depth = len(getattr(self.model, 'encoder', []))
        if depth == 0:
            raise ValueError('TerraMind backbone exposes no transformer blocks.')

        if hook_indices is None:
            if num_features <= 0:
                raise ValueError('num_features must be positive when hook_indices is not provided.')
            step = depth / float(num_features)
            derived = []
            for i in range(1, num_features + 1):
                idx = int(round(i * step) - 1)
                idx = max(0, min(depth - 1, idx))
                derived.append(idx)
            hook_indices = tuple(sorted(set(derived)))
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected backbone.')
            hook_indices = tuple(sorted(set(filtered)))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state, dict):
                for key_name in ('model', 'state_dict', 'encoder'):
                    if key_name in state:
                        state = state[key_name]
                        break
            load_result = self.model.load_state_dict(state, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            if missing or unexpected:
                print(f'[TerraMind] Missing keys: {missing}')
                print(f'[TerraMind] Unexpected keys: {unexpected}')

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[list[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')

        outputs = self.model(x)
        if not isinstance(outputs, list) or not outputs:
            raise RuntimeError('TerraMindViT forward did not return a feature list.')

        patch_h = max(1, int(round((x.shape[-2] + self.patch_size - 1) / self.patch_size)))
        patch_w = max(1, int(round((x.shape[-1] + self.patch_size - 1) / self.patch_size)))

        prepared_features: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx in self.hook_indices:
            feat = outputs[idx]
            if not isinstance(feat, torch.Tensor) or feat.dim() != 3:
                raise RuntimeError(f'Unexpected TerraMind feature at hook index {idx}: {type(feat)}.')

            token_count = feat.shape[1]
            cls_token = None
            patch_tokens = feat
            if token_count == patch_h * patch_w + 1:
                cls_token = feat[:, 0, :]
                patch_tokens = feat[:, 1:, :]
            elif token_count != patch_h * patch_w:
                inferred = int(round(token_count ** 0.5))
                if inferred * inferred == token_count:
                    patch_h = inferred
                    patch_w = inferred
                else:
                    raise RuntimeError(
                        f'TerraMind feature token count {token_count} does not match the expected grid.'
                    )

            prepared_features.append((patch_tokens, cls_token))

        return prepared_features, (patch_h, patch_w)


class TerraMindDPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        backbone: str = 'terramind-v1-base',
        checkpoint_path: Optional[str] = None,
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = TerraMindFeatureExtractor(
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
            **backbone_kwargs,
        )

        stage_channels = tuple(int(c) for c in (
            decoder_stage_channels if decoder_stage_channels is not None else _default_stage_channels(self.encoder.embed_dim)
        ))
        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels

        self.decoder = DPTDecoder(
            embed_dim=self.encoder.embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=False,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)

        self.hook_indices = self.encoder.hook_indices
        self.num_features = len(self.hook_indices)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, patch_shape = self.encoder(x)
        return self.decoder(features, patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        decoded = self.forward_features(x)
        refined = self.head(decoded)
        logits = self.classifier(refined)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits
