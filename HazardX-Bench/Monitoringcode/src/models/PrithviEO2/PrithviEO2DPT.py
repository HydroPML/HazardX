import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels


_PRITHVI_VARIANTS = {
    'prithvi-eo-2.0-300m': dict(embed_dim=1024, depth=24, num_heads=16, patch_size=(1, 16, 16), num_frames=1),
    'prithvi-eo-2.0-300m-tl': dict(embed_dim=1024, depth=24, num_heads=16, patch_size=(1, 16, 16), num_frames=1,
                                   coords_encoding=['time', 'location'], coords_scale_learn=True),
    'prithvi-eo-2.0-600m': dict(embed_dim=1280, depth=32, num_heads=16, patch_size=(1, 14, 14), num_frames=1),
    'prithvi-eo-2.0-600m-tl': dict(embed_dim=1280, depth=32, num_heads=16, patch_size=(1, 14, 14), num_frames=1,
                                   coords_encoding=['time', 'location'], coords_scale_learn=True),
}


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f'Expected a tuple of length 2, got {value}.')
        return int(value[0]), int(value[1])
    return int(value), int(value)


class PrithviEO2FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = 'prithvi-eo-2.0-300m',
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        num_frames: int = 1,
        temporal_coords: bool = False,
        location_coords: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        try:
            from .prithvi_mae import PrithviViT
        except Exception as exc:  # pragma: no cover - dependency-specific failure
            raise ImportError(
                'PrithviEO2 requires terratorch with terratorch.models.backbones.prithvi_mae.PrithviViT installed.'
            ) from exc

        key = backbone.lower()
        if key not in _PRITHVI_VARIANTS:
            raise ValueError(f'Unsupported PrithviEO2 backbone "{backbone}".')

        cfg = dict(_PRITHVI_VARIANTS[key])
        cfg.update(backbone_kwargs)
        cfg.setdefault('img_size', 224)
        cfg.setdefault('patch_size', (1, 16, 16) if '600' not in key else (1, 14, 14))
        cfg.setdefault('num_frames', num_frames)
        cfg.setdefault('in_chans', in_channels)
        cfg.setdefault('embed_dim', 1024 if '300' in key else 1280)
        cfg.setdefault('depth', 24 if '300' in key else 32)
        cfg.setdefault('num_heads', 16)
        cfg.setdefault('mlp_ratio', 4.0)
        cfg.setdefault('norm_layer', nn.LayerNorm)
        coords_encoding = list(cfg.pop('coords_encoding', []))
        if temporal_coords and 'time' not in coords_encoding:
            coords_encoding.append('time')
        if location_coords and 'location' not in coords_encoding:
            coords_encoding.append('location')
        cfg['coords_encoding'] = coords_encoding or None

        self.model = PrithviViT(**cfg)
        self.embed_dim = int(getattr(self.model, 'embed_dim'))
        self.patch_size = _to_2tuple(getattr(self.model.patch_embed, 'patch_size', (1, 16, 16))[1:])
        self.num_frames = int(getattr(self.model, 'num_frames', num_frames))

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('PrithviEO2 backbone exposes no transformer blocks.')

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
            try:
                try:
                    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                except TypeError:
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
                    warnings.warn(
                        f'Loaded PrithviEO2 backbone with missing keys: {missing} and unexpected keys: {unexpected}.',
                        stacklevel=2,
                    )
            except FileNotFoundError:
                raise
            except Exception as exc:
                warnings.warn(f'Failed to load PrithviEO2 checkpoint {checkpoint_path}: {exc}', stacklevel=2)

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _reshape_tokens(tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.permute(0, 2, 1).reshape(batch, channels, patch_h, patch_w)

    def forward(self, x: torch.Tensor) -> Tuple[list[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')

        if x.shape[1] != self.model.in_chans:
            raise ValueError(
                f'PrithviEO2 backbone expects {self.model.in_chans} input channels, got {x.shape[1]}.')

        if x.shape[-1] % self.patch_size[1] != 0 or x.shape[-2] % self.patch_size[0] != 0:
            warnings.warn(
                f'Input spatial size {tuple(x.shape[-2:])} is not divisible by patch size {self.patch_size}; '
                'PrithviEO2 will ignore the border unless padding is enabled in the caller.',
                stacklevel=2,
            )

        if x.dim() == 4 and self.model.patch_embed.input_size[0] == 1:
            x = x.unsqueeze(2)

        temporal_coords = None
        location_coords = None
        features = self.model.forward_features(x, temporal_coords, location_coords)
        if not isinstance(features, list) or not features:
            raise RuntimeError('PrithviViT forward_features did not return a feature list.')

        height, width = x.shape[-2:]
        patch_h = max(1, height // self.patch_size[0])
        patch_w = max(1, width // self.patch_size[1])
        num_tokens = patch_h * patch_w

        prepared_features: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx in self.hook_indices:
            feat = features[idx]
            if feat.dim() != 3:
                raise RuntimeError(f'Unexpected Prithvi feature rank {feat.dim()} at hook index {idx}.')
            if feat.shape[1] == num_tokens + 1:
                cls_token = feat[:, 0, :]
                patch_tokens = feat[:, 1:, :]
            elif feat.shape[1] == num_tokens:
                cls_token = None
                patch_tokens = feat
            else:
                raise RuntimeError(
                    f'Prithvi feature token count {feat.shape[1]} does not match expected grid {num_tokens}.'
                )
            prepared_features.append((patch_tokens, cls_token))

        if patch_h * patch_w != prepared_features[0][0].shape[1]:
            raise RuntimeError('Failed to infer patch grid size from PrithviEO2 tokens.')

        return prepared_features, (patch_h, patch_w)


class PrithviEO2DPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        backbone: str = 'prithvi-eo-2.0-300m',
        checkpoint_path: Optional[str] = None,
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        num_frames: int = 1,
        temporal_coords: bool = False,
        location_coords: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = PrithviEO2FeatureExtractor(
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
            num_frames=num_frames,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
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
            use_cls_token=True,
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
