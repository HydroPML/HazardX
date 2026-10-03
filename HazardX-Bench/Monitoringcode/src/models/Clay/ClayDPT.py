"""Clay v1.5 encoder adapted for AnyDisasterMapping semantic segmentation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .backbone import Transformer
from .factory import DynamicEmbedding
from .utils import posemb_sincos_2d_with_gsd


_CLAY_VARIANTS = {
    'clay-v1.5-tiny': dict(dim=192, depth=6, heads=4, dim_head=48, mlp_ratio=2),
    'clay-v1.5-small': dict(dim=384, depth=6, heads=6, dim_head=64, mlp_ratio=2),
    'clay-v1.5-base': dict(dim=768, depth=12, heads=12, dim_head=64, mlp_ratio=4),
    'clay-v1.5-large': dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4),
}


class SegmentEncoder(nn.Module):
    """Clay's finetune/segment encoder: unmasked patches plus a class token."""

    def __init__(self, patch_size: int, **cfg) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.dim = int(cfg['dim'])
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.patch_embedding = DynamicEmbedding(
            wave_dim=128, num_latent_tokens=128, patch_size=self.patch_size,
            embed_dim=self.dim, is_decoder=False,
        )
        self.transformer = Transformer(
            dim=self.dim, depth=cfg['depth'], heads=cfg['heads'], dim_head=cfg['dim_head'],
            mlp_dim=int(self.dim * cfg['mlp_ratio']), fused_attn=True,
        )

    def forward(self, datacube: dict[str, torch.Tensor | str]) -> torch.Tensor:
        """Encode the TerraTorch Clay v1.5-style datacube."""
        pixels = datacube['pixels']
        time = datacube['time']
        latlon = datacube['latlon']
        gsd = datacube['gsd']
        waves = datacube['waves']
        batch, _, height, width = pixels.shape
        patches, _ = self.patch_embedding(pixels, waves)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        if grid_h * grid_w != patches.shape[1]:
            raise ValueError('Clay patch embedding produced an unexpected token grid.')
        pos = posemb_sincos_2d_with_gsd(grid_h, grid_w, self.dim - 8, gsd=gsd).to(pixels.device)
        metadata = torch.cat((time, latlon), dim=-1).to(pixels.device)
        encoding = torch.cat(
            (repeat(pos, 'l d -> b l d', b=batch), repeat(metadata, 'b d -> b l d', l=patches.shape[1])),
            dim=-1,
        )
        patches = patches + encoding
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch)
        return self.transformer(torch.cat((cls, patches), dim=1))[:, 1:]


class ClayDPT(nn.Module):
    """Clay's native finetune/segment PixelShuffle segmentation head."""

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 2,
        backbone: str = 'clay-v1.5-large',
        checkpoint_path: Optional[str] = None,
        patch_size: int = 8,
        platform: str = 'l4s',
        wavelengths: Optional[Sequence[float]] = None,
        gsd: float = 10.0,
        month: int = 6,
        hidden_dim: int = 512,
        decoder_channels: int = 64,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        key = backbone.lower()
        if key not in _CLAY_VARIANTS:
            raise ValueError(f'Unsupported Clay backbone "{backbone}".')
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.platform = str(platform)
        values = tuple(float(v) for v in (wavelengths or ()))
        if len(values) != self.in_channels:
            raise ValueError('wavelengths must provide one value for every input channel.')
        self.register_buffer('waves', torch.tensor(values, dtype=torch.float32), persistent=False)
        self.register_buffer('gsd', torch.tensor(float(gsd), dtype=torch.float32), persistent=False)
        self.month = int(month)
        self.encoder = SegmentEncoder(patch_size=self.patch_size, **_CLAY_VARIANTS[key])
        self.embed_dim = self.encoder.dim
        # Matches claymodel/finetune/segment/factory.py::Segmentor.
        self.conv1 = nn.Conv2d(self.embed_dim, int(hidden_dim), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(hidden_dim))
        self.conv2 = nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(int(hidden_dim))
        self.conv_ps = nn.Conv2d(
            int(hidden_dim), int(decoder_channels) * self.patch_size ** 2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=self.patch_size)
        self.conv_out = nn.Conv2d(int(decoder_channels), num_classes, kernel_size=3, padding=1)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        if freeze_backbone:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f'Clay checkpoint does not exist: {path}')
        state = torch.load(path, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        encoder_state = {
            name.removeprefix('model.encoder.'): value
            for name, value in state.items() if name.startswith('model.encoder.')
        }
        if not encoder_state:
            raise ValueError('Checkpoint does not contain Clay encoder weights under model.encoder.')
        model_state = self.encoder.state_dict()
        compatible = {
            name: value for name, value in encoder_state.items()
            if name in model_state and value.shape == model_state[name].shape
        }
        self.encoder.load_state_dict(compatible, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'ClayDPT expects BCHW input with {self.in_channels} channels.')
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(f'Input size must be divisible by patch_size={self.patch_size}.')
        batch, _, height, width = x.shape
        time = torch.zeros(batch, 4, dtype=x.dtype, device=x.device)
        time[:, 1] = math.sin(2 * math.pi * self.month / 12)
        time[:, 3] = math.cos(2 * math.pi * self.month / 12)
        latlon = torch.zeros(batch, 4, dtype=x.dtype, device=x.device)
        # This matches Clay1_5ModelFactory.ModelWrapper's input convention,
        # while keeping AnyDisasterMapping's tensor-only model API.
        datacube: dict[str, torch.Tensor | str] = {
            'pixels': x,
            'time': time,
            'platform': self.platform,
            'latlon': latlon,
            'waves': self.waves.to(x.device),
            'gsd': self.gsd.to(x.device),
        }
        tokens = self.encoder(datacube)
        features = tokens.transpose(1, 2).reshape(batch, self.embed_dim, height // self.patch_size, width // self.patch_size)
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.pixel_shuffle(self.conv_ps(features))
        return self.conv_out(features)
