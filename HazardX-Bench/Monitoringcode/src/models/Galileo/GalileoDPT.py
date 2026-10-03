"""Galileo encoder with a DPT semantic-segmentation decoder.

The Galileo project is kept as a sibling checkout.  It is loaded under a
private module name so its own ``src`` package does not collide with this
project's ``src`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels
from . import galileo_core as galileo


_GALILEO_VARIANTS = {
    'galileo-nano': dict(
        max_patch_size=8, embedding_size=128, depth=4, num_heads=8,
        mlp_ratio=4, max_sequence_length=24, freeze_projections=False, drop_path=0.1,
    ),
}


class GalileoDPT(nn.Module):
    """Adapt Galileo's Sentinel-2 and topographic inputs to BCHW segmentation.

    L4S supplies 12 Sentinel-2 bands followed by ALOS PALSAR slope and DEM.
    The latter pair is routed to Galileo's pretrained SRTM slope/elevation
    spatial group, so all 14 input channels are used by the encoder.
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 2,
        backbone: str = 'galileo-nano',
        checkpoint_path: Optional[str] = None,
        patch_size: int = 8,
        month: int = 6,
        s2_channel_indices: Optional[Sequence[int]] = None,
        s2_band_names: Sequence[str] = (
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
        ),
        topographic_channel_indices: Optional[Sequence[int]] = (12, 13),
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        freeze_backbone: bool = False,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.s2_channel_indices = tuple(
            range(12) if s2_channel_indices is None else (int(i) for i in s2_channel_indices)
        )
        self.s2_band_names = tuple(str(name) for name in s2_band_names)
        self.topographic_channel_indices = tuple(
            int(i) for i in (topographic_channel_indices or ())
        )
        if len(self.s2_channel_indices) != len(self.s2_band_names):
            raise ValueError('s2_channel_indices and s2_band_names must have the same length.')
        if len(self.topographic_channel_indices) not in (0, 2):
            raise ValueError('topographic_channel_indices must be null or [slope_channel, dem_channel].')
        used_indices = self.s2_channel_indices + self.topographic_channel_indices
        if min(used_indices) < 0 or max(used_indices) >= self.in_channels:
            raise ValueError('All Galileo input channel indices must be valid for the input tensor.')

        backbone_key = backbone.lower()
        if backbone_key not in _GALILEO_VARIANTS:
            raise ValueError(f'Unsupported Galileo backbone "{backbone}". Available: {sorted(_GALILEO_VARIANTS)}.')
        self.encoder = galileo.Encoder(**_GALILEO_VARIANTS[backbone_key])
        if not checkpoint_path:
            raise ValueError('checkpoint_path is required for GalileoDPT.')
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Galileo checkpoint does not exist: {checkpoint}')
        try:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        state = {key.replace('.backbone', ''): value for key, value in state.items()}
        self.encoder.load_state_dict(state, strict=True)
        self.embed_dim = int(self.encoder.embedding_size)
        self.month = int(month)
        missing_s2_bands = set(galileo.S2_BANDS) - set(self.s2_band_names)
        if missing_s2_bands:
            raise ValueError(f's2_band_names is missing Galileo-required bands: {sorted(missing_s2_bands)}.')
        self._galileo_s2_indices = tuple(self.s2_band_names.index(band) for band in galileo.S2_BANDS)
        self._galileo_s2_destinations = tuple(galileo.SPACE_TIME_BANDS.index(band) for band in galileo.S2_BANDS)
        self._s2_group_indices = tuple(
            idx for idx, name in enumerate(galileo.SPACE_TIME_BANDS_GROUPS_IDX) if name.startswith('S2_')
        )
        self._space_time_bands = len(galileo.SPACE_TIME_BANDS)
        self._space_bands = len(galileo.SPACE_BANDS)
        self._time_bands = len(galileo.TIME_BANDS)
        self._static_bands = len(galileo.STATIC_BANDS)
        self._space_time_groups = len(galileo.SPACE_TIME_BANDS_GROUPS_IDX)
        self._space_groups = len(galileo.SPACE_BAND_GROUPS_IDX)
        self._srtm_group_index = tuple(galileo.SPACE_BAND_GROUPS_IDX).index('SRTM')
        self._elevation_index = galileo.SPACE_BANDS.index('elevation')
        self._slope_index = galileo.SPACE_BANDS.index('slope')
        self._time_groups = len(galileo.TIME_BAND_GROUPS_IDX)
        self._static_groups = len(galileo.STATIC_BAND_GROUPS_IDX)

        stage_channels = tuple(int(c) for c in (
            decoder_stage_channels
            if decoder_stage_channels is not None
            else _default_stage_channels(self.embed_dim)
        ))
        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels
        self.decoder = DPTDecoder(
            embed_dim=self.embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=False,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)

        if freeze_backbone:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        if x.shape[1] != self.in_channels:
            raise ValueError(f'GalileoDPT expects {self.in_channels} input channels, got {x.shape[1]}.')
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(
                f'Input size {tuple(x.shape[-2:])} must be divisible by Galileo patch_size={self.patch_size}.'
            )

        s2 = x[:, self.s2_channel_indices].permute(0, 2, 3, 1).contiguous()
        topography = (
            x[:, self.topographic_channel_indices].permute(0, 2, 3, 1).contiguous()
            if self.topographic_channel_indices else None
        )
        tokens = self._encode_s2(s2, topography)
        patch_shape: Tuple[int, int] = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        expected_tokens = patch_shape[0] * patch_shape[1]
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f'Galileo returned {tokens.shape[1]} tokens; expected {expected_tokens} for grid {patch_shape}.'
            )

        # Galileo's public wrapper exposes the final encoder representation.
        # DPT still needs four scales; its resize projections create them from
        # this shared pretrained representation.
        return self.decoder([(tokens, None)] * 4, patch_shape)

    def _encode_s2(self, s2: torch.Tensor, topography: Optional[torch.Tensor]) -> torch.Tensor:
        """Convert S2 plus [slope, DEM] inputs to Galileo's grouped input API."""
        batch, height, width, _ = s2.shape
        dtype, device = s2.dtype, s2.device
        s_t_x = torch.zeros(batch, height, width, 1, self._space_time_bands, dtype=dtype, device=device)
        s_t_x[..., 0, self._galileo_s2_destinations] = s2[..., self._galileo_s2_indices]
        s_t_m = torch.ones(batch, height, width, 1, self._space_time_groups, dtype=dtype, device=device)
        s_t_m[..., 0, self._s2_group_indices] = 0
        sp_x = torch.zeros(batch, height, width, self._space_bands, dtype=dtype, device=device)
        if topography is not None:
            sp_x[..., self._slope_index] = topography[..., 0]
            sp_x[..., self._elevation_index] = topography[..., 1]
        t_x = torch.zeros(batch, 1, self._time_bands, dtype=dtype, device=device)
        st_x = torch.zeros(batch, self._static_bands, dtype=dtype, device=device)
        sp_m = torch.ones(batch, height, width, self._space_groups, dtype=dtype, device=device)
        if topography is not None:
            sp_m[..., self._srtm_group_index] = 0
        t_m = torch.ones(batch, 1, self._time_groups, dtype=dtype, device=device)
        st_m = torch.ones(batch, self._static_groups, dtype=dtype, device=device)
        months = torch.full((batch, 1), self.month, dtype=torch.long, device=device)
        s_t_out, _, _, _, _, _, _, _, _ = self.encoder(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
            patch_size=self.patch_size, add_layernorm_on_exit=True,
        )
        return s_t_out[..., self._s2_group_indices, :].mean(dim=(3, 4)).reshape(batch, -1, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.classifier(self.head(self.forward_features(x)))
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits
