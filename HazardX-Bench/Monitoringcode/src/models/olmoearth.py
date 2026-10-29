"""OLMO-Earth semantic-segmentation model."""

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    """Return a GroupNorm group count valid for any configured width."""
    # Keep at least four channels per group so the 1x1 PPM branch also works
    # with batch size one during validation or inference.
    for groups in range(min(32, max(1, channels // 4)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class _PyramidPooling(nn.Module):
    """PSP context branch used at the deepest OLMO token scale."""

    def __init__(self, channels: int, pool_scales: tuple[int, ...]) -> None:
        super().__init__()
        branch_channels = max(channels // len(pool_scales), 16)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.GroupNorm(_groups(branch_channels), branch_channels),
                nn.GELU(),
            )
            for scale in pool_scales
        )
        self.bottleneck = _ConvNormAct(channels + len(pool_scales) * branch_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        pooled = [x]
        for branch in self.branches:
            pooled.append(F.interpolate(branch(x), spatial_size, mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(pooled, dim=1))


class OlmoEarthUPerDecoder(nn.Module):
    """UPerNet-style decoder built from the public final OLMO token grid.

    OLMO-Earth's minimal package exposes final tokens, rather than four encoder
    stages. Strided token-grid branches form a learned feature pyramid; PPM and
    top-down FPN fusion then supply local and global segmentation context.
    """

    def __init__(
        self, embedding_dim: int, classes: int, channels: int = 256,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.stage1 = _ConvNormAct(embedding_dim, channels)
        self.stage2 = _ConvNormAct(channels, channels, stride=2)
        self.stage3 = _ConvNormAct(channels, channels, stride=2)
        self.stage4 = _ConvNormAct(channels, channels, stride=2)
        self.ppm = _PyramidPooling(channels, pool_scales)
        self.laterals = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(3))
        self.fpn_convs = nn.ModuleList(_ConvNormAct(channels, channels) for _ in range(3))
        self.fusion = _ConvNormAct(channels * 4, channels)
        self.classifier = nn.Conv2d(channels, classes, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        stages = [self.stage1(tokens)]
        stages.append(self.stage2(stages[-1]))
        stages.append(self.stage3(stages[-1]))
        stages.append(self.stage4(stages[-1]))
        current = self.ppm(stages[-1])
        outputs = [current]
        for lateral, fpn_conv, stage in zip(reversed(self.laterals), reversed(self.fpn_convs), reversed(stages[:-1])):
            current = fpn_conv(lateral(stage) + F.interpolate(current, stage.shape[-2:], mode='bilinear', align_corners=False))
            outputs.append(current)
        outputs.reverse()
        target_size = outputs[0].shape[-2:]
        fused = torch.cat([F.interpolate(feature, target_size, mode='bilinear', align_corners=False) for feature in outputs], dim=1)
        return self.classifier(self.fusion(fused))


class OlmoEarthSeg(nn.Module):
    """OLMO-Earth v1 encoder with a UPerNet-style segmentation decoder.

    Input has 12 L2A bands in OLMO order plus one SRTM/DEM channel. A source
    slope band is intentionally excluded because OLMO-Earth has no matching
    pretraining modality.
    """

    def __init__(
        self,
        classes: int = 2,
        size: str = 'nano',
        patch_size: int = 4,
        timestamp: tuple[int, int, int] = (15, 5, 2020),
        pretrained: bool = True,
        pretrained_checkpoint_path: str | None = None,
        freeze_encoder: bool = False,
        decoder_channels: int = 256,
        decoder_pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        in_channels: int = 13,
        projection_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        from torchgeo.models import OlmoEarthV1_Weights, olmoearth_v1

        if size not in {'nano', 'tiny', 'base', 'large'}:
            raise ValueError("size must be 'nano', 'tiny', 'base', or 'large'.")
        weights = getattr(OlmoEarthV1_Weights, size.upper()) if pretrained and not pretrained_checkpoint_path else None
        self.encoder = olmoearth_v1(weights=weights, max_patch_size=patch_size)
        if pretrained_checkpoint_path:
            checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
            state = checkpoint.get('model', checkpoint)
            self.encoder.load_state_dict(state, strict=False)
        self.input_projection = nn.Conv2d(in_channels, 13, 1, bias=False)
        with torch.no_grad():
            self.input_projection.weight.zero_()
            indices = projection_indices or list(range(13))
            if len(indices) != 13 or min(indices) < -1 or max(indices) >= in_channels:
                raise ValueError(
                    'projection_indices must contain 13 source indices; use -1 for a missing modality.'
                )
            for output_channel, input_channel in enumerate(indices):
                if input_channel >= 0:
                    self.input_projection.weight[output_channel, input_channel, 0, 0] = 1.0
        self.patch_size = patch_size
        self.timestamp = timestamp
        dim = {'nano': 128, 'tiny': 192, 'base': 768, 'large': 1024}[size]
        self.decoder = OlmoEarthUPerDecoder(
            dim, classes, channels=decoder_channels, pool_scales=decoder_pool_scales
        )
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

        x = self.input_projection(x)
        batch, _, height, width = x.shape
        s2 = x[:, :12].permute(0, 2, 3, 1).unsqueeze(3)
        dem = x[:, 12:].permute(0, 2, 3, 1)
        sample = MaskedOlmoEarthSample(
            timestamps=torch.tensor(self.timestamp, device=x.device).view(1, 1, 3).repeat(batch, 1, 1),
            sentinel2_l2a=s2,
            sentinel2_l2a_mask=torch.zeros(batch, height, width, 1, 3, dtype=torch.long, device=x.device),
            srtm=dem,
            srtm_mask=torch.zeros(batch, height, width, 1, dtype=torch.long, device=x.device),
        )
        tokens = self.encoder.encoder(sample, patch_size=self.patch_size)['tokens_and_masks'].sentinel2_l2a
        features = tokens.mean(dim=(3, 4)).permute(0, 3, 1, 2).contiguous()
        logits = self.decoder(features)
        return F.interpolate(logits, (height, width), mode='bilinear', align_corners=False)
