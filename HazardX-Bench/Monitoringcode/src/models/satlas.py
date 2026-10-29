"""SatlasPretrain semantic-segmentation model."""

import torch
from torch import nn
from torch.nn import functional as F


class _FPNHead(nn.Module):
    """Minimal trainable decoder for multi-scale Swin features."""

    def __init__(self, channels: list[int], classes: int) -> None:
        super().__init__()
        self.project = nn.ModuleList(nn.Conv2d(channel, 128, 1) for channel in channels)
        self.head = nn.Sequential(
            nn.Conv2d(128 * len(channels), 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, classes, 1),
        )

    def forward(self, features: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = [
            F.interpolate(layer(feature), target, mode='bilinear', align_corners=False)
            for layer, feature in zip(self.project, features)
        ]
        logits = self.head(torch.cat(projected, dim=1))
        return F.interpolate(logits, size, mode='bilinear', align_corners=False)


class SatlasSeg(nn.Module):
    """Satlas Sentinel-2 single-image MS Swin encoder and FPN decoder."""

    def __init__(
        self,
        classes: int = 2,
        variant: str = 'tiny',
        pretrained: bool = True,
        pretrained_checkpoint_path: str | None = None,
        freeze_encoder: bool = False,
        in_channels: int = 9,
        projection_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        from torchgeo.models import Swin_V2_B_Weights, Swin_V2_T_Weights, swin_v2_b, swin_v2_t

        if variant == 'tiny':
            self.encoder = swin_v2_t(
                Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS if pretrained and not pretrained_checkpoint_path else None
            )
            channels = [96, 192, 384, 768]
        elif variant == 'base':
            self.encoder = swin_v2_b(
                Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS if pretrained and not pretrained_checkpoint_path else None
            )
            channels = [128, 256, 512, 1024]
        else:
            raise ValueError("variant must be 'tiny' or 'base'.")
        # Factories only replace this layer when a weights enum is supplied.
        # Keep the architecture 9-channel also for testing/resuming without a
        # second download of the pretrained backbone.
        patch_embed = self.encoder.features[0][0]
        if patch_embed.in_channels != 9:
            self.encoder.features[0][0] = nn.Conv2d(
                9, patch_embed.out_channels, kernel_size=4, stride=4
            )
        if pretrained_checkpoint_path:
            # TorchGeo's Satlas checkpoint contains a 9-channel patch embedding.
            checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
            state = checkpoint.get('model', checkpoint)
            self.encoder.load_state_dict(state, strict=False)
        self.input_projection = nn.Conv2d(in_channels, 9, 1, bias=False)
        with torch.no_grad():
            self.input_projection.weight.zero_()
            indices = projection_indices or list(range(9))
            if len(indices) != 9 or min(indices) < 0 or max(indices) >= in_channels:
                raise ValueError('projection_indices must contain 9 valid source indices.')
            for output_channel, input_channel in enumerate(indices):
                self.input_projection.weight[output_channel, input_channel, 0, 0] = 1.0
        self.decoder = _FPNHead(channels, classes)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size, outputs = x.shape[-2:], []
        x = self.input_projection(x)
        for index, layer in enumerate(self.encoder.features):
            x = layer(x)
            if index in (1, 3, 5, 7):
                outputs.append(x.permute(0, 3, 1, 2).contiguous())
        return self.decoder(outputs, size)
