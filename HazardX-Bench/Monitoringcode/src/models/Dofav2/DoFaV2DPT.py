from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels

from .dofa_v2 import vit_base_patch14, vit_large_patch14


_DOFA_CHECKPOINTS = {
    "dofa_vit_base": "dofav2_vit_base_e150.pth",
    "dofa_vit_large": "dofav2_vit_large_e150.pth",
    "base": "dofav2_vit_base_e150.pth",
    "large": "dofav2_vit_large_e150.pth",
}


def _to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a tuple of length 2, got {value!r}.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _resolve_wave_list(
    wave_list: Optional[Sequence[float]],
    band_names: Optional[Sequence[str]],
    in_channels: int,
) -> list[float]:
    if wave_list is not None:
        resolved = [float(x) for x in wave_list]
    elif band_names is not None:
        wave_map = {
            "COASTAL_AEROSOL": 0.44,
            "BLUE": 0.49,
            "GREEN": 0.56,
            "RED": 0.665,
            "RED_EDGE_1": 0.705,
            "RED_EDGE_2": 0.74,
            "RED_EDGE_3": 0.783,
            "NIR_BROAD": 0.832,
            "NIR_NARROW": 0.864,
            "WATER_VAPOR": 0.945,
            "CIRRUS": 1.373,
            "SWIR_1": 1.61,
            "SWIR_2": 2.20,
            "THERMAL_INFRARED_1": 10.90,
            "THERMAL_INFRARED_12": 12.00,
            "VV": 5.405,
            "VH": 5.405,
            "ASC_VV": 5.405,
            "ASC_VH": 5.405,
            "DSC_VV": 5.405,
            "DSC_VH": 5.405,
            "VV-VH": 5.405,
        }
        resolved = []
        for band in band_names:
            key = str(band).split(".")[-1]
            if key not in wave_map:
                raise ValueError(f'Unknown DOFA band name "{band}".')
            resolved.append(float(wave_map[key]))
    elif in_channels == 3:
        resolved = [0.665, 0.56, 0.49]
    else:
        raise ValueError("DOFA v2 requires `wave_list`, `band_names`, or a 3-channel RGB input.")

    if len(resolved) != in_channels:
        raise ValueError(f"Expected {in_channels} wavelengths, got {len(resolved)}.")
    return resolved


class DOFAV2FeatureExtractor(nn.Module):
    def __init__(
        self,
        variant: str = "base",
        img_size: int | Tuple[int, int] = 224,
        wave_list: Optional[Sequence[float]] = None,
        band_names: Optional[Sequence[str]] = None,
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        freeze_backbone: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        variant_key = variant.lower()
        if variant_key in {"base", "dofa_vit_base"}:
            backbone = vit_base_patch14
        elif variant_key in {"large", "dofa_vit_large"}:
            backbone = vit_large_patch14
        else:
            raise ValueError(f'Unsupported DOFA v2 variant "{variant}".')

        self.model = backbone(img_size=img_size, **backbone_kwargs)
        self.img_size = self.model.img_size
        self.patch_size = _to_2tuple(self.model.patch_embed.patch_size)
        self.embed_dim = int(self.model.model.embed_dim)
        self.in_channels = int(in_channels)
        self.wave_list = _resolve_wave_list(wave_list, band_names, self.in_channels)
        self.out_indices = tuple(int(idx) for idx in self.model.out_indices)

        if checkpoint_path is None:
            checkpoint_name = _DOFA_CHECKPOINTS.get(variant_key)
            if checkpoint_name is not None:
                candidate = Path("/root/workcsc/DOFA") / checkpoint_name
                if candidate.exists():
                    checkpoint_path = str(candidate)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("model", "state_dict", "encoder"):
                    if key in state:
                        state = state[key]
                        break
            load_result = self.model.load_state_dict(state, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing or unexpected:
                print(f"[DOFA v2] Missing keys: {missing}")
                print(f"[DOFA v2] Unexpected keys: {unexpected}")

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[list[torch.Tensor], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError("Input tensor must have shape (B, C, H, W).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}.")

        outputs = self.model(x, self.wave_list)
        if not isinstance(outputs, list) or not outputs:
            raise RuntimeError("DOFA v2 backbone did not return feature maps.")

        patch_h = max(1, int(round(x.shape[-2] / float(self.patch_size[0]))))
        patch_w = max(1, int(round(x.shape[-1] / float(self.patch_size[1]))))
        return outputs, (patch_h, patch_w)


class DoFaV2DPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        variant: str = "base",
        img_size: int | Tuple[int, int] = 224,
        wave_list: Optional[Sequence[float]] = None,
        band_names: Optional[Sequence[str]] = None,
        checkpoint_path: Optional[str] = None,
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        self.encoder = DOFAV2FeatureExtractor(
            variant=variant,
            img_size=img_size,
            wave_list=wave_list,
            band_names=band_names,
            checkpoint_path=checkpoint_path,
            in_channels=in_channels,
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

        self.num_features = len(self.encoder.out_indices)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, patch_shape = self.encoder(x)
        return self.decoder(features, patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        decoded = self.forward_features(x)
        refined = self.head(decoded)
        logits = self.classifier(refined)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)
        return logits
