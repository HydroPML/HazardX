"""Official DPOT-S pretrained/fine-tuned FloodCastBench wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    load_pretrained_with_report,
    load_torch_checkpoint_state,
)
from gphyt.model.official_import import sha256_file, verify_source_file
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/thu-ml/DPOT"
COMMIT = "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17"
SOURCE_RELATIVE_PATH = "models/dpot.py"
SOURCE_SHA256 = "d80f208fadc33d43ba2edf5995ac28c830997ad4b73194eed91184d7cf756163"
CHECKPOINT_REPOSITORY = "hzk17/DPOT"
CHECKPOINT_FILENAME = "model_S.pth"
CHECKPOINT_SHA256 = "26078da23181a6e8c4c87a0fae71b1bd91ee02b9bef748e1b29569ed052770f4"


def _load_dpot_module(source_file: Path):
    spec = importlib.util.spec_from_file_location("flood_official_dpot_s", source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _convert_patch_input(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # Official forward concatenates physical variables before [x,y,t] grid.
    # The S checkpoint has 4 physical + 3 coordinate channels; FloodCastBench
    # has 3 physical + the unchanged 3 coordinate channels.
    if key != "patch_embed.proj.0.weight":
        raise RuntimeError(f"unexpected DPOT input converter key {key}")
    if source.ndim != 4 or target.ndim != 4:
        raise RuntimeError("DPOT patch input must be a Conv2d kernel")
    if source.shape[1] != 7 or target.shape[1] != 6:
        raise RuntimeError(
            f"unconfirmed DPOT channel layout: source={tuple(source.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if tuple(source.shape[0:1] + source.shape[2:]) != tuple(target.shape[0:1] + target.shape[2:]):
        raise RuntimeError("DPOT patch kernel differs outside its input-channel axis")
    physical_mean = source[:, :4].mean(dim=1, keepdim=True)
    physical = physical_mean.repeat(1, 3, 1, 1)
    coordinates = source[:, 4:7]
    return torch.cat((physical, coordinates), dim=1)


def _convert_time_aggregator(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if key != "time_agg_layer.w" or source.ndim != 3 or target.ndim != 3:
        raise RuntimeError("unconfirmed DPOT temporal adapter tensor")
    if target.shape[0] != 1 or tuple(source.shape[1:]) != tuple(target.shape[1:]):
        raise RuntimeError(
            f"unconfirmed DPOT time weight axes: {tuple(source.shape)} -> {tuple(target.shape)}"
        )
    return source.mean(dim=0, keepdim=True)


class DPOTSPTFT(nn.Module):
    experiment_name = "DPOT-S-PT-FT"
    source_repository = REPOSITORY
    source_commit = COMMIT
    checkpoint_repository = CHECKPOINT_REPOSITORY
    checkpoint_sha256 = CHECKPOINT_SHA256
    patch_size = 8

    def __init__(
        self,
        runtime_height: int,
        runtime_width: int,
        *,
        official_root: str,
        checkpoint_path: str,
        report_path: str,
    ) -> None:
        super().__init__()
        source_file = verify_source_file(
            official_root,
            SOURCE_RELATIVE_PATH,
            SOURCE_SHA256,
            repository=REPOSITORY,
            commit=COMMIT,
        )
        module = _load_dpot_module(source_file)
        # Construct at the checkpoint's 128x128 pretraining grid so its learned
        # position table remains a directly loadable parameter.  Runtime
        # rectangular grids use bicubic interpolation in the wrapper.
        self.official_config = {
            "img_size": 128,
            "patch_size": 8,
            "mixing_type": "afno",
            "in_channels": 3,
            "out_channels": 4,
            "in_timesteps": 1,
            "out_timesteps": 1,
            "n_blocks": 8,
            "embed_dim": 1024,
            "out_layer_dim": 32,
            "depth": 6,
            "modes": 32,
            "mlp_ratio": 1.0,
            "normalize": False,
        }
        self.backbone = module.DPOTNet(**self.official_config)
        # DPOT's cls_head predicts the source pretraining-dataset identity. It
        # has no target or loss in FloodCastBench, so retaining it as a
        # trainable module would leave disconnected parameters during full
        # fine-tuning. Remove the task-specific head before audited loading;
        # its checkpoint tensors are consequently reported as unexpected,
        # while every AFNO/backbone parameter remains trainable and connected
        # to the flood-regression loss.
        classification_head = self.backbone.cls_head
        if (
            not isinstance(classification_head, nn.Sequential)
            or len(classification_head) != 5
        ):
            raise RuntimeError("official DPOT classification-head layout changed")
        final_classifier = classification_head[-1]
        if (
            not isinstance(final_classifier, nn.Linear)
            or final_classifier.out_features != 12
        ):
            raise RuntimeError("official DPOT dataset-classification output changed")
        self.backbone.cls_head = nn.Identity()
        self.discarded_pretrained_components = ("cls_head",)

        old_last = self.backbone.out_layer[-1]
        if not isinstance(old_last, nn.Conv2d) or old_last.out_channels != 4:
            raise RuntimeError("official DPOT output recovery layout changed")
        self.backbone.out_layer[-1] = nn.Conv2d(
            old_last.in_channels, 1, kernel_size=1, stride=1
        )
        self.backbone.out_channels = 1
        self.backbone.out_timesteps = 1

        digest = sha256_file(checkpoint_path)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"DPOT-S checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}"
            )
        state, loaded_digest = load_torch_checkpoint_state(checkpoint_path)
        if loaded_digest != digest:
            raise AssertionError("DPOT checkpoint changed while loading")
        converters = {
            "patch_embed.proj.0.weight": _convert_patch_input,
            "time_agg_layer.w": _convert_time_aggregator,
        }
        self.loading_report: PretrainedLoadingReport = load_pretrained_with_report(
            self.backbone,
            state,
            checkpoint_sha256=digest,
            report_path=report_path,
            converters=converters,
            newly_initialized_prefixes=("out_layer.4",),
            backbone_excluded_prefixes=("out_layer",),
            minimum_backbone_ratio=0.90,
        )
        discarded_cls_keys = tuple(
            sorted(key for key in state if key.startswith("cls_head."))
        )
        if not discarded_cls_keys:
            raise RuntimeError("DPOT checkpoint contains no official cls_head tensors")
        if not set(discarded_cls_keys).issubset(self.loading_report.unexpected):
            raise AssertionError(
                "discarded DPOT cls_head tensors were not fully recorded as unexpected"
            )
        self.discarded_pretrained_keys = discarded_cls_keys
        assert_full_finetuning(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, self.patch_size)
        # Official DPOT public layout is [B,X,Y,T,C].
        official_input = padded.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        backbone = self.backbone
        batch, height, width, timesteps, _ = official_input.shape
        grid = backbone.get_grid_3d(official_input)
        hidden = torch.cat((official_input, grid), dim=-1).contiguous()
        hidden = rearrange(hidden, "b x y t c -> (b t) c x y")
        # Call the official projection directly: PatchEmbed.forward's square
        # assertion is metadata-only and would reject native rectangular grids.
        hidden = backbone.patch_embed.proj(hidden)
        position = F.interpolate(
            backbone.pos_embed,
            size=hidden.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        hidden = hidden + position
        hidden = rearrange(hidden, "(b t) c x y -> b x y t c", b=batch, t=timesteps)
        hidden = backbone.time_agg_layer(hidden)
        hidden = rearrange(hidden, "b x y c -> b c x y")
        for block in backbone.blocks:
            hidden = block(hidden)
        prediction = backbone.out_layer(hidden).permute(0, 2, 3, 1)
        prediction = prediction.reshape(batch, height, width, 1, 1)
        prediction = prediction[..., 0, 0].unsqueeze(1)
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
