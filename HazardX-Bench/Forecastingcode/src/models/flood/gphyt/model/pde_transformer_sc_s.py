"""Official separate-channel PDE-Transformer SC-S FloodCastBench wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from gphyt.model.checkpoint_loading import (
    PretrainedLoadingReport,
    assert_full_finetuning,
    load_pretrained_with_report,
)
from gphyt.model.official_import import import_official_module, verify_source_file
from gphyt.model.safetensors_io import load_safetensors_state
from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


REPOSITORY = "https://github.com/tum-pbs/pde-transformer"
COMMIT = "850e09d352487922e4a853ee90a2624717769cc7"
SOURCE_RELATIVE_PATH = "pdetransformer/core/separate_channels/pde_transformer.py"
SOURCE_SHA256 = "cdc7a89765ce586d09edc1cec0d482f4bbc18d0738978ded2b7fe61516dc921a"
CHECKPOINT_REPOSITORY = "thuerey-group/pde-transformer"
CHECKPOINT_SUBFOLDER = "sc-s"
CHECKPOINT_SHA256 = "90555b8277687939571163e6fcd9725039d994a95c1dc3ad367849c995cebcde"



class PDETransformerSCSPTFT(nn.Module):
    experiment_name = "PDE-T-SC-S-PT-FT"
    source_repository = REPOSITORY
    source_commit = COMMIT
    checkpoint_repository = CHECKPOINT_REPOSITORY
    checkpoint_sha256 = CHECKPOINT_SHA256
    # patch_size=4 and two factor-2 encoder downsamples; window padding is
    # performed inside the official PDEStage implementation.
    spatial_multiple = 16

    def __init__(
        self,
        runtime_height: int,
        runtime_width: int,
        *,
        official_root: str,
        checkpoint_path: str,
        report_path: str,
        channel_type_ids: Tuple[int, int, int] = (0, 1, 2),
    ) -> None:
        super().__init__()
        if not official_root:
            raise ValueError("PDE-T SC-S requires an exact official source checkout")
        verify_source_file(
            official_root,
            SOURCE_RELATIVE_PATH,
            SOURCE_SHA256,
            repository=REPOSITORY,
            commit=COMMIT,
        )
        module = import_official_module(
            "pdetransformer.core.separate_channels.pde_transformer",
            root=official_root,
            import_subdirs=("",),
            install_hint=(
                f"install the dependencies of pdetransformer@{COMMIT}; its "
                "published package requires Python >=3.11"
            ),
        )
        self.official_config = {
            "sample_size": max(int(runtime_height), int(runtime_width)),
            "num_timesteps": 2,
            "type": "PDE-S",
            "patch_size": 4,
            "periodic": False,
            "carrier_token_active": False,
        }
        self.backbone = module.PDETransformer(**self.official_config)
        self.channel_type_ids = tuple(int(v) for v in channel_type_ids)
        if len(self.channel_type_ids) != 3:
            raise ValueError("PDE-T requires exactly three channel type ids")

        state, digest = load_safetensors_state(checkpoint_path)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"PDE-T checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}"
            )
        self.loading_report: PretrainedLoadingReport = load_pretrained_with_report(
            self.backbone,
            state,
            checkpoint_sha256=digest,
            report_path=report_path,
            converters={},
            newly_initialized_prefixes=(),
            backbone_excluded_prefixes=(),
            minimum_backbone_ratio=0.90,
        )
        assert_full_finetuning(self)

    @staticmethod
    def _per_channel_metadata(
        batch: int,
        channel_ids: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ):
        zeros = [torch.zeros(batch, device=device, dtype=dtype) for _ in range(3)]
        channel_type = [
            torch.full((batch,), value, device=device, dtype=torch.long)
            for value in channel_ids
        ]
        labels = [torch.zeros(batch, device=device, dtype=torch.long) for _ in range(3)]
        parameters = [torch.zeros(batch, 1, device=device, dtype=dtype) for _ in range(3)]
        parameter_classes = [torch.zeros(batch, 1, device=device, dtype=torch.long) for _ in range(3)]
        task = [torch.zeros(batch, device=device, dtype=torch.long) for _ in range(3)]
        return zeros, channel_type, labels, parameters, parameter_classes, task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, self.spatial_multiple)
        fields = [torch.cat((padded[:, index : index + 1], torch.zeros_like(padded[:, index : index + 1])), dim=1) for index in range(3)]
        zeros, channel_type, labels, parameters, parameter_classes, task = self._per_channel_metadata(
            batch, self.channel_type_ids, padded.device, padded.dtype
        )
        result = self.backbone(
            x=fields,
            simulation_time=zeros,
            channel_type=channel_type,
            pde_type=labels,
            pde_parameters=parameters,
            pde_parameters_class=parameter_classes,
            simulation_dt=[torch.ones_like(value) for value in zeros],
            task=task,
            t=zeros,
            return_dict=True,
        )
        outputs = result.sample
        if not isinstance(outputs, list) or len(outputs) != 3:
            raise AssertionError("SC-S must return one tensor per physical channel")
        prediction = outputs[0]
        if prediction.ndim != 4 or prediction.shape[1] != 2:
            raise AssertionError(f"unexpected SC-S field shape {tuple(prediction.shape)}")
        prediction = prediction[:, 1:2]
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x)
