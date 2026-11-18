"""Registry for native 12-channel/1024 FloodSimBench seven baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch.nn as nn

from gphyt.model.floodsimbench_context12_models import MODEL_CLASSES
from gphyt.model.model_registry_seven import (
    BASELINE_NAMES,
    CHECKPOINT_SHA256,
    OFFICIAL_REPOSITORIES,
    PT_FT_NAMES,
)


RUNTIME_HEIGHT = 1024
RUNTIME_WIDTH = 1024
INPUT_CHANNELS = 12
OUTPUT_CHANNELS = 1


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_floodsimbench_baseline(
    name: str,
    *,
    official_root: str,
    checkpoint_path: str,
    report_dir: str,
    activation_checkpointing: bool = True,
) -> nn.Module:
    if name not in BASELINE_NAMES:
        raise KeyError(f"unknown seven-baseline model {name!r}")
    report_path = str(Path(report_dir) / name / "pretrained_loading_report.json")
    model_class = MODEL_CLASSES[name]
    if name == "ConvLSTM-Scratch":
        model = model_class()
    elif name == "EarthFormer-Scratch":
        model = model_class(
            RUNTIME_HEIGHT, RUNTIME_WIDTH, official_root=official_root
        )
    elif name == "Prithvi-EO2-300M-PT-FT":
        model = model_class(
            RUNTIME_HEIGHT,
            RUNTIME_WIDTH,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
        )
    else:
        model = model_class(
            RUNTIME_HEIGHT,
            RUNTIME_WIDTH,
            official_root=official_root,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
        )
    if activation_checkpointing:
        method = getattr(model, "enable_activation_checkpointing", None)
        if callable(method):
            method()
    return model


def model_metadata(model: nn.Module, name: str) -> Dict[str, Any]:
    repository, commit = OFFICIAL_REPOSITORIES[name]
    report = getattr(model, "loading_report", None)
    checkpointing = {
        "requested": True,
        "official_api_enabled": getattr(
            model, "activation_checkpointing_enabled", None
        ),
        "model_flag": getattr(model, "activation_checkpointing", None),
        "earthformer_checkpoint_level": getattr(
            model, "official_config", {}
        ).get("checkpoint_level"),
    }
    return {
        "experiment_name": name,
        "official_repository": repository,
        "official_commit": commit,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameters_total": parameter_count(model),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256.get(name),
        "backbone_loading_ratio": (
            report.backbone_loading_ratio if report is not None else None
        ),
        "input": "[B,12,1024,1024] context-6 physical channels",
        "channel_order": [
            "R[t-5]",
            "R[t-4]",
            "R[t-3]",
            "R[t-2]",
            "R[t-1]",
            "R[t]",
            "DEM",
            "h[t-5]",
            "h[t-4]",
            "h[t-3]",
            "h[t-2]",
            "h[t-1]",
        ],
        "output": "[B,1,1024,1024] direct normalized h[t]",
        "spatial_resize": False,
        "spatial_subtiling": False,
        "activation_checkpointing": checkpointing,
    }


__all__ = [
    "BASELINE_NAMES",
    "PT_FT_NAMES",
    "build_floodsimbench_baseline",
    "model_metadata",
    "parameter_count",
]
