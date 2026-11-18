"""Registry for the seven requested FloodCastBench baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch.nn as nn

from gphyt.model.convlstm_openstl import ConvLSTMScratch
from gphyt.model.dpot_s import DPOTSPTFT
from gphyt.model.earthformer_official import EarthFormerScratch
from gphyt.model.gphyt_s import GPhyTSPTFT
from gphyt.model.pde_transformer_sc_s import PDETransformerSCSPTFT
from gphyt.model.poseidon_t import PoseidonTPTFT
from gphyt.model.prithvi_eo2_300m import PrithviEO2300MPTFT


BASELINE_NAMES = (
    "ConvLSTM-Scratch",
    "EarthFormer-Scratch",
    "PDE-T-SC-S-PT-FT",
    "DPOT-S-PT-FT",
    "Poseidon-T-PT-FT",
    "GPhyT-S-PT-FT",
    "Prithvi-EO2-300M-PT-FT",
)

PT_FT_NAMES = BASELINE_NAMES[2:]

OFFICIAL_REPOSITORIES = {
    "ConvLSTM-Scratch": (
        "https://github.com/chengtan9907/OpenSTL",
        "eecf8a3078f0a178dbc7b28723da20f94ce36985",
    ),
    "EarthFormer-Scratch": (
        "https://github.com/amazon-science/earth-forecasting-transformer",
        "7732b03bdb366110563516c3502315deab4c2026",
    ),
    "PDE-T-SC-S-PT-FT": (
        "https://github.com/tum-pbs/pde-transformer",
        "850e09d352487922e4a853ee90a2624717769cc7",
    ),
    "DPOT-S-PT-FT": (
        "https://github.com/thu-ml/DPOT",
        "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17",
    ),
    "Poseidon-T-PT-FT": (
        "https://github.com/camlab-ethz/poseidon",
        "b8fa28f59bd7f7673323f28d11a12c6f3a215c61",
    ),
    "GPhyT-S-PT-FT": (
        "https://github.com/FloWsnr/General-Physics-Transformer",
        "df1527c4773ababc6f8af0d117fb5803fa3bf74b",
    ),
    "Prithvi-EO2-300M-PT-FT": (
        "https://github.com/NASA-IMPACT/Prithvi-EO-2.0",
        "758311280f935d4775c05b26d792768bf1fda841",
    ),
}

CHECKPOINT_SHA256 = {
    "PDE-T-SC-S-PT-FT": "90555b8277687939571163e6fcd9725039d994a95c1dc3ad367849c995cebcde",
    "DPOT-S-PT-FT": "26078da23181a6e8c4c87a0fae71b1bd91ee02b9bef748e1b29569ed052770f4",
    "Poseidon-T-PT-FT": "a363f7317fbc3a900a318fc63cc53197705d95fce0e0ce28dd3c8844a89112e2",
    "GPhyT-S-PT-FT": "a7beab39e6e9527d44a7a43a7f2e95a5a22aaa69f5932cc9809f9dabb0b2e6c0",
    "Prithvi-EO2-300M-PT-FT": "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7",
}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_flood_baseline(
    name: str,
    *,
    runtime_height: int,
    runtime_width: int,
    official_roots: Mapping[str, str],
    checkpoint_paths: Mapping[str, str],
    report_dir: str,
) -> nn.Module:
    if name not in BASELINE_NAMES:
        raise KeyError(f"unknown seven-baseline model {name!r}")
    report_path = str(
        Path(report_dir) / name / "pretrained_loading_report.json"
    )
    if name == "ConvLSTM-Scratch":
        return ConvLSTMScratch()
    if name == "EarthFormer-Scratch":
        return EarthFormerScratch(
            runtime_height,
            runtime_width,
            official_root=official_roots.get(name, ""),
        )
    if name == "PDE-T-SC-S-PT-FT":
        return PDETransformerSCSPTFT(
            runtime_height,
            runtime_width,
            official_root=official_roots.get(name, ""),
            checkpoint_path=checkpoint_paths.get(name, ""),
            report_path=report_path,
        )
    if name == "DPOT-S-PT-FT":
        return DPOTSPTFT(
            runtime_height,
            runtime_width,
            official_root=official_roots.get(name, ""),
            checkpoint_path=checkpoint_paths.get(name, ""),
            report_path=report_path,
        )
    if name == "Poseidon-T-PT-FT":
        return PoseidonTPTFT(
            runtime_height,
            runtime_width,
            official_root=official_roots.get(name, ""),
            checkpoint_path=checkpoint_paths.get(name, ""),
            report_path=report_path,
        )
    if name == "GPhyT-S-PT-FT":
        return GPhyTSPTFT(
            runtime_height,
            runtime_width,
            official_root=official_roots.get(name, ""),
            checkpoint_path=checkpoint_paths.get(name, ""),
            report_path=report_path,
        )
    if name == "Prithvi-EO2-300M-PT-FT":
        return PrithviEO2300MPTFT(
            runtime_height,
            runtime_width,
            checkpoint_path=checkpoint_paths.get(name, ""),
            report_path=report_path,
        )
    raise AssertionError(name)


def model_metadata(model: nn.Module, name: str) -> Dict[str, Any]:
    repository, commit = OFFICIAL_REPOSITORIES[name]
    report = getattr(model, "loading_report", None)
    metadata = {
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
        "input": "[B,3,H,W] = [h_t, rainfall_t, DEM]",
        "output": "[B,1,H,W] = h_{t+5min}",
        "spatial_resize": False,
    }
    inactive_names = tuple(
        getattr(model, "structurally_inactive_parameter_names", ())
    )
    if inactive_names:
        metadata["structurally_inactive_parameter_names"] = list(inactive_names)
        metadata["structurally_inactive_parameter_count"] = int(
            getattr(model, "structurally_inactive_parameter_count")
        )
        metadata["structurally_inactive_reason"] = (
            "official terminal encoder global-vector update has no decoder consumer"
        )
    return metadata
