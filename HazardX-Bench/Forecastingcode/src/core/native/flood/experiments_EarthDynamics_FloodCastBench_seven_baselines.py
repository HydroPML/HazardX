"""Unified FloodCastBench entry point for seven requested Markov baselines.

This file installs model/optimizer/scheduler hooks into the same validated
dataset, normalization, masked-Huber, validation, checkpoint and strict
autoregressive rollout lifecycle used by
``experiments_EarthDynamics_FloodCastBench_four_baselines.py``.  It does not
modify that reference entry point or any existing model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
import floodcast_four_baselines as four
import floodcast_transolver_centered_dem_common as centered
from gphyt.model.model_registry_seven import (
    BASELINE_NAMES,
    PT_FT_NAMES,
    build_flood_baseline,
    model_metadata,
)


EXPERIMENT_NAME = "FloodCastBench_Seven_Official_Markov_Baselines"
DEFAULT_OUTPUT_DIR = (
    "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
    "FloodCastBench_seven_official_baselines"
)
LOSS_HUBER = centered.LOSS_HUBER

_BASE_TRAIN = baseline.train
_BASE_SMOKE = baseline.run_smoke_tests
_REAL_ADAM = torch.optim.Adam
_ACTIVE_MODEL_NAME = BASELINE_NAMES[0]
_LAST_BUILT_MODEL: Optional[nn.Module] = None


OPTIMIZATION = {
    "ConvLSTM-Scratch": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "onecycle",
        "grad_clip": 0.0,
    },
    "EarthFormer-Scratch": {
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "warmup_cosine_20",
        "grad_clip": 1.0,
    },
    "PDE-T-SC-S-PT-FT": {
        "optimizer": "AdamW",
        "lr": 4e-5,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "none",
        "grad_clip": 0.0,
    },
    "DPOT-S-PT-FT": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.9),
        "scheduler": "dpot_cycle",
        "grad_clip": 0.0,
    },
    "Poseidon-T-PT-FT": {
        "optimizer": "AdamW",
        "lr": 5e-5,
        "adapter_lr": 5e-4,
        "weight_decay": 1e-6,
        "betas": (0.9, 0.999),
        "scheduler": "cosine",
        "grad_clip": 5.0,
    },
    "GPhyT-S-PT-FT": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "scheduler": "gphyt_warmup_cosine",
        "grad_clip": 5.0,
    },
    "Prithvi-EO2-300M-PT-FT": {
        "optimizer": "AdamW",
        "lr": 5e-5,
        "weight_decay": 0.05,
        "betas": (0.9, 0.999),
        "scheduler": "cosine",
        "grad_clip": 1.0,
    },
}


class TrainerDeltaAdapter(nn.Module):
    """Map the public h_next API to the reference trainer's dh contract."""

    direct_derivative_output = True
    core_prediction = "h_next"
    trainer_adapter = "h_next_minus_h_t"

    def __init__(self, core: nn.Module, model_name: str) -> None:
        super().__init__()
        self.core = core
        self.model_name = model_name

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        height, width = four._validate_common_inputs(x, static, metadata)
        fields = torch.cat((x, static), dim=1)
        predicted_next = self.core(fields)
        expected = (x.shape[0], 1, height, width)
        if tuple(predicted_next.shape) != expected:
            raise AssertionError(
                f"{self.model_name} h_next {tuple(predicted_next.shape)} != {expected}"
            )
        return four._direct_next_to_trainer_delta(predicted_next, x)


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(dest)


def build_parser() -> argparse.ArgumentParser:
    parser = four.build_parser()
    parser.description = (
        "Seven official FloodCastBench one-step Markov baselines using the "
        "reference data/train/validation/strict-rollout lifecycle"
    )
    model_action = _action(parser, "model_type")
    model_action.choices = BASELINE_NAMES
    model_action.default = "ConvLSTM-Scratch"
    _action(parser, "output_method").default = four.OUTPUT_DIRECT
    parser.add_argument(
        "--model-runtime-height",
        type=int,
        default=810,
        help="native H used to construct shape-bound official backbones",
    )
    parser.add_argument(
        "--model-runtime-width",
        type=int,
        default=441,
        help="native W used to construct shape-bound official backbones",
    )
    parser.add_argument(
        "--official-root",
        default="",
        help="checkout of the selected model's exact official commit",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default="",
        help="local official checkpoint; weights are never copied to results",
    )
    parser.add_argument(
        "--pretrained-report-dir",
        default="",
        help="defaults to <output-dir>/pretrained_reports",
    )
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR,
        loss=LOSS_HUBER,
        output_method=four.OUTPUT_DIRECT,
        normalization="train_standard",
    )
    return parser


def _resolve_optimization(args: argparse.Namespace) -> None:
    config = OPTIMIZATION[args.model_type]
    args.optimizer_name = config["optimizer"]
    args.learning_rate = float(config["lr"])
    args.weight_decay = float(config["weight_decay"])
    args.scheduler = config["scheduler"]
    args.grad_clip = float(config["grad_clip"])
    if args.model_type == "Prithvi-EO2-300M-PT-FT":
        args.amp = True


def validate_args(args: argparse.Namespace) -> None:
    global _ACTIVE_MODEL_NAME
    if args.model_type not in BASELINE_NAMES:
        raise ValueError(args.model_type)
    if min(args.model_runtime_height, args.model_runtime_width) < 1:
        raise ValueError("model runtime H/W must be positive")
    if args.model_type not in ("ConvLSTM-Scratch", "Prithvi-EO2-300M-PT-FT"):
        if not args.official_root:
            raise ValueError(f"{args.model_type} requires --official-root")
    if args.model_type in PT_FT_NAMES and not args.pretrained_checkpoint:
        raise ValueError(f"{args.model_type} requires --pretrained-checkpoint")
    args.pretrained_report_dir = args.pretrained_report_dir or str(
        Path(args.output_dir) / "pretrained_reports"
    )
    _resolve_optimization(args)

    selected_scheduler = args.scheduler
    selected_loss = args.loss
    args.loss = "mse"
    # The base validator knows cosine/none, while architecture-specific names
    # are installed by this file after validation.
    args.scheduler = "none" if selected_scheduler == "none" else "cosine"
    try:
        centered._BASE_VALIDATE_ARGS(args)
    finally:
        args.loss = selected_loss
        args.scheduler = selected_scheduler
    if selected_loss != LOSS_HUBER:
        raise ValueError("the seven-baseline comparison fixes masked physical Huber")
    if not math.isfinite(args.huber_delta_m) or args.huber_delta_m <= 0:
        raise ValueError("--huber-delta-m must be finite and positive")

    _ACTIVE_MODEL_NAME = args.model_type
    four._ACTIVE_MODEL_TYPE = args.model_type
    four._ACTIVE_OUTPUT_METHOD = four.OUTPUT_DIRECT
    four._ACTIVE_LOSS_KIND = LOSS_HUBER
    centered._ACTIVE_LOSS_KIND = LOSS_HUBER
    centered._ACTIVE_HUBER_DELTA_M = float(args.huber_delta_m)
    centered._ACTIVE_TRAIN_WATER_STD = None
    centered._ACTIVE_HUBER_DELTA_NORM = None
    store = centered._configure_store(args.dem_root)
    if store.static_channels != 1:
        raise AssertionError("the shared centered DEM adapter must provide one channel")


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_type": args.model_type,
        "runtime_height": int(args.model_runtime_height),
        "runtime_width": int(args.model_runtime_width),
        "official_root": str(args.official_root),
        "checkpoint_path": str(args.pretrained_checkpoint),
        "report_dir": str(args.pretrained_report_dir),
        "direct_derivative_output": True,
        "output_method": four.OUTPUT_DIRECT,
    }


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    global _LAST_BUILT_MODEL
    values = dict(config)
    model_name = values.pop("model_type")
    values.pop("direct_derivative_output", None)
    values.pop("output_method", None)
    official_root = values.pop("official_root", "")
    checkpoint_path = values.pop("checkpoint_path", "")
    core = build_flood_baseline(
        model_name,
        runtime_height=values.pop("runtime_height"),
        runtime_width=values.pop("runtime_width"),
        official_roots={model_name: official_root},
        checkpoint_paths={model_name: checkpoint_path},
        report_dir=values.pop("report_dir"),
    )
    if values:
        raise ValueError(f"unused model config keys: {sorted(values)}")
    model = TrainerDeltaAdapter(core, model_name).to(device)
    _LAST_BUILT_MODEL = model
    print("MODEL_INFO", json.dumps(model_metadata(core, model_name), ensure_ascii=False))
    return model


def build_optimizer(
    parameters: Iterable[nn.Parameter], args: argparse.Namespace
) -> torch.optim.Optimizer:
    config = OPTIMIZATION[args.model_type]
    params: Any = parameters
    if args.model_type == "Poseidon-T-PT-FT":
        if _LAST_BUILT_MODEL is None:
            raise RuntimeError("Poseidon parameter groups requested before model build")
        core = _LAST_BUILT_MODEL.core
        params = core.parameter_groups(
            backbone_lr=config["lr"], adapter_lr=config["adapter_lr"]
        )
    if config["optimizer"] == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=config["betas"],
        )
    return _REAL_ADAM(
        params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        foreach=False,
    )


def _warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int,
):
    warmup_updates = max(1, min(int(warmup_updates), total_updates))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_updates
    )
    if warmup_updates >= total_updates:
        return warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates - warmup_updates
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_updates]
    )


def make_scheduler(optimizer, args):
    total_updates = max(1, int(args.epochs) * int(args.updates_per_epoch))
    kind = OPTIMIZATION[args.model_type]["scheduler"]
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=OPTIMIZATION[args.model_type]["lr"],
            total_steps=total_updates,
        )
    if kind == "dpot_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=OPTIMIZATION[args.model_type]["lr"],
            total_steps=total_updates,
            div_factor=1e4,
            pct_start=0.2,
            final_div_factor=1e4,
        )
    if kind == "warmup_cosine_20":
        return _warmup_cosine(optimizer, total_updates, int(0.2 * total_updates))
    if kind == "gphyt_warmup_cosine":
        return _warmup_cosine(
            optimizer, total_updates, min(5000, int(0.2 * total_updates))
        )
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )
    if kind == "none":
        return None
    raise ValueError(kind)


def _with_optimizer(function, args, *call_args):
    original_adam = torch.optim.Adam

    def factory(parameters, *unused, **ignored):
        del unused, ignored
        optimizer = build_optimizer(parameters, args)
        print(
            "OPTIMIZER_INFO",
            json.dumps(
                {
                    "model_type": args.model_type,
                    **OPTIMIZATION[args.model_type],
                    "updates_per_epoch": getattr(args, "updates_per_epoch", None),
                },
                ensure_ascii=False,
            ),
        )
        return optimizer

    torch.optim.Adam = factory
    try:
        return function(*call_args)
    finally:
        torch.optim.Adam = original_adam


def train_with_optimizer(args, specs, device, distributed, rank, world_size):
    return _with_optimizer(
        _BASE_TRAIN, args, args, specs, device, distributed, rank, world_size
    )


def smoke_with_optimizer(args, specs, device, rank, world_size):
    return _with_optimizer(_BASE_SMOKE, args, args, specs, device, rank, world_size)


def run_experiment() -> None:
    """Install isolated hooks and enter the unchanged benchmark lifecycle."""
    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.compute_training_normalizer = centered._compute_training_normalizer
    baseline.FloodDomainDataset = dem_common.DemFeatureFloodDomainDataset
    baseline.rollout_sequence = dem_common.rollout_sequence
    baseline.describe_dataset = centered._describe_dataset
    baseline.load_checkpoint_file = four._selected_load_checkpoint_file
    baseline.checkpoint_payload = four._selected_checkpoint_payload
    baseline.load_eval_model = four._selected_load_eval_model
    baseline.masked_loss = four._selected_masked_loss
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.make_scheduler = make_scheduler
    baseline.train = train_with_optimizer
    baseline.run_smoke_tests = smoke_with_optimizer
    from gphyt.data.floodcast_window_integration import install_window_protocol
    install_window_protocol(globals())
    baseline.main()


if __name__ == "__main__":
    run_experiment()
