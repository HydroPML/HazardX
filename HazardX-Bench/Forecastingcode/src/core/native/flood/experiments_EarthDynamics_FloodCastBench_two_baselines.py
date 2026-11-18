"""128-window FloodCastBench entry point for scratch UNet and FNO.

This entry reuses the exact UNet/FNO backbones and optimization defaults from
``experiments_EarthDynamics_FloodCastBench_four_baselines.py`` and installs the
same audited 128/96 random-crop, tiled-validation and tiled-rollout protocol as
the seven-baseline benchmark.  Existing project files are not modified.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
import floodcast_four_baselines as four
import floodcast_transolver_centered_dem_common as centered
from gphyt.model.unet_fno_128 import (
    BASELINE_NAMES,
    FNODirectNext,
    UNetDirectNext,
)


EXPERIMENT_NAME = "FloodCastBench_Two_Scratch_128_Baselines"
DEFAULT_OUTPUT_DIR = (
    "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
    "FloodCastBench_2baselines_128"
)
LOSS_HUBER = centered.LOSS_HUBER

_BASE_TRAIN = baseline.train
_BASE_SMOKE = baseline.run_smoke_tests
_ACTIVE_MODEL_NAME = BASELINE_NAMES[0]


class TrainerDeltaAdapter(nn.Module):
    """Adapt an absolute h_next core to the validated trainer's dh boundary."""

    direct_derivative_output = True
    core_prediction = "h_next"
    trainer_adapter = "h_next_minus_h_t"

    def __init__(self, core: nn.Module, model_name: str) -> None:
        super().__init__()
        self.core = core
        self.model_name = str(model_name)

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
        "Scratch UNet/FNO on the audited FloodCastBench 128-window protocol"
    )
    model_action = _action(parser, "model_type")
    model_action.choices = BASELINE_NAMES
    model_action.default = "UNet"
    _action(parser, "output_method").default = four.OUTPUT_DIRECT
    _action(parser, "loss").choices = (LOSS_HUBER,)
    _action(parser, "loss").default = LOSS_HUBER
    parser.add_argument("--model-runtime-height", type=int, default=128)
    parser.add_argument("--model-runtime-width", type=int, default=128)
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR,
        normalization="train_standard",
        output_method=four.OUTPUT_DIRECT,
        loss=LOSS_HUBER,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    global _ACTIVE_MODEL_NAME
    if args.model_type not in BASELINE_NAMES:
        raise ValueError(args.model_type)
    if args.output_method != four.OUTPUT_DIRECT:
        raise ValueError("UNet/FNO 128 benchmark fixes direct h_next prediction")
    if args.loss != LOSS_HUBER:
        raise ValueError("UNet/FNO 128 benchmark fixes MSE-matched masked Huber")
    four.validate_args(args)
    if args.output_method != four.OUTPUT_DIRECT or args.loss != LOSS_HUBER:
        raise AssertionError("UNet/FNO output/loss resolution changed unexpectedly")
    _ACTIVE_MODEL_NAME = str(args.model_type)


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = four.model_config_from_args(args)
    if config.get("output_method") != four.OUTPUT_DIRECT:
        raise AssertionError("checkpoint model config must record direct h_next")
    return config


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    values = dict(config)
    model_name = str(values.pop("model_type"))
    values.pop("direct_derivative_output", None)
    output_method = values.pop("output_method", four.OUTPUT_DIRECT)
    if output_method != four.OUTPUT_DIRECT:
        raise ValueError("checkpoint does not use direct h_next prediction")
    if model_name == "UNet":
        core = UNetDirectNext(**values)
    elif model_name == "FNO":
        core = FNODirectNext(**values)
    else:
        raise ValueError(f"unknown two-baseline model {model_name!r}")
    model = TrainerDeltaAdapter(core, model_name).to(device)
    info = {
        "experiment": EXPERIMENT_NAME,
        "model_type": model_name,
        "initialization": "scratch",
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "public_core_input": "[B,3,128,128] = [h_t, rainfall_t, DEM]",
        "public_core_output": "[B,1,128,128] = h_next",
        "trainer_adapter": "h_next_minus_h_t",
        "spatial_resize": False,
        "configuration": values,
    }
    print("MODEL_INFO", json.dumps(info, ensure_ascii=False))
    return model


def _with_optimizer(function: Any, args: argparse.Namespace, *call_args: Any):
    original_adam = torch.optim.Adam

    def factory(
        parameters: Iterable[nn.Parameter], *unused: Any, **ignored: Any
    ) -> torch.optim.Optimizer:
        del unused, ignored
        optimizer = four.build_optimizer(parameters, args)
        print("OPTIMIZER_INFO", json.dumps(four.optimizer_summary(args)))
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
    return _with_optimizer(
        _BASE_SMOKE, args, args, specs, device, rank, world_size
    )


def run_experiment() -> None:
    """Install process-local hooks and enter the shared benchmark lifecycle."""
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
    baseline.make_scheduler = four.make_scheduler
    baseline.train = train_with_optimizer
    baseline.run_smoke_tests = smoke_with_optimizer

    from gphyt.data.floodcast_window_integration import install_window_protocol

    install_window_protocol(globals())
    baseline.main()


if __name__ == "__main__":
    run_experiment()
