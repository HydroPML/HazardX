"""Seven official baselines with exact 21-scene trajectory training epochs.

This entrypoint composes existing, audited model and evaluation hooks.  It
changes only the training sampler/schedule, the requested DPOT stability
settings, DDP report publication, and final AMP optimizer-step accounting.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import experiments_EarthDynamics_FloodCastBench_seven_baselines as seven
import floodcast_dem_sensitivity_common as dem_common
import floodcast_transolver_centered_dem_common as centered
from gphyt.model.ddp_pretrained_report_scene_trajectory import (
    build_model_with_ddp_safe_report,
    install_ddp_report_training_scope,
)


EXPERIMENT_NAME = "FloodCastBench_Seven_128_SceneTrajectory"
DEFAULT_OUTPUT_DIR = (
    "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
    "FloodCastBench_7baselines_128_scene_trajectory"
)
DPOT_MAX_LR = 3.0e-5
DPOT_GRAD_CLIP = 1.0
EARTHFORMER_LR = 3.0e-4
DEFAULT_DDP_TIMEOUT_SECONDS = 7200
_ACTIVE_DDP_TIMEOUT_SECONDS = DEFAULT_DDP_TIMEOUT_SECONDS


def _build_parser():
    parser = seven.build_parser()
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=DEFAULT_DDP_TIMEOUT_SECONDS,
        help=(
            "Process-group collective timeout. Rank 0 performs full-grid "
            "validation while the other ranks wait, so this must exceed the "
            "slowest validation duration."
        ),
    )
    return parser


def _validate_args(args) -> None:
    global _ACTIVE_DDP_TIMEOUT_SECONDS
    seven.validate_args(args)
    if args.model_type == "EarthFormer-Scratch":
        args.amp = False
        print(
            "EARTHFORMER_STABILITY_INFO "
            + json.dumps(
                {
                    "precision": "fp32",
                    "amp": False,
                    "lr": EARTHFORMER_LR,
                    "scheduler": "warmup_cosine_20",
                    "grad_clip": 1.0,
                }
            ),
            flush=True,
        )
    if args.ddp_timeout_seconds < 1:
        raise ValueError("--ddp-timeout-seconds must be positive")
    _ACTIVE_DDP_TIMEOUT_SECONDS = int(args.ddp_timeout_seconds)


def _setup_distributed():
    """Use an explicit timeout long enough for rank-0-only validation.

    The reference lifecycle lets nonzero ranks wait in the post-validation
    state broadcast. Poseidon's full-day tiled validation can exceed
    PyTorch's ten-minute default process-group timeout, even though both ranks
    are healthy. Only the timeout changes here; backend and device placement
    remain identical to the reference setup.
    """

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=_ACTIVE_DDP_TIMEOUT_SECONDS),
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backend = None

    if rank == 0:
        print(
            "DDP_TIMEOUT_INFO "
            + json.dumps(
                {
                    "distributed": distributed,
                    "world_size": world_size,
                    "backend": backend,
                    "timeout_seconds": _ACTIVE_DDP_TIMEOUT_SECONDS,
                    "reason": "rank0_full_grid_validation",
                }
            ),
            flush=True,
        )
    return distributed, rank, world_size, local_rank, device


def _configure_requested_dpot_stability() -> None:
    config = seven.OPTIMIZATION["DPOT-S-PT-FT"]
    config["lr"] = DPOT_MAX_LR
    config["grad_clip"] = DPOT_GRAD_CLIP
    if config["optimizer"] != "Adam" or tuple(config["betas"]) != (0.9, 0.9):
        raise AssertionError("DPOT optimizer identity changed unexpectedly")
    if config["scheduler"] != "dpot_cycle":
        raise AssertionError("DPOT OneCycle scheduler changed unexpectedly")


def _configure_requested_earthformer_stability() -> None:
    config = seven.OPTIMIZATION["EarthFormer-Scratch"]
    config["lr"] = EARTHFORMER_LR
    if config["optimizer"] != "AdamW" or config["weight_decay"] != 0.0:
        raise AssertionError("EarthFormer optimizer identity changed unexpectedly")
    if config["scheduler"] != "warmup_cosine_20":
        raise AssertionError("EarthFormer scheduler changed unexpectedly")
    if config["grad_clip"] != 1.0:
        raise AssertionError("EarthFormer gradient clipping changed unexpectedly")


def _build_model(
    config: Mapping[str, Any], device: torch.device
) -> nn.Module:
    return build_model_with_ddp_safe_report(seven.build_model, config, device)


def install_experiment_hooks() -> None:
    """Install this entrypoint's process-local hooks without starting a run."""

    _configure_requested_dpot_stability()
    _configure_requested_earthformer_stability()
    baseline.build_parser = _build_parser
    baseline.validate_args = _validate_args
    baseline.setup_distributed = _setup_distributed
    baseline.compute_training_normalizer = centered._compute_training_normalizer
    baseline.FloodDomainDataset = dem_common.DemFeatureFloodDomainDataset
    baseline.rollout_sequence = dem_common.rollout_sequence
    baseline.describe_dataset = centered._describe_dataset
    baseline.load_checkpoint_file = seven.four._selected_load_checkpoint_file
    baseline.checkpoint_payload = seven.four._selected_checkpoint_payload
    baseline.load_eval_model = seven.four._selected_load_eval_model
    baseline.masked_loss = seven.four._selected_masked_loss
    baseline.model_config_from_args = seven.model_config_from_args
    baseline.build_model = _build_model
    baseline.make_scheduler = seven.make_scheduler
    baseline.train = seven.train_with_optimizer
    baseline.run_smoke_tests = seven.smoke_with_optimizer

    from gphyt.data.floodcast_window_integration import install_window_protocol

    install_window_protocol(vars(seven))

    from gphyt.data.floodcast_scene_tile_training import install_scene_tile_training

    install_scene_tile_training(vars(seven))

    from gphyt.data.floodcast_scene_tile_update_audit import (
        install_optimizer_update_audit,
    )

    install_optimizer_update_audit(vars(seven))
    install_ddp_report_training_scope()


def run_experiment() -> None:
    """Install isolated hooks and enter the unchanged benchmark lifecycle."""

    install_experiment_hooks()
    baseline.main()


if __name__ == "__main__":
    run_experiment()
