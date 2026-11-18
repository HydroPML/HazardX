"""UNet/FNO benchmark with exact 21-scene, one-tile trajectory epochs.

Model, loss, optimizer, scheduler, validation, source test and Mozambique OOD
hooks are imported unchanged from the existing two-baseline entrypoint.  This
entrypoint replaces only its training dataset/sampler and the minimal trainer
iteration needed to consume an odd 6027 samples exactly under DDP.  Its DDP
timeout also allows peers to wait for rank-0-only full-grid validation.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import experiments_EarthDynamics_FloodCastBench_two_baselines as two
import floodcast_dem_sensitivity_common as dem_common
import floodcast_transolver_centered_dem_common as centered


EXPERIMENT_NAME = "FloodCastBench_Two_Scratch_128_SceneTrajectory"
DEFAULT_DDP_TIMEOUT_SECONDS = 7200
_ACTIVE_DDP_TIMEOUT_SECONDS = DEFAULT_DDP_TIMEOUT_SECONDS


def _build_parser():
    parser = two.build_parser()
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
    if args.ddp_timeout_seconds < 1:
        raise ValueError("--ddp-timeout-seconds must be positive")
    two.validate_args(args)
    _ACTIVE_DDP_TIMEOUT_SECONDS = int(args.ddp_timeout_seconds)


def _setup_distributed():
    """Keep the reference setup, with a validation-aware collective timeout."""

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


def install_experiment_hooks() -> None:
    """Install isolated hooks without starting data loading or training."""

    baseline.build_parser = _build_parser
    baseline.validate_args = _validate_args
    baseline.setup_distributed = _setup_distributed
    baseline.compute_training_normalizer = centered._compute_training_normalizer
    baseline.FloodDomainDataset = dem_common.DemFeatureFloodDomainDataset
    baseline.rollout_sequence = dem_common.rollout_sequence
    baseline.describe_dataset = centered._describe_dataset
    baseline.load_checkpoint_file = two.four._selected_load_checkpoint_file
    baseline.checkpoint_payload = two.four._selected_checkpoint_payload
    baseline.load_eval_model = two.four._selected_load_eval_model
    baseline.masked_loss = two.four._selected_masked_loss
    baseline.model_config_from_args = two.model_config_from_args
    baseline.build_model = two.build_model
    baseline.make_scheduler = two.four.make_scheduler
    baseline.train = two.train_with_optimizer
    baseline.run_smoke_tests = two.smoke_with_optimizer

    # First retain the existing split audit and unchanged 128/96 tiled
    # validation/test/rollout hooks.
    from gphyt.data.floodcast_window_integration import install_window_protocol

    install_window_protocol(vars(two))

    # Then replace only the train dataset/schedule and its exact-tail iterator.
    from gphyt.data.floodcast_scene_tile_training import install_scene_tile_training

    install_scene_tile_training(vars(two))


def run_experiment() -> None:
    install_experiment_hooks()
    baseline.main()


if __name__ == "__main__":
    run_experiment()

