"""Post-training optimizer-step audit for the scene-trajectory protocol."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, MutableMapping

import torch

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
from gphyt.data.floodcast_scene_tile_trajectory import EXPECTED_SAMPLES


MAX_SKIPPED_RATIO = 1.0e-3


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def install_optimizer_update_audit(namespace: MutableMapping[str, Any]) -> None:
    """Wrap the installed trajectory trainer without replacing its loop."""

    previous_train = baseline.train

    def train(args, specs, device, distributed, rank, world_size):
        checkpoint_path = previous_train(
            args, specs, device, distributed, rank, world_size
        )
        effective_batch_size = (
            int(world_size) * int(args.micro_batch_size) * int(args.accum_steps)
        )
        updates_per_epoch = math.ceil(EXPECTED_SAMPLES / effective_batch_size)
        expected_updates = int(args.epochs) * updates_per_epoch
        audit = None
        error = None
        if baseline.is_main_process(rank):
            last_path = baseline.canonical_path(args.output_dir) / "last.pt"
            checkpoint = baseline.load_checkpoint_file(last_path, torch.device("cpu"))
            successful_updates = int(checkpoint["global_update"])
            skipped_updates = expected_updates - successful_updates
            if skipped_updates < 0:
                error = (
                    f"successful optimizer updates {successful_updates} exceed "
                    f"expected {expected_updates}"
                )
                skipped_ratio = -1.0
            else:
                skipped_ratio = skipped_updates / max(expected_updates, 1)
                if skipped_ratio > MAX_SKIPPED_RATIO:
                    error = (
                        f"AMP skipped-update ratio {skipped_ratio:.8f} exceeds "
                        f"limit {MAX_SKIPPED_RATIO:.8f}"
                    )
            audit = {
                "expected_optimizer_updates": expected_updates,
                "successful_optimizer_updates": successful_updates,
                "amp_skipped_updates": skipped_updates,
                "amp_skipped_ratio": skipped_ratio,
                "maximum_allowed_amp_skipped_ratio": MAX_SKIPPED_RATIO,
                "updates_per_epoch": updates_per_epoch,
                "logical_samples_per_epoch": EXPECTED_SAMPLES,
                "effective_batch_size": effective_batch_size,
                "scheduler_steps_only_after_successful_optimizer_step": True,
                "passed": error is None,
            }
            _atomic_write_json(
                audit,
                baseline.canonical_path(args.output_dir)
                / "optimizer_step_audit.json",
            )
            print(
                "OPTIMIZER_STEP_AUDIT",
                json.dumps(audit, ensure_ascii=False, sort_keys=True),
            )
        state = baseline.broadcast_object(
            {"audit": audit, "error": error}
            if baseline.is_main_process(rank)
            else None
        )
        if state["error"] is not None:
            raise RuntimeError(state["error"])
        namespace["LAST_OPTIMIZER_STEP_AUDIT"] = state["audit"]
        return checkpoint_path

    baseline.train = train

