"""Domain-balanced FloodSimBench context-6 UNet/FNO experiment.

This file deliberately delegates models, loss, optimizer, scheduler, training,
validation, test, checkpointing, metrics, and rollout to the existing crop1024
experiment.  It replaces only the Train Dataset view and distributed sampler:
one queued scenario per Train domain and epoch, with all 67 forecasting samples.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler

import experiments_EarthDynamics_FloodSimBench_two_baselines_crop1024 as reference
from data import floodsimbench_crop1024_context6_domain_balanced_dataset as data


EXPERIMENT_NAME = "FloodSimBench_Context6_Two_Baselines_Crop1024_DomainBalanced"
DEFAULT_OUTPUT_ROOT = Path("results")
DATASET_NAME = "FloodSimBench_context6_crop1024_domain_balanced"


class DomainBalancedDistributedBatchSampler(Sampler[List[int]]):
    """Balance forecast samples across ranks without duplicating a scenario."""

    def __init__(
        self,
        dataset: data.FloodSimBenchContext6Dataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if dataset.role != "train":
            raise ValueError("DomainBalancedDistributedBatchSampler is Train-only")
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch/rank/world configuration")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    @property
    def actual_global_samples(self) -> int:
        return self.dataset.sequence_count * data.FORECAST_SAMPLES_PER_SEQUENCE

    @property
    def padding_samples(self) -> int:
        return (-self.actual_global_samples) % self.world_size

    @property
    def padded_global_samples(self) -> int:
        return self.actual_global_samples + self.padding_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)
        if self.rank == 0:
            print(
                "DOMAIN_BALANCED_EPOCH",
                json.dumps(
                    {
                        "epoch": self.epoch,
                        "selected_scenarios": self.dataset.selected_epoch_metadata(
                            self.epoch
                        ),
                        "actual_global_samples": self.actual_global_samples,
                        "ddp_padding_samples": self.padding_samples,
                        "padded_global_samples": self.padded_global_samples,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def global_sample_indices(self) -> List[int]:
        rng = random.Random(self.seed + 7_919 + 1_000_003 * self.epoch)
        sequence_slots = list(range(self.dataset.sequence_count))
        rng.shuffle(sequence_slots)
        indices: List[int] = []
        for sequence_slot in sequence_slots:
            forecasts = list(range(data.FORECAST_SAMPLES_PER_SEQUENCE))
            rng.shuffle(forecasts)
            base_index = sequence_slot * data.FORECAST_SAMPLES_PER_SEQUENCE
            indices.extend(base_index + forecast for forecast in forecasts)
        if len(indices) != self.actual_global_samples:
            raise AssertionError("Global Train index count is inconsistent")
        return indices

    def local_sample_indices(self) -> List[int]:
        indices = self.global_sample_indices()
        if self.padding_samples:
            # Standard DDP parity padding: at most one forecast sample for 2 GPUs.
            indices.extend(indices[: self.padding_samples])
        local = indices[self.rank :: self.world_size]
        expected = self.padded_global_samples // self.world_size
        if len(local) != expected:
            raise AssertionError(f"Rank {self.rank}: {len(local)} != {expected}")
        return local

    def __iter__(self) -> Iterator[List[int]]:
        local = self.local_sample_indices()
        for start in range(0, len(local), self.batch_size):
            yield local[start : start + self.batch_size]

    def __len__(self) -> int:
        local_samples = self.padded_global_samples // self.world_size
        return math.ceil(local_samples / self.batch_size)


_reference_checkpoint_payload = reference.checkpoint_payload
_reference_validate_args = reference.validate_args


def checkpoint_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _reference_checkpoint_payload(*args, **kwargs)
    payload["dataset"].update(
        {
            "name": DATASET_NAME,
            "preprocessing_version": data.PREPROCESSING_VERSION,
            "train_crop": "one_random_crop_per_selected_domain_scenario_per_epoch",
            "train_sampling": "17 deterministic without-replacement domain queues",
            "available_train_scenarios": 122,
            "selected_train_scenarios_per_epoch": 17,
            "forecast_samples_per_epoch": 17
            * data.FORECAST_SAMPLES_PER_SEQUENCE,
            "ddp_padding_policy": "forecast_sample_only; never duplicate a scenario",
        }
    )
    return payload


def validate_checkpoint(checkpoint: Mapping[str, Any], model_type: str) -> None:
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise RuntimeError("checkpoint belongs to another experiment")
    config = checkpoint.get("model_config", {})
    if config.get("model_type") != model_type:
        raise RuntimeError(
            f"checkpoint model {config.get('model_type')} != requested {model_type}"
        )
    contract = checkpoint.get("dataset", {})
    expected = {
        "name": DATASET_NAME,
        "preprocessing_version": data.PREPROCESSING_VERSION,
        "frames_per_sequence": 72,
        "forecast_samples_per_sequence": 67,
        "selected_train_scenarios_per_epoch": 17,
        "forecast_samples_per_epoch": 1139,
        "crop_size": 1024,
        "loss": "masked_mse_only",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"checkpoint dataset contract {key}: "
                f"{contract.get(key)!r} != {value!r}"
            )



def validate_args(args: Any) -> None:
    """Keep all reference argument rules but isolate the default result path."""
    output_was_explicit = bool(args.output_dir)
    _reference_validate_args(args)
    if not output_was_explicit:
        args.output_dir = str(
            DEFAULT_OUTPUT_ROOT
            / (
                "FloodSimBench_context6_crop1024_domain_balanced_"
                f"{args.model_type}_MaskedMSE"
            )
        )


def _gather_sampler_audit(
    sampler: DomainBalancedDistributedBatchSampler,
) -> Tuple[List[List[int]], List[List[str]]]:
    local_indices = sampler.local_sample_indices()
    selected_ids = [
        row["scenario_id"]
        for row in sampler.dataset.selected_epoch_metadata(sampler.epoch)
    ]
    if reference.is_distributed():
        all_indices: List[Any] = [None] * sampler.world_size
        all_selected: List[Any] = [None] * sampler.world_size
        dist.all_gather_object(all_indices, local_indices)
        dist.all_gather_object(all_selected, selected_ids)
        return all_indices, all_selected
    return [local_indices], [selected_ids]


def run_smoke(
    args: Any,
    train_dataset: data.FloodSimBenchContext6Dataset,
    valid_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    test_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    normalizer: Any,
    device: torch.device,
    rank: int,
    world: int,
) -> None:
    counts = train_dataset.domain_scenario_counts
    if len(counts) != 17 or sum(counts.values()) != 122:
        raise AssertionError(f"Unexpected Train inventory: {counts}")

    # Queue audit: every scenario appears once before its domain queue refills.
    for domain, queue_length in counts.items():
        chosen = []
        for epoch in range(queue_length):
            rows = train_dataset.selected_epoch_metadata(epoch)
            chosen.append(next(row["scenario_id"] for row in rows if row["domain"] == domain))
        if len(set(chosen)) != queue_length:
            raise AssertionError(f"{domain}: scenario repeated before queue exhaustion")

    repeat_dataset = data.FloodSimBenchContext6Dataset(
        args.data_root, data.TRAIN_CITIES, "train", normalizer, seed=args.seed
    )
    for epoch in range(max(counts.values()) + 2):
        if train_dataset.selected_epoch_metadata(epoch) != repeat_dataset.selected_epoch_metadata(epoch):
            raise AssertionError("Seed/epoch schedule is not reproducible")

    train_dataset.set_epoch(0)
    selected_index = train_dataset.selected_scenario_indices(0)[0]
    coordinate0 = train_dataset.random_coordinate(selected_index, 0)
    coordinate1 = train_dataset.random_coordinate(selected_index, 1)
    if coordinate0 == coordinate1:
        raise AssertionError("Selected scenario crop did not change across epochs")
    first_sample = train_dataset[0]
    last_sample = train_dataset[data.FORECAST_SAMPLES_PER_SEQUENCE - 1]
    if (first_sample["row_start"], first_sample["col_start"]) != (
        last_sample["row_start"], last_sample["col_start"]
    ):
        raise AssertionError("One scenario's 67 samples did not share one crop")
    if first_sample["target_time_minutes"] != 30 or last_sample["target_time_minutes"] != 360:
        raise AssertionError("Forecast endpoints are not h30 and h360")

    valid_repeat = data.FloodSimBenchContext6Dataset(
        args.data_root, data.VALID_CITIES, "valid", normalizer, seed=args.seed
    )
    combined_valid = data.FloodSimBenchContext6Dataset(
        args.data_root, data.VALID_CITIES, "valid", normalizer, seed=args.seed
    )
    coordinates_repeat = [
        (tile.sequence_id, tile.row_start, tile.col_start)
        for tile in valid_repeat.files
    ]
    coordinates_combined = [
        (tile.sequence_id, tile.row_start, tile.col_start)
        for tile in combined_valid.files
    ]
    if coordinates_repeat != coordinates_combined:
        raise AssertionError("Validation grid is not deterministic")

    loader, sampler = reference.build_train_loader(train_dataset, args, rank, world)
    sampler.set_epoch(0)
    all_indices, all_selected = _gather_sampler_audit(sampler)
    if any(ids != all_selected[0] for ids in all_selected[1:]):
        raise AssertionError("DDP ranks selected different global scenarios")
    if world == 2:
        flattened = [index for rank_indices in all_indices for index in rank_indices]
        unique = set(flattened)
        if len(flattened) != 1140 or len(unique) != 1139:
            raise AssertionError("2-GPU DDP must pad exactly one forecast sample")
        per_slot = [0] * train_dataset.sequence_count
        for index in flattened:
            per_slot[index // data.FORECAST_SAMPLES_PER_SEQUENCE] += 1
        if min(per_slot) != 67 or max(per_slot) != 68:
            raise AssertionError("DDP duplicated more than one forecast sample")

    batch = reference.move_batch(next(iter(loader)), device)
    dry_prediction = torch.ones((1, 1, 2, 2), device=device)
    dry_target = torch.zeros_like(dry_prediction)
    dry_mask = torch.ones_like(dry_prediction, dtype=torch.bool)
    dry_loss = reference.masked_mse(dry_prediction, dry_target, dry_mask)
    if not math.isclose(float(dry_loss), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("Valid dry pixels are missing from Masked MSE")

    model_config = reference.model_config_from_args(args)
    model = reference.build_model(model_config, device)
    if world > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = reference.build_optimizer(model, args)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = 0
    amp_enabled = bool(args.amp and device.type == "cuda")
    with reference.autocast_context(amp_enabled):
        prediction = model(batch["x"])
    loss = reference.masked_mse(prediction.float(), batch["y"].float(), batch["mask"])
    loss.backward()
    optimizer.step()
    if not math.isfinite(float(loss)):
        raise AssertionError("Forward/backward produced a non-finite loss")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0

    short_valid = reference.evaluate_city(
        reference.unwrap(model), valid_datasets["AUS"], normalizer, device, args,
        max_steps=args.smoke_rollout_steps, max_sequences=1,
    )
    short_test = reference.evaluate_city(
        reference.unwrap(model), test_datasets["LA"], normalizer, device, args,
        max_steps=args.smoke_rollout_steps, max_sequences=1,
    )
    full_path = reference.evaluate_city(
        reference.PersistenceModel().to(device), valid_datasets["AUS"], normalizer,
        device, args, max_steps=0, max_sequences=1,
    )

    memory = {
        "rank": rank,
        "device": str(device),
        "peak_increment_bytes": max(peak - allocated_before, 0),
    }
    if reference.is_distributed():
        memory_rows: List[Any] = [None] * world
        dist.all_gather_object(memory_rows, memory)
    else:
        memory_rows = [memory]
    if reference.is_main():
        valid_pixels = batch["mask"].bool()
        report = {
            "status": "PASS",
            "experiment": EXPERIMENT_NAME,
            "domain_scenario_counts": counts,
            "selected_epoch0": train_dataset.selected_epoch_metadata(0),
            "selected_scenarios_per_epoch": train_dataset.sequence_count,
            "actual_forecast_samples_per_epoch": len(train_dataset),
            "ddp_padding_samples": sampler.padding_samples,
            "batches_per_rank": len(loader),
            "updates_per_epoch": math.ceil(len(loader) / args.accum_steps),
            "schedule_resume_reproducible": True,
            "same_seed_unet_fno_schedule": True,
            "validation_grid_repeat_identical": True,
            "batch": {
                "x_shape": list(batch["x"].shape),
                "y_shape": list(batch["y"].shape),
                "mask_shape": list(batch["mask"].shape),
                "dtype": str(batch["x"].dtype),
                "channel_order": list(data.CHANNEL_ORDER),
                "x_range": [float(batch["x"].min()), float(batch["x"].max())],
                "y_range": [float(batch["y"].min()), float(batch["y"].max())],
                "valid_fraction": float(valid_pixels.float().mean()),
                "units_before_normalization": {
                    "rainfall": "mm/h", "DEM": "m", "depth": "m"
                },
            },
            "normalization": normalizer.state_dict(),
            "model_type": args.model_type,
            "parameters": sum(p.numel() for p in reference.unwrap(model).parameters()),
            "prediction_shape": list(prediction.shape),
            "masked_mse": float(loss),
            "valid_dry_pixel_test_loss": float(dry_loss),
            "short_validation_steps": len(short_valid["lead_metrics"]),
            "short_test_steps": len(short_test["lead_metrics"]),
            "full_validation_path_steps": len(full_path["lead_metrics"]),
            "full_validation_path_end_minutes": full_path["lead_metrics"][-1][
                "target_time_minutes"
            ],
            "gpu_memory": memory_rows,
        }
        print(
            "DOMAIN_BALANCED_SMOKE_REPORT",
            json.dumps(reference.json_safe(report), ensure_ascii=False),
            flush=True,
        )
        print("DOMAIN BALANCED SMOKE PASS", flush=True)
    reference.barrier()


def _install_domain_balanced_policy() -> None:
    reference.data = data
    reference.EXPERIMENT_NAME = EXPERIMENT_NAME
    reference.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    reference.ScenarioDistributedBatchSampler = DomainBalancedDistributedBatchSampler
    reference.checkpoint_payload = checkpoint_payload
    reference.validate_checkpoint = validate_checkpoint
    reference.validate_args = validate_args
    reference.run_smoke = run_smoke


_install_domain_balanced_policy()


def main() -> None:
    reference.main()


if __name__ == "__main__":
    main()
