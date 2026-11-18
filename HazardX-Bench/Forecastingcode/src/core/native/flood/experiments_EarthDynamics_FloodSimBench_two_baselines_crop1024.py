"""FloodSimBench 1 m context-6 dynamic forecasting with UNet or FNO.

This independent experiment reuses the repository UNet2d/FNO2d backbones and
their Adam/cosine defaults, while implementing the FloodSimBench dynamic task

    R[t-5:t], DEM, h[t-5:t-1] -> h[t].

Training uses one random 1024 crop per scenario and epoch.  Validation/test use
fixed non-overlapping 1024 grids and strict 67-step autoregressive rollout.
Only masked MSE is optimized.  Existing experiments are not modified.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as metrics_base
import floodcast_four_baselines as four
from data import floodsimbench_crop1024_context6_dataset as data
from models.FNO import FNO2d
from models.Unet import UNet2d


EXPERIMENT_NAME = "FloodSimBench_Context6_Two_Baselines_Crop1024"
DEFAULT_OUTPUT_ROOT = Path("results")
MODEL_TYPES = ("UNet", "FNO")


def json_safe(value: Any) -> Any:
    return metrics_base.json_safe(value)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank_world() -> Tuple[int, int]:
    if not is_distributed():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def is_main() -> bool:
    return rank_world()[0] == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def setup_distributed(timeout_hours: float) -> Tuple[bool, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not math.isfinite(timeout_hours) or timeout_hours <= 0:
        raise ValueError("--ddp-timeout-hours must be finite and positive")
    if distributed:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
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
            timeout=timedelta(hours=float(timeout_hours)),
        )
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backend = "none"
    if rank == 0:
        print(
            "DISTRIBUTED_INFO",
            json.dumps(
                {
                    "distributed": distributed,
                    "backend": backend,
                    "world_size": world,
                    "timeout_hours": timeout_hours,
                }
            ),
            flush=True,
        )
    return distributed, rank, world, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


class ScenarioDistributedBatchSampler(Sampler[List[int]]):
    """Shard complete scenarios across ranks and group their 67 samples."""

    def __init__(
        self,
        dataset: data.FloodSimBenchContext6Dataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if dataset.role != "train":
            raise ValueError("ScenarioDistributedBatchSampler is Train-only")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def _local_scenarios(self) -> List[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        indices = list(range(self.dataset.sequence_count))
        rng.shuffle(indices)
        missing = (-len(indices)) % self.world_size
        if missing:
            indices.extend(indices[:missing])
        return indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + 7_919 + 1_000_003 * self.epoch)
        for scenario_index in self._local_scenarios():
            forecasts = list(range(data.FORECAST_SAMPLES_PER_SEQUENCE))
            rng.shuffle(forecasts)
            base = scenario_index * data.FORECAST_SAMPLES_PER_SEQUENCE
            for start in range(0, len(forecasts), self.batch_size):
                yield [base + item for item in forecasts[start : start + self.batch_size]]

    def __len__(self) -> int:
        local_scenarios = math.ceil(self.dataset.sequence_count / self.world_size)
        batches_per_scenario = math.ceil(
            data.FORECAST_SAMPLES_PER_SEQUENCE / self.batch_size
        )
        return local_scenarios * batches_per_scenario


class Context6UNet(nn.Module):
    """Repository UNet2d with 12 physical input channels."""

    def __init__(self, init_features: int = 32, grid_type: str = "cartesian") -> None:
        super().__init__()
        self.core = UNet2d(
            in_channels=data.INPUT_CHANNELS,
            out_channels=1,
            init_features=int(init_features),
            grid_type=str(grid_type),
        )
        self.init_features = int(init_features)
        self.grid_type = str(grid_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_model_input(x)
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        fields = x
        if pad_h or pad_w:
            fields = torch.nn.functional.pad(fields, (0, pad_w, 0, pad_h))
        official = fields.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        output = self.core(official)
        prediction = output[:, :height, :width, 0, 0].unsqueeze(1).contiguous()
        return _validate_model_output(prediction, x)


class Context6FNO(nn.Module):
    """Repository four-layer FNO2d with 12 physical input channels."""

    def __init__(
        self,
        modes_height: int = 12,
        modes_width: int = 12,
        width: int = 20,
        grid_type: str = "cartesian",
    ) -> None:
        super().__init__()
        self.core = FNO2d(
            num_channels=data.INPUT_CHANNELS,
            modes1=int(modes_height),
            modes2=int(modes_width),
            width=int(width),
            initial_step=1,
            grid_type=str(grid_type),
        )
        for name in ("conv0", "conv1", "conv2", "conv3"):
            setattr(
                self.core,
                name,
                four.DDPCompatibleSpectralConv2d(getattr(self.core, name)),
            )
        self.modes_height = int(modes_height)
        self.modes_width = int(modes_width)
        self.width = int(width)
        self.grid_type = str(grid_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_model_input(x)
        official = x.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        with torch.cuda.amp.autocast(enabled=False):
            output = self.core(official.float())
        prediction = output[:, :, :, 0, 0].unsqueeze(1).contiguous()
        return _validate_model_output(prediction, x)


def _validate_model_input(x: torch.Tensor) -> None:
    expected = (data.INPUT_CHANNELS, data.CROP_SIZE, data.CROP_SIZE)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise AssertionError(f"model input {tuple(x.shape)} != [B,{expected}]")
    if not torch.isfinite(x).all():
        raise FloatingPointError("model input contains NaN/Inf")


def _validate_model_output(prediction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    expected = (x.shape[0], 1, x.shape[-2], x.shape[-1])
    if tuple(prediction.shape) != expected:
        raise AssertionError(f"prediction {tuple(prediction.shape)} != {expected}")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("prediction contains NaN/Inf")
    return prediction


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.model_type == "UNet":
        return {
            "model_type": "UNet",
            "in_channels": data.INPUT_CHANNELS,
            "out_channels": 1,
            "init_features": args.unet_init_features,
            "grid_type": args.unet_grid_type,
        }
    return {
        "model_type": "FNO",
        "in_channels": data.INPUT_CHANNELS,
        "out_channels": 1,
        "modes_height": args.fno_modes_height,
        "modes_width": args.fno_modes_width,
        "width": args.fno_width,
        "spectral_layers": 4,
        "grid_type": args.fno_grid_type,
    }


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    values = dict(config)
    model_type = values.pop("model_type")
    if values.pop("in_channels") != data.INPUT_CHANNELS or values.pop("out_channels") != 1:
        raise ValueError("checkpoint physical channel contract is incompatible")
    if model_type == "UNet":
        model = Context6UNet(**values)
    elif model_type == "FNO":
        if values.pop("spectral_layers") != 4:
            raise ValueError("FNO checkpoint must use four spectral layers")
        model = Context6FNO(**values)
    else:
        raise ValueError(model_type)
    model = model.to(device)
    if is_main():
        print(
            "MODEL_INFO",
            json.dumps(
                {
                    "experiment": EXPERIMENT_NAME,
                    "model_type": model_type,
                    "parameters_total": sum(p.numel() for p in model.parameters()),
                    "parameters_trainable": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "physical_input": "[B,12,1024,1024]",
                    "channel_order": list(data.CHANNEL_ORDER),
                    "output": "direct normalized h[t] [B,1,1024,1024]",
                    "configuration": values,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise AssertionError(
            f"masked MSE shapes disagree: {prediction.shape}, {target.shape}, {mask.shape}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("prediction contains NaN/Inf")
    valid = mask.bool() & torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("masked MSE has no valid target pixels")
    loss = ((prediction.float() - target.float()).square())[valid].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("masked MSE is non-finite")
    return loss


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    if is_main():
        print(
            "OPTIMIZER_INFO",
            json.dumps(
                {
                    "optimizer": "Adam",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "scheduler": "CosineAnnealingLR",
                }
            ),
            flush=True,
        )
    return optimizer


def autocast_context(enabled: bool) -> contextlib.AbstractContextManager:
    return metrics_base.autocast_context(enabled)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in ("x", "y", "mask"):
        moved[key] = batch[key].to(device, non_blocking=True)
    _validate_model_input(moved["x"])
    return moved


def build_datasets(
    args: argparse.Namespace, normalizer: Any
) -> Tuple[data.FloodSimBenchContext6Dataset, Dict[str, data.FloodSimBenchContext6Dataset], Dict[str, data.FloodSimBenchContext6Dataset]]:
    train = data.FloodSimBenchContext6Dataset(
        args.data_root, data.TRAIN_CITIES, "train", normalizer, seed=args.seed
    )
    valid = {
        city: data.FloodSimBenchContext6Dataset(
            args.data_root, (city,), "valid", normalizer, seed=args.seed
        )
        for city in data.VALID_CITIES
    }
    test = {
        city: data.FloodSimBenchContext6Dataset(
            args.data_root, (city,), "test", normalizer, seed=args.seed
        )
        for city in data.TEST_CITIES
    }
    return train, valid, test


def build_train_loader(
    dataset: data.FloodSimBenchContext6Dataset,
    args: argparse.Namespace,
    rank: int,
    world: int,
) -> Tuple[DataLoader, ScenarioDistributedBatchSampler]:
    sampler = ScenarioDistributedBatchSampler(
        dataset, args.batch_size, args.seed, rank, world
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=data.homogeneous_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    return loader, sampler


def reduce_pair(total: float, count: int, device: torch.device) -> Tuple[float, int]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    sampler: ScenarioDistributedBatchSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_update: int,
) -> Tuple[float, int, float]:
    sampler.set_epoch(epoch)
    model.train()
    iterator = iter(loader)
    num_batches = len(loader)
    total_loss = 0.0
    loss_count = 0
    start_time = time.perf_counter()
    amp_enabled = bool(args.amp and device.type == "cuda")
    updates = math.ceil(num_batches / args.accum_steps)
    for update_index in range(updates):
        group_size = min(
            args.accum_steps, num_batches - update_index * args.accum_steps
        )
        optimizer.zero_grad(set_to_none=True)
        for micro in range(group_size):
            batch = move_batch(next(iterator), device)
            should_sync = micro == group_size - 1
            sync_context = (
                contextlib.nullcontext()
                if should_sync or not isinstance(model, DDP)
                else model.no_sync()
            )
            with sync_context:
                with autocast_context(amp_enabled):
                    prediction = model(batch["x"])
                with autocast_context(False):
                    loss = masked_mse(prediction.float(), batch["y"].float(), batch["mask"])
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach().cpu())
            loss_count += 1
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("gradient norm is NaN/Inf")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        succeeded = (not amp_enabled) or scaler.get_scale() >= old_scale
        if succeeded:
            global_update += 1
            scheduler.step()
        if is_main() and args.log_every_updates > 0 and (
            (update_index + 1) % args.log_every_updates == 0
            or update_index + 1 == updates
        ):
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "update_in_epoch": update_index + 1,
                        "updates_per_epoch": updates,
                        "global_update": global_update,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )
    reduced_total, reduced_count = reduce_pair(total_loss, loss_count, device)
    return reduced_total / max(reduced_count, 1), global_update, time.perf_counter() - start_time


def build_context_input(
    rain_values_mm_h: Sequence[float],
    terrain: torch.Tensor,
    depth_history_normalized: Sequence[torch.Tensor],
    valid: torch.Tensor,
    normalizer: Any,
) -> torch.Tensor:
    if len(rain_values_mm_h) != data.RAINFALL_CONTEXT_STEPS:
        raise AssertionError("rainfall context must have six values")
    if len(depth_history_normalized) != data.DEPTH_HISTORY_STEPS:
        raise AssertionError("depth context must have five states")
    rain_channels = [
        torch.full_like(
            terrain,
            (float(value) - normalizer.mean[1]) / normalizer.std[1],
        )
        for value in rain_values_mm_h
    ]
    channels = [*rain_channels, terrain, *depth_history_normalized]
    x = torch.cat(channels, dim=1)
    x = torch.where(valid.expand_as(x), x, torch.zeros_like(x))
    return x


def rollout_sequence(
    model: nn.Module,
    dataset: data.FloodSimBenchContext6Dataset,
    sequence_index: int,
    normalizer: Any,
    device: torch.device,
    *,
    max_steps: int = 0,
) -> Dict[str, Any]:
    tile = dataset.eval_tile(sequence_index)
    steps = (
        data.FORECAST_SAMPLES_PER_SEQUENCE
        if max_steps <= 0
        else min(int(max_steps), data.FORECAST_SAMPLES_PER_SEQUENCE)
    )
    terrain_cpu = dataset.load_eval_terrain(tile)
    valid_cpu = torch.isfinite(terrain_cpu)
    if not bool(valid_cpu.any()):
        raise ValueError(f"{tile.sequence_id}: no valid pixels")
    terrain = torch.where(
        valid_cpu, terrain_cpu, torch.zeros_like(terrain_cpu)
    )[None, None].to(device)
    valid = valid_cpu[None, None].to(device)
    history: List[torch.Tensor] = []
    for frame_index in range(data.DEPTH_HISTORY_STEPS):
        depth = dataset.load_eval_depth(tile, frame_index)
        if not torch.equal(torch.isfinite(depth), valid_cpu):
            raise ValueError(f"{tile.sequence_id}: initial depth/DEM masks differ")
        normalized = normalizer.normalize(depth, 0)
        normalized = torch.where(valid_cpu, normalized, torch.zeros_like(normalized))
        history.append(normalized[None, None].to(device))

    lead_metrics: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for forecast_index in range(steps):
            rain_indices, _, target_index = data.forecasting_frame_indices(
                forecast_index
            )
            rain_values = [
                data.rainfall_at_frame_mm_h(tile.scenario.rainfall_mm, item)
                for item in rain_indices
            ]
            x = build_context_input(rain_values, terrain, history, valid, normalizer)
            prediction_n = model(x).float()
            prediction_phys = normalizer.denormalize(prediction_n, 0)
            prediction_phys = torch.where(
                valid,
                prediction_phys.clamp_min(0.0),
                torch.full_like(prediction_phys, float("nan")),
            )
            prediction_phys = torch.where(
                valid & (prediction_phys < data.DEPTH_CLEAN_THRESHOLD_M),
                torch.zeros_like(prediction_phys),
                prediction_phys,
            )
            if not torch.isfinite(prediction_phys[valid]).all():
                raise FloatingPointError(
                    f"{tile.sequence_id}: non-finite rollout prediction"
                )
            target_cpu = dataset.load_eval_depth(tile, target_index)
            if not torch.equal(torch.isfinite(target_cpu), valid_cpu):
                raise ValueError(f"{tile.sequence_id}: target/DEM masks differ")
            target = target_cpu[None, None].to(device)
            row = metrics_base.masked_step_metrics(prediction_phys, target, valid)
            row.update(
                {
                    "lead_step": forecast_index + 1,
                    "lead_minutes": (forecast_index + 1) * data.DT_MINUTES,
                    "target_time_minutes": (target_index + 1) * data.DT_MINUTES,
                    "valid_pixels": int(valid_cpu.sum()),
                }
            )
            lead_metrics.append(row)
            next_n = normalizer.normalize(prediction_phys, 0)
            next_n = torch.where(valid, next_n, torch.zeros_like(next_n)).detach()
            history = [*history[1:], next_n]

    metric_names = list(lead_metrics[0].keys())
    metric_names = [
        name
        for name in metric_names
        if name not in {"lead_step", "lead_minutes", "target_time_minutes", "valid_pixels"}
    ]
    overall = {
        name: metrics_base.mean_finite(row[name] for row in lead_metrics)
        for name in metric_names
    }
    return {
        "sequence_index": sequence_index,
        "sample_id": tile.sequence_id,
        "city": tile.scenario.city,
        "domain_name": tile.scenario.domain,
        "scenario_id": tile.scenario.scenario_id,
        "tile_id": tile.tile_id,
        "row_start": tile.row_start,
        "col_start": tile.col_start,
        "states_available": data.FRAMES_PER_SCENARIO,
        "prediction_steps": steps,
        "start_time_minutes": 30,
        "end_time_minutes": 25 + steps * data.DT_MINUTES,
        "overall": overall,
        "lead_metrics": lead_metrics,
    }


def aggregate_city(
    city: str,
    role: str,
    sequences: Sequence[Mapping[str, Any]],
    normalizer: Any,
) -> Dict[str, Any]:
    if not sequences:
        raise ValueError(f"{city}: no evaluation sequences")
    metric_names = list(sequences[0]["overall"])
    overall = {
        name: metrics_base.mean_finite(item["overall"][name] for item in sequences)
        for name in metric_names
    }
    max_leads = max(len(item["lead_metrics"]) for item in sequences)
    lead_metrics: List[Dict[str, Any]] = []
    for lead_index in range(max_leads):
        available = [
            item["lead_metrics"][lead_index]
            for item in sequences
            if lead_index < len(item["lead_metrics"])
        ]
        row: Dict[str, Any] = {
            "lead_step": lead_index + 1,
            "lead_minutes": (lead_index + 1) * data.DT_MINUTES,
            "target_time_minutes": 30 + lead_index * data.DT_MINUTES,
            "sequence_count": len(available),
        }
        row.update(
            {
                name: metrics_base.mean_finite(item[name] for item in available)
                for name in metric_names
            }
        )
        lead_metrics.append(row)
    return {
        "domain": city,
        "role": role,
        "sequence_count": len(sequences),
        "dt_minutes": data.DT_MINUTES,
        "spatial_shape": [data.CROP_SIZE, data.CROP_SIZE],
        "normalization": normalizer.state_dict(),
        "overall": overall,
        "lead_metrics": lead_metrics,
        "sequences": list(sequences),
    }


def evaluate_city(
    model: nn.Module,
    dataset: data.FloodSimBenchContext6Dataset,
    normalizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    *,
    max_steps: int = 0,
    max_sequences: int = 0,
) -> Dict[str, Any]:
    rank, world = rank_world()
    count = dataset.sequence_count
    if max_sequences > 0:
        count = min(count, int(max_sequences))
    local: List[Dict[str, Any]] = []
    for position, sequence_index in enumerate(range(rank, count, world), start=1):
        local.append(
            rollout_sequence(
                model, dataset, sequence_index, normalizer, device, max_steps=max_steps
            )
        )
        if args.eval_log_every > 0 and position % args.eval_log_every == 0:
            print(
                json.dumps(
                    {
                        "evaluation_progress": dataset.cities[0],
                        "rank": rank,
                        "local_completed": position,
                        "local_total": len(range(rank, count, world)),
                    }
                ),
                flush=True,
            )
    if is_distributed():
        gathered: List[Any] = [None for _ in range(world)]
        dist.all_gather_object(gathered, local)
        sequences = [item for shard in gathered for item in shard]
    else:
        sequences = local
    sequences.sort(key=lambda item: int(item["sequence_index"]))
    if len(sequences) != count:
        raise AssertionError(f"distributed evaluation gathered {len(sequences)} != {count}")
    return aggregate_city(dataset.cities[0], dataset.role, sequences, normalizer)


def evaluate_split(
    model: nn.Module,
    datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    normalizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    *,
    max_steps: int = 0,
    max_sequences: int = 0,
) -> Dict[str, Any]:
    results = {
        city: evaluate_city(
            model,
            dataset,
            normalizer,
            device,
            args,
            max_steps=max_steps,
            max_sequences=max_sequences,
        )
        for city, dataset in datasets.items()
    }
    macro = metrics_base.macro_average(results)
    return {"domains": results, "macro": macro}


def write_evaluation(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for city, city_result in result["domains"].items():
        metrics_base.write_evaluation(city_result, output_dir)
    (output_dir / "city_macro.json").write_text(
        json.dumps(json_safe(result["macro"]), indent=2), encoding="utf-8"
    )


def atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    normalizer: Any,
    model_config: Mapping[str, Any],
    args: argparse.Namespace,
    epoch: int,
    global_update: int,
    best_score: float,
    best_epoch: int,
) -> Dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "model_state": unwrap(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "normalization": normalizer.state_dict(),
        "model_config": dict(model_config),
        "run_config": vars(args),
        "epoch": epoch,
        "global_update": global_update,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "dataset": {
            "name": "FloodSimBench_context6_crop1024",
            "preprocessing_version": data.PREPROCESSING_VERSION,
            "train_cities": list(data.TRAIN_CITIES),
            "validation_cities": list(data.VALID_CITIES),
            "test_cities": list(data.TEST_CITIES),
            "frames_per_sequence": data.FRAMES_PER_SCENARIO,
            "forecast_samples_per_sequence": data.FORECAST_SAMPLES_PER_SEQUENCE,
            "initial_history": "observed_h5_h10_h15_h20_h25",
            "rollout": "autoregressive_h30_to_h360_67_steps",
            "crop_size": data.CROP_SIZE,
            "train_crop": "one_random_crop_per_scenario_per_epoch",
            "evaluation_grid": "stride1024_overlap0_no_remainder_completion",
            "channel_order": list(data.CHANNEL_ORDER),
            "loss": "masked_mse_only",
        },
        "checkpoint_selection": {
            "metric": "0.5*AUS_rollout_RMSE+0.5*MIA_rollout_RMSE",
            "validate_every_epochs": args.validate_every,
            "lower_is_better": True,
            "test_not_used_for_selection": True,
        },
    }


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
        "name": "FloodSimBench_context6_crop1024",
        "preprocessing_version": data.PREPROCESSING_VERSION,
        "frames_per_sequence": 72,
        "forecast_samples_per_sequence": 67,
        "crop_size": 1024,
        "loss": "masked_mse_only",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"checkpoint dataset contract {key}: {contract.get(key)!r} != {value!r}"
            )


def load_checkpoint(path: str | Path, device: torch.device, model_type: str) -> Dict[str, Any]:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device)
    validate_checkpoint(checkpoint, model_type)
    return checkpoint


def train(
    args: argparse.Namespace,
    train_dataset: data.FloodSimBenchContext6Dataset,
    valid_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    normalizer: Any,
    device: torch.device,
    rank: int,
    world: int,
) -> Path:
    loader, sampler = build_train_loader(train_dataset, args, rank, world)
    updates_per_epoch = math.ceil(len(loader) / args.accum_steps)
    model_config = model_config_from_args(args)
    model = build_model(model_config, device)
    if world > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * updates_per_epoch)
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=args.amp_initial_scale
    )
    start_epoch = 0
    global_update = 0
    best_score = float("inf")
    best_epoch = -1
    output_dir = Path(args.output_dir).expanduser().resolve()
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best_validation.pt"
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device, args.model_type)
        if dict(checkpoint["model_config"]) != model_config:
            raise RuntimeError("resume model configuration mismatch")
        unwrap(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_update = int(checkpoint["global_update"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
    if is_main():
        print(
            "TRAIN_INFO",
            json.dumps(
                {
                    "train_scenarios": train_dataset.sequence_count,
                    "forecast_samples_global": len(train_dataset),
                    "batches_per_rank": len(loader),
                    "updates_per_epoch": updates_per_epoch,
                    "effective_global_batch": args.batch_size * args.accum_steps * world,
                    "validation_every_epochs": args.validate_every,
                    "loss": "masked_mse",
                }
            ),
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs):
        train_loss, global_update, epoch_seconds = train_one_epoch(
            model,
            loader,
            sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            global_update,
        )
        validation_result: Optional[Dict[str, Any]] = None
        validation_score = float("nan")
        should_validate = (
            (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1
        )
        improved = False
        if should_validate:
            validation_result = evaluate_split(
                unwrap(model), valid_datasets, normalizer, device, args
            )
            validation_score = 0.5 * (
                validation_result["domains"]["AUS"]["overall"]["rmse"]
                + validation_result["domains"]["MIA"]["overall"]["rmse"]
            )
            improved = validation_score < best_score
            if improved:
                best_score = validation_score
                best_epoch = epoch
        if is_main():
            print(
                json.dumps(
                    json_safe(
                        {
                            "epoch": epoch,
                            "epoch_seconds": epoch_seconds,
                            "train_masked_mse": train_loss,
                            "validation_score": validation_score,
                            "validation_macro": (
                                None if validation_result is None else validation_result["macro"]
                            ),
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "global_update": global_update,
                        }
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                normalizer,
                model_config,
                args,
                epoch,
                global_update,
                best_score,
                best_epoch,
            )
            atomic_save(payload, last_path)
            if improved:
                atomic_save(payload, best_path)
                assert validation_result is not None
                write_evaluation(
                    validation_result, output_dir / "best_validation_metrics"
                )
        barrier()
    if not best_path.exists():
        raise RuntimeError("training completed without a validation-selected checkpoint")
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    barrier()
    return best_path


def test_best(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
    test_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device, args.model_type)
    normalizer = metrics_base.ChannelNormalizer.from_state_dict(
        checkpoint["normalization"]
    )
    model = build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    result = evaluate_split(model, test_datasets, normalizer, device, args)
    if is_main():
        output_dir = Path(args.output_dir).expanduser().resolve() / "test_67step_rollout"
        write_evaluation(result, output_dir)
        print(
            "TEST_RESULT",
            json.dumps(json_safe(result["macro"]), ensure_ascii=False),
            flush=True,
        )
    barrier()
    return result


class PersistenceModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :, :]


def _coordinates(dataset: data.FloodSimBenchContext6Dataset) -> List[Tuple[str, int, int]]:
    return [
        (tile.sequence_id, tile.row_start, tile.col_start)
        for tile in dataset.files
    ]


def run_smoke(
    args: argparse.Namespace,
    train_dataset: data.FloodSimBenchContext6Dataset,
    valid_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    test_datasets: Mapping[str, data.FloodSimBenchContext6Dataset],
    normalizer: Any,
    device: torch.device,
    rank: int,
    world: int,
) -> None:
    inventory = data.split_inventory(args.data_root)
    train_dataset.set_epoch(0)
    coordinate0 = train_dataset.random_coordinate(0, 0)
    coordinate1 = train_dataset.random_coordinate(0, 1)
    if coordinate0 == coordinate1:
        raise AssertionError("Train crop did not change across epochs")
    first_sample = train_dataset[0]
    last_sample = train_dataset[data.FORECAST_SAMPLES_PER_SEQUENCE - 1]
    if (
        (first_sample["row_start"], first_sample["col_start"])
        != (last_sample["row_start"], last_sample["col_start"])
    ):
        raise AssertionError("One scenario remaining samples did not share one crop")
    if (
        int(first_sample["target_time_minutes"]) != 30
        or int(last_sample["target_time_minutes"]) != 360
    ):
        raise AssertionError("Forecast sample endpoints are not h30 and h360")
    for sample in (first_sample, last_sample):
        valid = sample["mask"][0]
        observed_rain = [
            float(
                sample["x"][channel][valid].mean() * normalizer.std[1]
                + normalizer.mean[1]
            )
            for channel in range(data.RAINFALL_CONTEXT_STEPS)
        ]
        expected_rain = [
            data.rainfall_at_frame_mm_h(
                train_dataset.scenarios[0].rainfall_mm, frame_index
            )
            for frame_index in sample["rainfall_frame_indices"]
        ]
        if any(
            not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-3)
            for got, want in zip(observed_rain, expected_rain)
        ):
            raise AssertionError("Rainfall context channels/timestamps disagree")
    valid_repeat = data.FloodSimBenchContext6Dataset(
        args.data_root, data.VALID_CITIES, "valid", normalizer, seed=args.seed
    )
    combined_valid = data.FloodSimBenchContext6Dataset(
        args.data_root, data.VALID_CITIES, "valid", normalizer, seed=args.seed
    )
    if _coordinates(valid_repeat) != _coordinates(combined_valid):
        raise AssertionError("Validation grid is not deterministic")
    loader, sampler = build_train_loader(train_dataset, args, rank, world)
    sampler.set_epoch(0)
    batch = move_batch(next(iter(loader)), device)
    if tuple(batch["x"].shape[1:]) != (
        data.INPUT_CHANNELS, data.CROP_SIZE, data.CROP_SIZE
    ):
        raise AssertionError("Train batch is not [B,12,1024,1024]")
    if batch["target_time_minutes"][0] not in range(30, 361, 5):
        raise AssertionError("Forecast target time is invalid")

    dry_prediction = torch.ones((1, 1, 2, 2), device=device)
    dry_target = torch.zeros_like(dry_prediction)
    dry_mask = torch.ones_like(dry_prediction, dtype=torch.bool)
    dry_loss = masked_mse(dry_prediction, dry_target, dry_mask)
    if not math.isclose(float(dry_loss), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("Valid dry pixels are not included in masked MSE")

    model_config = model_config_from_args(args)
    model = build_model(model_config, device)
    if world > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = build_optimizer(model, args)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = 0
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(args.amp and device.type == "cuda")
    with autocast_context(amp_enabled):
        prediction = model(batch["x"])
    loss = masked_mse(prediction.float(), batch["y"].float(), batch["mask"])
    loss.backward()
    grad_l1 = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    if grad_l1 <= 0 or not math.isfinite(float(loss)):
        raise AssertionError("forward/backward smoke failed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0

    short_valid = evaluate_city(
        unwrap(model),
        valid_datasets["AUS"],
        normalizer,
        device,
        args,
        max_steps=args.smoke_rollout_steps,
        max_sequences=1,
    )
    short_test = evaluate_city(
        unwrap(model),
        test_datasets["LA"],
        normalizer,
        device,
        args,
        max_steps=args.smoke_rollout_steps,
        max_sequences=1,
    )
    # Exercise the complete 67-step autoregressive state-update path without
    # paying for 67 heavy UNet/FNO forwards.
    full_valid = evaluate_city(
        PersistenceModel().to(device),
        valid_datasets["AUS"],
        normalizer,
        device,
        args,
        max_steps=0,
        max_sequences=1,
    )
    local_memory = {
        "rank": rank,
        "device": str(device),
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak,
        "peak_increment_bytes": max(peak - allocated_before, 0),
    }
    if is_distributed():
        memory_rows: List[Any] = [None for _ in range(world)]
        dist.all_gather_object(memory_rows, local_memory)
    else:
        memory_rows = [local_memory]
    if is_main():
        first_rain, first_depth, first_target = data.forecasting_frame_indices(0)
        last_rain, last_depth, last_target = data.forecasting_frame_indices(66)
        report = {
            "status": "PASS",
            "new_dataset": data.PREPROCESSING_VERSION,
            "splits": inventory["splits"],
            "train_random_coordinates": {
                "same_scenario_epoch0": list(coordinate0),
                "same_scenario_epoch1": list(coordinate1),
                "different": coordinate0 != coordinate1,
            },
            "validation_grid_repeat_identical": True,
            "fixed_grid_stride": data.EVAL_STRIDE,
            "fixed_grid_overlap": 0,
            "padding_domains": [
                domain
                for domain, row in inventory["geometries"].items()
                if row["padding_bottom"] or row["padding_right"]
            ],
            "ordinary_remainder_policy": "discard; no append(H-1024)/append(W-1024)",
            "batch": {
                "x": list(batch["x"].shape),
                "y": list(batch["y"].shape),
                "mask": list(batch["mask"].shape),
                "dtype": str(batch["x"].dtype),
                "channel_order": list(data.CHANNEL_ORDER),
            },
            "forecast_contract": {
                "frames": 72,
                "samples": 67,
                "first": {
                    "rain_indices": list(first_rain),
                    "depth_indices": list(first_depth),
                    "target_index": first_target,
                    "target_minutes": 30,
                },
                "last": {
                    "rain_indices": list(last_rain),
                    "depth_indices": list(last_depth),
                    "target_index": last_target,
                    "target_minutes": 360,
                },
            },
            "model_type": args.model_type,
            "parameters": sum(p.numel() for p in unwrap(model).parameters()),
            "prediction_shape": list(prediction.shape),
            "masked_mse": float(loss),
            "valid_dry_pixel_test_loss": float(dry_loss),
            "gradient_l1": grad_l1,
            "gpu_memory": memory_rows,
            "short_validation_steps": len(short_valid["lead_metrics"]),
            "short_test_steps": len(short_test["lead_metrics"]),
            "full_autoregressive_validation_steps": len(full_valid["lead_metrics"]),
            "full_rollout_end_minutes": full_valid["lead_metrics"][-1][
                "target_time_minutes"
            ],
        }
        print("SMOKE_REPORT", json.dumps(json_safe(report), ensure_ascii=False), flush=True)
        print("SMOKE PASS", flush=True)
    barrier()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inspect", "smoke", "train", "test", "all"), default="all"
    )
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="UNet")
    parser.add_argument("--data-root", default=str(data.DEFAULT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--statistics-cache", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", type=metrics_base.str2bool, default=True)
    parser.add_argument("--amp", type=metrics_base.str2bool, default=True)
    parser.add_argument("--amp-initial-scale", type=float, default=128.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every-updates", type=int, default=10)
    parser.add_argument("--eval-log-every", type=int, default=10)
    parser.add_argument("--ddp-timeout-hours", type=float, default=72.0)
    parser.add_argument("--smoke-rollout-steps", type=int, default=2)
    parser.add_argument("--unet-init-features", type=int, default=32)
    parser.add_argument("--unet-grid-type", default="cartesian")
    parser.add_argument("--fno-modes-height", type=int, default=12)
    parser.add_argument("--fno-modes-width", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=20)
    parser.add_argument("--fno-grid-type", default="cartesian")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "num_workers+1": args.num_workers + 1,
        "validate_every": args.validate_every,
        "smoke_rollout_steps": args.smoke_rollout_steps,
        "unet_init_features": args.unet_init_features,
        "fno_modes_height": args.fno_modes_height,
        "fno_modes_width": args.fno_modes_width,
        "fno_width": args.fno_width,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"positive arguments required: {invalid}")
    if args.validate_every != 5:
        raise ValueError("This benchmark fixes validation/checkpoint selection every 5 epochs")
    if args.model_type == "FNO" and args.amp:
        args.amp = False
        args.amp_disabled_reason = "repository FNO FFT baseline runs in FP32"
    if not args.output_dir:
        args.output_dir = str(
            DEFAULT_OUTPUT_ROOT
            / f"FloodSimBench_context6_crop1024_{args.model_type}_MaskedMSE"
        )
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--mode test requires --checkpoint")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    distributed, rank, world, device = setup_distributed(args.ddp_timeout_hours)
    try:
        seed_everything(args.seed, args.deterministic)
        data.configure_terrain_store(args.data_root)
        statistics_cache = args.statistics_cache or None
        if distributed:
            normalizer = None
            if rank == 0:
                normalizer = data.make_channel_normalizer(
                    metrics_base.ChannelNormalizer,
                    args.data_root,
                    statistics_cache,
                )
            barrier()
            if rank != 0:
                normalizer = data.make_channel_normalizer(
                    metrics_base.ChannelNormalizer,
                    args.data_root,
                    statistics_cache,
                )
            assert normalizer is not None
        else:
            normalizer = data.make_channel_normalizer(
                metrics_base.ChannelNormalizer,
                args.data_root,
                statistics_cache,
            )
        if args.mode == "inspect":
            if is_main():
                print(
                    "DATA_INVENTORY",
                    json.dumps(data.split_inventory(args.data_root), ensure_ascii=False),
                )
            return
        train_dataset, valid_datasets, test_datasets = build_datasets(args, normalizer)
        if args.mode == "smoke":
            run_smoke(
                args,
                train_dataset,
                valid_datasets,
                test_datasets,
                normalizer,
                device,
                rank,
                world,
            )
            return
        checkpoint_path: Optional[Path] = None
        if args.mode in {"train", "all"}:
            checkpoint_path = train(
                args,
                train_dataset,
                valid_datasets,
                normalizer,
                device,
                rank,
                world,
            )
        elif args.checkpoint:
            checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if args.mode in {"test", "all"}:
            if checkpoint_path is None:
                raise RuntimeError("test requires a checkpoint")
            test_best(checkpoint_path, args, test_datasets, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
