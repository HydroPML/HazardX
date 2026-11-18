"""Install exact 21-scene trajectory sampling for the UNet/FNO entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
from gphyt.data.floodcast_scene_tile_trajectory import (
    EXPECTED_PAIRS,
    EXPECTED_SAMPLES,
    EXPECTED_SCENES,
    TRAIN_DOMAINS,
    TrajectoryEpochBatchSampler,
    UnifiedSceneTileTrajectoryDataset,
    configure_trajectory_schedule,
    plan_statistics,
    run_schedule_sanity,
    trajectory_collate,
)
from gphyt.data.floodcast_split_audit import protocol_hash


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(dest)


class TrajectoryCheckpointState:
    """Checkpoint-compatible state; schedules themselves are epoch-pure."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.completed_epoch = -1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "scene_tile_trajectory_v1",
            "seed": self.seed,
            "completed_epoch": self.completed_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "scene_tile_trajectory_v1":
            raise ValueError("checkpoint does not contain the trajectory sampler state")
        if int(state["seed"]) != self.seed:
            raise ValueError("checkpoint trajectory seed differs from --window-seed")
        self.completed_epoch = int(state["completed_epoch"])


def _trajectory_protocol(args: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    effective = int(world_size) * int(args.micro_batch_size) * int(args.accum_steps)
    return {
        "format_version": 1,
        "training_only": True,
        "train_domains": list(TRAIN_DOMAINS),
        "window_size": 128,
        "spatial_stride": 128,
        "overlap": False,
        "complete_tiles_only": True,
        "scene_counts": dict(EXPECTED_SCENES),
        "tiles_per_scene": {"australia": 16, "pakistan": 18, "uk_30m": 2},
        "states_per_scene": 288,
        "markov_pairs_per_scene": EXPECTED_PAIRS,
        "crop_trajectories_per_epoch": 21,
        "logical_samples_per_epoch": EXPECTED_SAMPLES,
        "global_deterministic_shuffle": True,
        "same_crop_for_all_scene_times": True,
        "tile_pool_without_replacement": True,
        "window_seed": int(args.window_seed),
        "micro_batch_size": int(args.micro_batch_size),
        "accumulation_steps": int(args.accum_steps),
        "world_size": int(world_size),
        "effective_batch_size": effective,
        "updates_per_epoch": math.ceil(EXPECTED_SAMPLES / effective),
        "tail_policy": "exact_short_batch_no_drop_no_duplicate",
        "validation_test_protocol": "unchanged_full_grid_128_stride96",
    }


def _broadcast_model_state(model: torch.nn.Module, src: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for value in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(value.detach(), src=src)


def _manual_allreduce_scaled_gradients(model: torch.nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _rank_sample_count(
    global_count: int,
    rank: int,
    micro_batch_size: int,
    accumulation_steps: int,
    world_size: int,
) -> int:
    capacity = int(micro_batch_size) * int(accumulation_steps)
    per_rank, remainder = divmod(int(global_count), int(world_size))
    count = per_rank + (1 if int(rank) < remainder else 0)
    if count > capacity:
        raise AssertionError("rank assignment exceeds accumulation capacity")
    return count


def _rank_microbatch_count(local_samples: int, micro_batch_size: int) -> int:
    return math.ceil(int(local_samples) / int(micro_batch_size)) if local_samples else 0


def _build_train_data(
    args: argparse.Namespace,
    specs: Mapping[str, baseline.DomainSpec],
    normalizer: baseline.ChannelNormalizer,
    rank: int,
    world_size: int,
) -> tuple[UnifiedSceneTileTrajectoryDataset, TrajectoryEpochBatchSampler, DataLoader]:
    configure_trajectory_schedule(
        seed=args.window_seed,
        micro_batch_size=args.micro_batch_size,
        accumulation_steps=args.accum_steps,
        world_size=world_size,
    )
    dataset = UnifiedSceneTileTrajectoryDataset(
        specs, normalizer=normalizer, noise_std=args.noise_std
    )
    sampler = TrajectoryEpochBatchSampler(dataset, rank=rank, world_size=world_size)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=trajectory_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=baseline.seed_worker,
    )
    return dataset, sampler, loader


def train_scene_tile_trajectories(
    args: argparse.Namespace,
    specs: Mapping[str, baseline.DomainSpec],
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Path:
    """The reference trainer with only its train loader/schedule replaced."""

    output_dir = baseline.canonical_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_checkpoint = baseline.load_checkpoint_file(args.resume, device)
        normalizer = baseline.ChannelNormalizer.from_state_dict(
            resume_checkpoint["normalization"]
        )
        if normalizer.mode != "train_standard":
            raise ValueError("trajectory training requires train_standard normalization")
        if resume_checkpoint.get("run_config", {}).get("loss") != "mse":
            raise ValueError("resume checkpoint loss is incompatible")
        model_config = dict(resume_checkpoint["model_config"])
    else:
        normalizer = baseline.compute_training_normalizer(
            {name: specs[name] for name in TRAIN_DOMAINS}
        )
        model_config = baseline.model_config_from_args(args)

    dataset, train_sampler, loader = _build_train_data(
        args, specs, normalizer, rank, world_size
    )
    initial_plan = train_sampler.set_epoch(0)
    stats = run_schedule_sanity(dataset)
    args.updates_per_epoch = int(initial_plan.updates)
    args.auto_updates_per_epoch = True
    args.logical_train_samples_per_epoch = EXPECTED_SAMPLES
    args.domain_loader_steps_per_rank = {"unified_trajectory": len(loader)}
    if baseline.is_main_process(rank):
        report_dir = output_dir / "data_protocol"
        report_dir.mkdir(parents=True, exist_ok=True)
        with (report_dir / "trajectory_sampler_sanity.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print("TRAJECTORY_SAMPLER_SANITY", json.dumps(stats, ensure_ascii=False))
        print("normalization", json.dumps(baseline.json_safe(normalizer.state_dict())))

    eval_datasets = baseline.build_eval_datasets(
        specs, normalizer, ("australia_valid", "pakistan_valid")
    )
    model = baseline.build_model(model_config, device)
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = baseline.make_scheduler(optimizer, args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=args.amp_initial_scale
    )
    sampler_state = TrajectoryCheckpointState(args.window_seed)

    start_epoch = 0
    global_update = 0
    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0
    if resume_checkpoint is not None:
        if dict(resume_checkpoint["model_config"]) != model_config:
            raise ValueError("resume model configuration mismatch")
        baseline.unwrap_model(model).load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state", {}))
        sampler_state.load_state_dict(resume_checkpoint["task_sampler_state"])
        baseline.restore_rng_state(resume_checkpoint["rng_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if sampler_state.completed_epoch != start_epoch - 1:
            raise RuntimeError("checkpoint epoch and trajectory sampler state disagree")
        global_update = int(resume_checkpoint["global_update"])
        best_score = float(resume_checkpoint["best_score"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        bad_epochs = int(resume_checkpoint["bad_epochs"])

    best_path = output_dir / "best_validation.pt"
    last_path = output_dir / "last.pt"
    effective = int(world_size) * int(args.micro_batch_size) * int(args.accum_steps)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        plan = train_sampler.set_epoch(epoch)
        if plan.updates != args.updates_per_epoch:
            raise AssertionError("optimizer update count changed between epochs")
        iterator = iter(loader)
        loss_sums = {name: 0.0 for name in TRAIN_DOMAINS}
        loss_counts = {name: 0 for name in TRAIN_DOMAINS}
        sampled_counts = {name: 0 for name in TRAIN_DOMAINS}
        epoch_start = time.perf_counter()

        for update_in_epoch in range(args.updates_per_epoch):
            global_start = update_in_epoch * effective
            global_count = min(effective, EXPECTED_SAMPLES - global_start)
            local_count = _rank_sample_count(
                global_count,
                rank,
                args.micro_batch_size,
                args.accum_steps,
                world_size,
            )
            local_microbatches = _rank_microbatch_count(
                local_count, args.micro_batch_size
            )
            exact_short_tail = global_count != effective
            optimizer.zero_grad(set_to_none=True)
            if exact_short_tail and isinstance(model, DDP):
                # Unequal 2/1 samples on the final two-rank update cannot use
                # DDP's per-rank averaging or per-forward buffer broadcasts.
                model.broadcast_buffers = False

            for micro_step in range(local_microbatches):
                batch = baseline.move_batch(next(iterator), device)
                batch_size = int(batch["x"].shape[0])
                if exact_short_tail:
                    sync_context = model.no_sync() if isinstance(model, DDP) else contextlib.nullcontext()
                else:
                    should_sync = micro_step == local_microbatches - 1
                    sync_context = (
                        contextlib.nullcontext()
                        if should_sync or not isinstance(model, DDP)
                        else model.no_sync()
                    )
                with sync_context:
                    with baseline.autocast_context(amp_enabled):
                        dh = model(x=batch["x"], static=batch["static"], metadata=batch)
                    prediction = baseline.constrain_nonnegative_water_state(
                        batch["x"][:, :1].float() + dh.float(), normalizer
                    )
                    with baseline.autocast_context(False):
                        raw_loss = baseline.masked_loss(
                            prediction,
                            batch["y"].float(),
                            batch["mask"],
                            kind="mse",
                        )
                        if exact_short_tail:
                            scaled_loss = raw_loss * (batch_size / global_count)
                        else:
                            scaled_loss = raw_loss / args.accum_steps
                    scaler.scale(scaled_loss).backward()

                domains = list(batch["domain"])
                for domain in set(domains):
                    indices = [i for i, value in enumerate(domains) if value == domain]
                    domain_loss = baseline.masked_loss(
                        prediction[indices],
                        batch["y"].float()[indices],
                        batch["mask"][indices],
                        kind="mse",
                    )
                    loss_sums[domain] += float(domain_loss.detach().cpu()) * len(indices)
                    loss_counts[domain] += len(indices)
                    sampled_counts[domain] += len(indices)

            if exact_short_tail and distributed:
                _manual_allreduce_scaled_gradients(model)
                _broadcast_model_state(baseline.unwrap_model(model), src=0)
                if isinstance(model, DDP):
                    model.broadcast_buffers = True

            gradient_finite = True
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                gradient_finite = bool(torch.isfinite(gradient_norm))
                if not gradient_finite and not amp_enabled:
                    raise FloatingPointError("gradient norm is NaN/Inf")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = gradient_finite and (
                (not amp_enabled) or scaler.get_scale() >= old_scale
            )
            if step_succeeded:
                global_update += 1
                if scheduler is not None:
                    scheduler.step()
            if (
                baseline.is_main_process(rank)
                and args.log_every_updates > 0
                and (update_in_epoch + 1) % args.log_every_updates == 0
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "update_in_epoch": update_in_epoch + 1,
                            "global_optimizer_updates": global_update,
                            "logical_samples_consumed": min(
                                (update_in_epoch + 1) * effective, EXPECTED_SAMPLES
                            ),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                )

        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("trajectory loader emitted extra micro-batches")

        epoch_seconds = time.perf_counter() - epoch_start
        reduced_losses = {
            name: baseline.reduce_sum(loss_sums[name], device)
            / max(baseline.reduce_sum(loss_counts[name], device), 1.0)
            for name in TRAIN_DOMAINS
        }
        reduced_sampled = {
            name: int(baseline.reduce_sum(sampled_counts[name], device))
            for name in TRAIN_DOMAINS
        }
        if sum(reduced_sampled.values()) != EXPECTED_SAMPLES:
            raise AssertionError(
                f"epoch consumed {sum(reduced_sampled.values())} real samples, "
                f"expected {EXPECTED_SAMPLES}"
            )

        validation_score = float("nan")
        validation_result: Optional[Dict[str, Any]] = None
        should_validate = (
            (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1
        )
        if should_validate and baseline.is_main_process(rank):
            validation_score, validation_result = baseline.run_source_validation(
                model, eval_datasets, normalizer, device, args
            )
            improved = validation_score < best_score
            if improved:
                best_score = validation_score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(
                json.dumps(
                    baseline.json_safe(
                        {
                            "epoch": epoch,
                            "epoch_seconds": epoch_seconds,
                            "logical_training_samples": EXPECTED_SAMPLES,
                            "train_loss_by_domain": reduced_losses,
                            "sampled_examples_by_domain_all_ranks": reduced_sampled,
                            "validation_score_macro_rmse": validation_score,
                            "validation_macro": validation_result["macro"],
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "global_update": global_update,
                        }
                    ),
                    ensure_ascii=False,
                )
            )
        else:
            improved = False

        state = baseline.broadcast_object(
            {
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "improved": improved,
            }
            if baseline.is_main_process(rank)
            else None
        )
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        improved = bool(state["improved"])
        sampler_state.completed_epoch = epoch

        if baseline.is_main_process(rank):
            payload = baseline.checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_update=global_update,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                task_sampler=sampler_state,
                normalizer=normalizer,
                model_config=model_config,
                args=args,
                specs=specs,
            )
            trajectory_protocol = _trajectory_protocol(args, world_size)
            payload["trajectory_training_protocol"] = trajectory_protocol
            window_protocol = dict(payload.get("window_protocol", {}))
            window_protocol.update(
                training_crop="fixed_nonoverlap_scene_trajectory",
                training_stride=128,
                train_windows_per_epoch=EXPECTED_SAMPLES,
                crop_trajectories_per_epoch=21,
                same_crop_for_all_scene_times=True,
                trajectory_training_protocol="trajectory_training_protocol",
            )
            window_protocol.pop("sha256", None)
            window_protocol["sha256"] = protocol_hash(window_protocol)
            payload["window_protocol"] = window_protocol
            baseline.atomic_torch_save(payload, last_path)
            if improved:
                baseline.atomic_torch_save(payload, best_path)
                if validation_result is not None:
                    validation_dir = output_dir / "best_validation_metrics"
                    for result in validation_result["domains"].values():
                        baseline.write_evaluation(result, validation_dir)
                    with (validation_dir / "source_macro.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            baseline.json_safe(validation_result["macro"]),
                            handle,
                            indent=2,
                        )
        baseline.distributed_barrier()
        stop = (
            args.early_stopping > 0
            and should_validate
            and bad_epochs >= args.early_stopping
        )
        stop = bool(
            baseline.broadcast_object(
                stop if baseline.is_main_process(rank) else None
            )
        )
        if stop:
            break

    if not best_path.exists() and baseline.is_main_process(rank):
        if last_path.exists():
            warnings.warn("no best checkpoint was selected; copying last checkpoint")
            checkpoint = torch.load(last_path, map_location="cpu")
            baseline.atomic_torch_save(checkpoint, best_path)
        else:
            raise RuntimeError("training finished without writing a checkpoint")
    baseline.distributed_barrier()
    return best_path


def install_scene_tile_training(namespace: MutableMapping[str, Any]) -> None:
    """Apply after the existing 128/96 validation/test protocol installer."""

    previous_build_parser = baseline.build_parser
    previous_validate_args = baseline.validate_args
    previous_smoke = baseline.run_smoke_tests
    previous_load_checkpoint = baseline.load_checkpoint_file
    with_optimizer = namespace["_with_optimizer"]

    def build_parser() -> argparse.ArgumentParser:
        parser = previous_build_parser()
        _action(parser, "train_windows_per_epoch").default = 0
        _action(parser, "max_optimizer_steps").default = 0
        return parser

    def validate_args(args: argparse.Namespace) -> None:
        previous_validate_args(args)
        if args.train_windows_per_epoch != 0 or args.max_optimizer_steps != 0:
            raise ValueError(
                "trajectory training defines exactly 6027 logical samples/epoch; "
                "--train-windows-per-epoch and --max-optimizer-steps must both be 0"
            )
        if args.window_size != 128 or args.window_stride != 96:
            raise ValueError("train is 128/128; unchanged validation/test are 128/96")

    def train(args, specs, device, distributed, rank, world_size):
        return with_optimizer(
            train_scene_tile_trajectories,
            args,
            args,
            specs,
            device,
            distributed,
            rank,
            world_size,
        )

    def smoke(args, specs, device, rank, world_size):
        configure_trajectory_schedule(
            seed=args.window_seed,
            micro_batch_size=1,
            accumulation_steps=1,
            world_size=1,
        )
        normalizer = baseline.compute_training_normalizer(
            {name: specs[name] for name in TRAIN_DOMAINS}
        )
        dataset = UnifiedSceneTileTrajectoryDataset(
            specs, normalizer=normalizer, noise_std=0.0
        )
        stats = run_schedule_sanity(dataset)
        print("TRAJECTORY_SAMPLER_SANITY", json.dumps(stats, ensure_ascii=False))
        sampler = TrajectoryEpochBatchSampler(dataset, rank=0, world_size=1)
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=trajectory_collate,
            num_workers=0,
        )
        batch = next(iter(loader))
        if tuple(batch["x"].shape) != (1, 2, 128, 128):
            raise AssertionError(f"trajectory smoke input shape {batch['x'].shape}")
        if tuple(batch["y"].shape) != (1, 1, 128, 128):
            raise AssertionError(f"trajectory smoke target shape {batch['y'].shape}")
        print(
            "TRAJECTORY_REAL_BATCH_PASS",
            json.dumps(
                {
                    "domain": batch["domain"],
                    "sample_id": batch["sample_id"],
                    "time_index": batch["time_index"],
                    "tile_id": batch["tile_id"],
                    "crop_top": batch["crop_top"],
                    "crop_left": batch["crop_left"],
                }
            ),
        )
        return previous_smoke(args, specs, device, rank, world_size)

    def load_checkpoint(path: Any, device: torch.device) -> Dict[str, Any]:
        checkpoint = previous_load_checkpoint(path, device)
        protocol = checkpoint.get("trajectory_training_protocol")
        if not isinstance(protocol, Mapping):
            raise RuntimeError("checkpoint lacks trajectory_training_protocol")
        if protocol.get("logical_samples_per_epoch") != EXPECTED_SAMPLES:
            raise RuntimeError("checkpoint trajectory sample count mismatch")
        if protocol.get("tail_policy") != "exact_short_batch_no_drop_no_duplicate":
            raise RuntimeError("checkpoint trajectory tail policy mismatch")
        return checkpoint

    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.train = train
    baseline.run_smoke_tests = smoke
    baseline.load_checkpoint_file = load_checkpoint

