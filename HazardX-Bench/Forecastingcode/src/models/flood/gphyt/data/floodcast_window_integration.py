"""Install the isolated 128/96 protocol into the seven-baseline lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import torch

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
from gphyt.data.floodcast_mozambique_ood import (
    build_mozambique_protocol_datasets,
    run_mozambique_day_test,
)
from gphyt.data.floodcast_random_crop_dataset import (
    DeterministicWindowBatchSampler,
    RandomCropDemFeatureFloodDomainDataset,
    configure_training_windows,
    install_base_collate,
    windowed_homogeneous_collate,
)
from gphyt.data.floodcast_split_audit import (
    TRAIN_DOMAINS,
    make_three_domain_specs,
    window_protocol,
)
from gphyt.data.floodcast_tiled_rollout import (
    configure_tiled_rollout,
    tiled_rollout_sequence,
)
from gphyt.data.floodcast_window_utils import tile_starts


_INSTALLED = False


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(dest)


def _configure_budget(args: argparse.Namespace, world_size: int) -> Optional[int]:
    batch_sizes = baseline.parse_name_values(
        args.domain_batch_sizes,
        TRAIN_DOMAINS,
        int,
        args.micro_batch_size,
    )
    if set(batch_sizes.values()) != {int(args.micro_batch_size)}:
        raise ValueError(
            "128-window benchmark requires the same micro-batch size in all "
            f"three domains, got {batch_sizes}"
        )
    batches_per_rank: Optional[int] = None
    if args.train_windows_per_epoch > 0:
        effective_batch = (
            int(world_size)
            * int(args.micro_batch_size)
            * int(args.accum_steps)
        )
        batches_per_rank = math.ceil(
            int(args.train_windows_per_epoch) / effective_batch
        )
        actual_windows = batches_per_rank * effective_batch
        args.actual_train_windows_per_epoch = actual_windows
        args.effective_global_batch = effective_batch
        if args.max_optimizer_steps > 0:
            if args.max_optimizer_steps % batches_per_rank:
                raise ValueError(
                    "--max-optimizer-steps must be divisible by the derived "
                    f"updates_per_epoch={batches_per_rank}; choose a compatible "
                    "budget so the reference epoch loop cannot overshoot"
                )
            args.epochs = args.max_optimizer_steps // batches_per_rank
    elif args.max_optimizer_steps > 0:
        raise ValueError(
            "--max-optimizer-steps requires a positive "
            "--train-windows-per-epoch"
        )
    configure_training_windows(
        window_size=args.window_size,
        window_seed=args.window_seed,
        batches_per_rank=batches_per_rank,
    )
    if int(args.epochs) % 5:
        raise ValueError(
            "the fixed full-grid validation cadence requires epochs to be "
            f"divisible by 5, got {args.epochs}"
        )
    return batches_per_rank


def _run_window_smoke(
    args: argparse.Namespace,
    specs: Mapping[str, baseline.DomainSpec],
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    if rank != 0 or world_size != 1:
        raise RuntimeError("window smoke test must run as one rank")
    configure_training_windows(
        window_size=args.window_size,
        window_seed=args.window_seed,
        batches_per_rank=2,
    )
    normalizer = baseline.compute_training_normalizer(
        {name: specs[name] for name in TRAIN_DOMAINS}
    )
    if tuple(normalizer.provenance.get("domains", ())) != TRAIN_DOMAINS:
        raise AssertionError("normalizer provenance is not the three-domain split")
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.num_workers = 0
    smoke_args.noise_std = 0.0
    batch_sizes = {name: 1 for name in TRAIN_DOMAINS}
    datasets, loaders, _ = baseline.build_train_loaders(
        specs, normalizer, batch_sizes, smoke_args, rank, world_size
    )
    batches: Dict[str, Dict[str, Any]] = {}
    coordinates: Dict[str, Any] = {}
    for name in TRAIN_DOMAINS:
        batch = next(iter(loaders[name]))
        if tuple(batch["x"].shape) != (1, 2, 128, 128):
            raise AssertionError(f"{name}: unexpected x shape {batch['x'].shape}")
        if tuple(batch["y"].shape) != (1, 1, 128, 128):
            raise AssertionError(f"{name}: unexpected y shape {batch['y'].shape}")
        batches[name] = batch
        coordinates[name] = {
            "sample_id": batch["sample_id"],
            "time_index": batch["time_index"],
            "top": batch["crop_top"],
            "left": batch["crop_left"],
        }
        first = datasets[name][(0, 7, 11)]
        second = datasets[name][(0, 7, 11)]
        if (first["crop_top"], first["crop_left"]) != (
            second["crop_top"],
            second["crop_left"],
        ) or not torch.equal(first["x"], second["x"]):
            raise AssertionError(f"{name}: deterministic crop reproduction failed")

    model = baseline.build_model(baseline.model_config_from_args(args), device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    losses: Dict[str, float] = {}
    for name in TRAIN_DOMAINS:
        batch = baseline.move_batch(batches[name], device)
        with baseline.autocast_context(amp_enabled):
            delta = model(x=batch["x"], static=batch["static"], metadata=batch)
        prediction = baseline.constrain_nonnegative_water_state(
            batch["x"][:, :1].float() + delta.float(), normalizer
        )
        with baseline.autocast_context(False):
            loss = baseline.masked_loss(
                prediction, batch["y"].float(), batch["mask"], kind="mse"
            ) / len(TRAIN_DOMAINS)
        loss.backward()
        losses[name] = float(loss.detach().cpu()) * len(TRAIN_DOMAINS)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
        raise FloatingPointError("smoke gradients are absent or non-finite")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_step_seconds = time.perf_counter() - started

    eval_dataset = baseline.build_eval_datasets(
        specs, normalizer, ("australia_valid",)
    )["australia_valid"]
    result = baseline.evaluate_domain(
        model=model,
        dataset=eval_dataset,
        normalizer=normalizer,
        device=device,
        amp=False,
        max_steps=max(4, int(args.smoke_rollout_steps)),
        max_sequences=1,
    )
    if len(result["lead_metrics"]) < 4:
        raise AssertionError("smoke rollout did not complete four steps")
    expected_starts = [0, 96, 192, 288, 384, 408]
    if tile_starts(536, 128, 96) != expected_starts:
        raise AssertionError("536-pixel stride-96 tile coordinates changed")
    first_sequence = result["sequences"][0]
    first_tiling = first_sequence["tiling"]["steps"][0]
    if int(first_tiling["tile_count"]) != 36:
        raise AssertionError("536x536 validation must use 36 tiles")
    tile_step_seconds = [
        float(step["inference_seconds"])
        for step in first_sequence["tiling"]["steps"]
    ]

    source_dataset = baseline.build_eval_datasets(
        specs, normalizer, ("australia_test",)
    )["australia_test"]
    source_result = baseline.evaluate_domain(
        model=model,
        dataset=source_dataset,
        normalizer=normalizer,
        device=device,
        amp=False,
        max_steps=4,
        max_sequences=1,
    )
    if len(source_result["lead_metrics"]) != 4:
        raise AssertionError("source-test smoke did not complete four steps")

    ood_summary: Dict[str, Any] = {"enabled": False}
    if args.smoke_ood_day_horizons:
        smoke_ood_steps = max(1, int(args.ood_debug_max_steps))
        ood_datasets = build_mozambique_protocol_datasets(
            args, dict(specs), normalizer
        )
        ood_summary = {"enabled": True, "debug_steps": smoke_ood_steps}
        for protocol, expected_count in (("1day", 6), ("2day", 3)):
            ood_result = baseline.evaluate_domain(
                model=model,
                dataset=ood_datasets[protocol],
                normalizer=normalizer,
                device=device,
                amp=False,
                max_steps=smoke_ood_steps,
                max_sequences=0,
            )
            if ood_result["sequence_count"] != expected_count:
                raise AssertionError(
                    f"{protocol} OOD smoke sequence-count mismatch"
                )
            if len(ood_result["lead_metrics"]) != smoke_ood_steps:
                raise AssertionError(f"{protocol} OOD smoke horizon mismatch")
            ood_summary[protocol] = {
                "states_per_trajectory": (288 if protocol == "1day" else 576),
                "prediction_steps_tested": smoke_ood_steps,
                "sequence_count": ood_result["sequence_count"],
                "tile_count_per_step": ood_result["sequences"][0]["tiling"][
                    "steps"
                ][0]["tile_count"],
            }

    report = {
        "model_type": args.model_type,
        "train_domains": list(TRAIN_DOMAINS),
        "crop_coordinates": coordinates,
        "input_shape": [1, 3, 128, 128],
        "output_shape": [1, 1, 128, 128],
        "losses": losses,
        "optimizer_step": "pass",
        "train_step_seconds": train_step_seconds,
        "rollout_steps": len(result["lead_metrics"]),
        "source_test_rollout_steps": len(source_result["lead_metrics"]),
        "mozambique_ood": ood_summary,
        "validation_grid": [536, 536],
        "tiles_per_step": 36,
        "tile_step_seconds": tile_step_seconds,
        "mean_tile_step_seconds": sum(tile_step_seconds) / len(tile_step_seconds),
        "tile_starts": expected_starts,
        "strict_full_grid_state_feedback": True,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    output = Path(args.output_dir).expanduser().resolve() / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.model_type}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("WINDOW_SMOKE_PASS", json.dumps(report, ensure_ascii=False))


def install_window_protocol(namespace: MutableMapping[str, Any]) -> None:
    """Replace only seven-baseline process-local hooks, never reference files."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    base_build_parser = baseline.build_parser
    base_validate_args = baseline.validate_args
    base_train = baseline.train
    base_make_specs = baseline.make_domain_specs
    base_checkpoint_payload = baseline.checkpoint_payload
    base_load_checkpoint = baseline.load_checkpoint_file
    base_collate = baseline.homogeneous_collate
    with_optimizer = namespace["_with_optimizer"]

    baseline.SOURCE_TRAIN_DOMAINS = TRAIN_DOMAINS
    dem_common.SOURCE_REGIONS = TRAIN_DOMAINS
    original_dem_metadata = dem_common.DemFeatureStore.metadata

    def three_domain_dem_metadata(store):
        metadata = original_dem_metadata(store)
        metadata["source_domain_weights"] = [
            1.0 / len(TRAIN_DOMAINS) for _ in TRAIN_DOMAINS
        ]
        return metadata

    dem_common.DemFeatureStore.metadata = three_domain_dem_metadata

    def build_parser() -> argparse.ArgumentParser:
        parser = base_build_parser()
        _action(parser, "model_runtime_height").default = 128
        _action(parser, "model_runtime_width").default = 128
        _action(parser, "domain_batch_sizes").default = (
            "australia=1,pakistan=1,uk_30m=1"
        )
        _action(parser, "task_weights").default = (
            "australia=1,pakistan=1,uk_30m=1"
        )
        _action(parser, "task_sampling").default = "uniform"
        _action(parser, "validate_every").default = 5
        # Zero means rollout to the last state. A 288-state one-day event
        # therefore produces all 287 available autoregressive forecast leads.
        _action(parser, "validation_max_steps").default = 0
        _action(parser, "eval_max_steps").default = 0
        _action(parser, "early_stopping").default = 20
        _action(parser, "smoke_rollout_steps").default = 4
        mozambique_split = _action(parser, "mozambique_split")
        mozambique_split.choices = ("1day", "2day", "both")
        mozambique_split.default = "both"
        parser.add_argument("--window-size", type=int, default=128)
        parser.add_argument("--window-stride", type=int, default=96)
        parser.add_argument("--window-seed", type=int, default=42)
        parser.add_argument("--train-windows-per-epoch", type=int, default=0)
        parser.add_argument("--max-optimizer-steps", type=int, default=0)
        parser.add_argument("--tile-batch-size", type=int, default=1)
        parser.add_argument("--ood-debug-max-steps", type=int, default=0)
        parser.add_argument("--window-num-workers", type=int, default=None)
        parser.add_argument(
            "--tiled-validation", type=baseline.str2bool, default=True
        )
        parser.add_argument(
            "--tiled-rollout", type=baseline.str2bool, default=True
        )
        parser.add_argument("--exclude-train-domain", default="uk")
        parser.add_argument(
            "--smoke-ood-day-horizons", type=baseline.str2bool, default=False
        )
        return parser

    def validate_args(args: argparse.Namespace) -> None:
        if args.window_num_workers is not None:
            args.num_workers = int(args.window_num_workers)
        base_validate_args(args)
        if (args.window_size, args.window_stride) != (128, 96):
            raise ValueError("the main benchmark fixes window_size=128 and stride=96")
        if (args.model_runtime_height, args.model_runtime_width) != (128, 128):
            raise ValueError("all seven model instances must be built for 128x128")
        if args.exclude_train_domain != "uk":
            raise ValueError("only exact internal domain key 'uk' may be excluded")
        if args.train_windows_per_epoch < 0 or args.max_optimizer_steps < 0:
            raise ValueError("window/update budgets cannot be negative")
        if args.validate_every != 5:
            raise ValueError("formal full-grid validation must run every 5 epochs")
        if args.early_stopping != 20:
            raise ValueError("early-stopping patience must be 20 validation checks")
        if args.ood_debug_max_steps < 0:
            raise ValueError("OOD debug max steps cannot be negative")
        if args.tile_batch_size < 1 or args.num_workers < 0:
            raise ValueError("tile batch must be positive and workers nonnegative")
        if args.task_sampling != "uniform":
            raise ValueError("three training domains must be sampled uniformly")
        if args.validation_max_steps != 0:
            raise ValueError(
                "formal validation must use the complete sequence "
                "(--validation-max-steps 0)"
            )
        if args.eval_max_steps != 0:
            raise ValueError(
                "formal source-test must use the complete sequence "
                "(--eval-max-steps 0)"
            )
        if not args.tiled_validation or not args.tiled_rollout:
            raise ValueError("validation and test must both use tiled full-grid inference")
        configure_tiled_rollout(
            window_size=args.window_size,
            stride=args.window_stride,
            tile_batch_size=args.tile_batch_size,
        )

    def make_specs(args: argparse.Namespace) -> Dict[str, baseline.DomainSpec]:
        return make_three_domain_specs(args, base_make_specs)

    def train(args: argparse.Namespace, specs: Any, device: Any, distributed: Any, rank: int, world_size: int):
        batches = _configure_budget(args, world_size)
        print(
            "WINDOW_TRAIN_BUDGET",
            json.dumps(
                {
                    "requested_windows_per_epoch": args.train_windows_per_epoch,
                    "actual_windows_per_epoch": getattr(
                        args, "actual_train_windows_per_epoch", None
                    ),
                    "updates_per_epoch": batches,
                    "max_optimizer_steps": args.max_optimizer_steps,
                    "epochs": args.epochs,
                    "world_size": world_size,
                }
            ),
        )
        return base_train(args, specs, device, distributed, rank, world_size)

    def smoke(args: argparse.Namespace, specs: Any, device: Any, rank: int, world_size: int):
        return with_optimizer(
            _run_window_smoke, args, args, specs, device, rank, world_size
        )

    def checkpoint_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        payload = base_checkpoint_payload(*args, **kwargs)
        run_args = kwargs.get("args")
        if run_args is None:
            raise RuntimeError("checkpoint hook requires keyword args namespace")
        payload["window_protocol"] = window_protocol(run_args)
        return payload

    def load_checkpoint(path: Any, device: torch.device) -> Dict[str, Any]:
        checkpoint = base_load_checkpoint(path, device)
        protocol = checkpoint.get("window_protocol")
        if not isinstance(protocol, Mapping):
            raise RuntimeError("checkpoint predates the audited 128/96 window protocol")
        if protocol.get("train_domains") != list(TRAIN_DOMAINS):
            raise RuntimeError("checkpoint training-domain protocol mismatch")
        if (protocol.get("window_size"), protocol.get("window_stride")) != (
            128,
            96,
        ):
            raise RuntimeError("checkpoint window geometry mismatch")
        if protocol.get("formal_validation_every_epochs") != 5:
            raise RuntimeError("checkpoint formal-validation cadence mismatch")
        if protocol.get("early_stopping_patience_checks") != 20:
            raise RuntimeError("checkpoint early-stopping patience mismatch")
        ood_protocol = protocol.get("mozambique_ood_protocol", {})
        if ood_protocol.get("source_key") != "mozambique_576":
            raise RuntimeError("checkpoint Mozambique OOD protocol mismatch")
        if protocol.get("formal_validation_horizon_steps") != 0:
            raise RuntimeError("checkpoint formal-validation horizon mismatch")
        return checkpoint

    install_base_collate(base_collate)
    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.make_domain_specs = make_specs
    baseline.FloodDomainDataset = RandomCropDemFeatureFloodDomainDataset
    baseline.SequenceLocalBatchSampler = DeterministicWindowBatchSampler
    baseline.homogeneous_collate = windowed_homogeneous_collate
    baseline.rollout_sequence = tiled_rollout_sequence
    baseline.run_mozambique_test = run_mozambique_day_test
    baseline.train = train
    baseline.run_smoke_tests = smoke
    baseline.checkpoint_payload = checkpoint_payload
    baseline.load_checkpoint_file = load_checkpoint
