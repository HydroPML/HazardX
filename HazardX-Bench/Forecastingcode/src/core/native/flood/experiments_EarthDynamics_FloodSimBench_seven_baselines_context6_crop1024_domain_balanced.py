"""Seven native-1024 FloodSimBench domain-balanced context-6 baselines.

Data, normalization, Masked MSE, deterministic domain queues, rollout,
metrics, checkpoint selection and DDP sample partitioning are delegated to the
existing FloodSimBench domain-balanced experiment.  Only model I/O,
architecture-specific optimization and audit hooks are installed here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import experiments_EarthDynamics_FloodSimBench_two_baselines_crop1024_domain_balanced as domain_balanced
from gphyt.model.ddp_pretrained_report_scene_trajectory import (
    build_model_with_ddp_safe_report,
    collective_pretrained_report_audit,
)
from gphyt.model.model_registry_seven_floodsimbench_context12_crop1024 import (
    BASELINE_NAMES,
    PT_FT_NAMES,
    build_floodsimbench_baseline,
    model_metadata,
)


reference = domain_balanced.reference
data = domain_balanced.data

EXPERIMENT_NAME = "FloodSimBench_Seven_Context6_Crop1024_DomainBalanced"
DATASET_NAME = domain_balanced.DATASET_NAME
DEFAULT_OUTPUT_ROOT = Path("results/FloodSimBench_context6_crop1024_domain_balanced_7baselines")

_REFERENCE_BUILD_PARSER = reference.build_parser
_REFERENCE_VALIDATE_ARGS = domain_balanced._reference_validate_args
_REFERENCE_TRAIN = reference.train
_REFERENCE_CHECKPOINT_PAYLOAD = domain_balanced.checkpoint_payload
_REAL_COSINE = torch.optim.lr_scheduler.CosineAnnealingLR
_LAST_MODEL: Optional[nn.Module] = None


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _instrument_first_training_step(model: nn.Module, device: torch.device) -> None:
    """Record the first real forward, backward and optimizer step without changing math."""
    state: Dict[str, Any] = {
        "forward_seconds": None,
        "backward_seconds": None,
        "optimizer_step_seconds": None,
        "backward_start": None,
    }
    model._floodsimbench_timing_audit = state
    original_forward = model.forward

    def timed_forward(*args: Any, **kwargs: Any):
        if state["forward_seconds"] is not None:
            return original_forward(*args, **kwargs)
        _sync_cuda(device)
        start = time.perf_counter()
        output = original_forward(*args, **kwargs)
        _sync_cuda(device)
        state["forward_seconds"] = time.perf_counter() - start
        if torch.is_tensor(output) and output.requires_grad:
            def mark_backward_start(gradient: torch.Tensor) -> torch.Tensor:
                _sync_cuda(device)
                state["backward_start"] = time.perf_counter()
                return gradient

            output.register_hook(mark_backward_start)
        return output

    model.forward = timed_forward


def _gradient_depth_audit(model: nn.Module) -> Dict[str, Any]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise AssertionError("model has no trainable parameters")
    rows: Dict[str, Any] = {}
    boundaries = (0, len(named) // 3, (2 * len(named)) // 3, len(named))
    for label, begin, end in zip(("shallow", "middle", "deep"), boundaries[:-1], boundaries[1:]):
        candidates = named[begin:end] or named[begin:begin + 1]
        finite_nonzero = []
        for name, parameter in candidates:
            gradient = parameter.grad
            if gradient is not None and torch.isfinite(gradient).all() and bool(torch.count_nonzero(gradient)):
                finite_nonzero.append(name)
        if not finite_nonzero:
            raise AssertionError(f"{label} parameter third has no finite non-zero gradient")
        rows[label] = finite_nonzero[0]
    rows["trainable_parameter_tensors"] = len(named)
    rows["gradient_parameter_tensors"] = sum(parameter.grad is not None for _, parameter in named)
    return rows


OPTIMIZATION: Dict[str, Dict[str, Any]] = {
    "ConvLSTM-Scratch": {
        "optimizer": "Adam",
        "lr": 1.0e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "onecycle",
        "grad_clip": 0.0,
    },
    "EarthFormer-Scratch": {
        "optimizer": "AdamW",
        "lr": 3.0e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "warmup_cosine_20",
        "grad_clip": 1.0,
    },
    "PDE-T-SC-S-PT-FT": {
        "optimizer": "AdamW",
        "lr": 4.0e-5,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler": "none",
        "grad_clip": 0.0,
    },
    "DPOT-S-PT-FT": {
        "optimizer": "Adam",
        "lr": 3.0e-5,
        "weight_decay": 0.0,
        "betas": (0.9, 0.9),
        "scheduler": "dpot_cycle",
        "grad_clip": 1.0,
    },
    "Poseidon-T-PT-FT": {
        "optimizer": "AdamW",
        "lr": 5.0e-5,
        "adapter_lr": 5.0e-4,
        "weight_decay": 1.0e-6,
        "betas": (0.9, 0.999),
        "scheduler": "cosine",
        "grad_clip": 5.0,
    },
    "GPhyT-S-PT-FT": {
        "optimizer": "AdamW",
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "betas": (0.9, 0.999),
        "scheduler": "gphyt_warmup_cosine",
        "grad_clip": 5.0,
    },
    "Prithvi-EO2-300M-PT-FT": {
        "optimizer": "AdamW",
        "lr": 5.0e-5,
        "weight_decay": 0.05,
        "betas": (0.9, 0.999),
        "scheduler": "cosine",
        "grad_clip": 1.0,
    },
}


def _action(parser: argparse.ArgumentParser, destination: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == destination:
            return action
    raise KeyError(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = _REFERENCE_BUILD_PARSER()
    parser.description = __doc__
    model_action = _action(parser, "model_type")
    model_action.choices = BASELINE_NAMES
    model_action.default = BASELINE_NAMES[0]
    _action(parser, "accum_steps").default = 8
    parser.add_argument(
        "--official-root",
        default="",
        help="checkout of the selected model's audited official commit",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default="",
        help="local official checkpoint for PT-FT models",
    )
    parser.add_argument(
        "--pretrained-report-dir",
        default="",
        help="defaults to <output-dir>/pretrained_reports",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=reference.metrics_base.str2bool,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    output_explicit = bool(args.output_dir)
    _REFERENCE_VALIDATE_ARGS(args)
    if args.model_type not in BASELINE_NAMES:
        raise ValueError(args.model_type)
    if args.batch_size != 1:
        raise ValueError("native-1024 benchmark fixes --batch-size 1")
    if args.validate_every != 5:
        raise ValueError("validation/checkpoint selection is fixed every 5 epochs")
    if args.model_type not in ("ConvLSTM-Scratch", "Prithvi-EO2-300M-PT-FT"):
        if not args.official_root:
            raise ValueError(f"{args.model_type} requires --official-root")
    if args.model_type in PT_FT_NAMES and not args.pretrained_checkpoint:
        raise ValueError(f"{args.model_type} requires --pretrained-checkpoint")
    if args.model_type == "Prithvi-EO2-300M-PT-FT" and not args.activation_checkpointing:
        raise ValueError("Prithvi native-1024 requires activation checkpointing")
    if args.model_type == "EarthFormer-Scratch" and args.amp:
        args.amp = False
        args.amp_disabled_reason = (
            "EarthFormer native-1024 is fixed to FP32 after the FP16 run produced "
            "non-finite forward output"
        )
    if args.model_type == "DPOT-S-PT-FT" and args.amp:
        args.amp = False
        args.amp_disabled_reason = (
            "DPOT AFNO FP16 produced non-finite output on the native 1024 real-data smoke; "
            "run this model in FP32"
        )
    config = OPTIMIZATION[args.model_type]
    args.learning_rate = float(config["lr"])
    args.weight_decay = float(config["weight_decay"])
    args.grad_clip = float(config["grad_clip"])
    if not output_explicit:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / args.model_type)
    args.pretrained_report_dir = args.pretrained_report_dir or str(
        Path(args.output_dir) / "pretrained_reports"
    )
    args._attempted_optimizer_updates = 0
    args._successful_optimizer_updates = 0
    args._amp_skipped_updates = 0


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_type": args.model_type,
        "runtime_height": 1024,
        "runtime_width": 1024,
        "input_channels": 12,
        "output_channels": 1,
        "official_root": str(args.official_root),
        "checkpoint_path": str(args.pretrained_checkpoint),
        "report_dir": str(args.pretrained_report_dir),
        "activation_checkpointing": bool(args.activation_checkpointing),
        "prediction": "direct_normalized_h_t",
    }


def _build_model_raw(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    global _LAST_MODEL
    values = dict(config)
    model_name = str(values.pop("model_type"))
    for key, expected in (
        ("runtime_height", 1024),
        ("runtime_width", 1024),
        ("input_channels", 12),
        ("output_channels", 1),
    ):
        if int(values.pop(key)) != expected:
            raise ValueError(f"model contract {key} must equal {expected}")
    if values.pop("prediction") != "direct_normalized_h_t":
        raise ValueError("seven baselines must directly predict normalized h[t]")
    model = build_floodsimbench_baseline(
        model_name,
        official_root=str(values.pop("official_root")),
        checkpoint_path=str(values.pop("checkpoint_path")),
        report_dir=str(values.pop("report_dir")),
        activation_checkpointing=bool(values.pop("activation_checkpointing")),
    )
    if values:
        raise ValueError(f"unused model config keys: {sorted(values)}")
    model = model.to(device)
    _instrument_first_training_step(model, device)
    _LAST_MODEL = model
    if reference.is_main():
        print(
            "MODEL_INFO",
            json.dumps(model_metadata(model, model_name), ensure_ascii=False),
            flush=True,
        )
    return model


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    return build_model_with_ddp_safe_report(_build_model_raw, config, device)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    config = OPTIMIZATION[args.model_type]
    core = reference.unwrap(model)
    parameters: Any = [parameter for parameter in core.parameters() if parameter.requires_grad]
    if args.model_type == "Poseidon-T-PT-FT":
        parameters = core.parameter_groups(
            backbone_lr=float(config["lr"]),
            adapter_lr=float(config["adapter_lr"]),
        )
    optimizer_class = (
        torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    )
    optimizer = optimizer_class(
        parameters,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(config["betas"]),
        foreach=False,
    )
    expected_parameter_ids = {id(parameter) for parameter in core.parameters() if parameter.requires_grad}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if expected_parameter_ids != optimizer_parameter_ids:
        missing = len(expected_parameter_ids - optimizer_parameter_ids)
        unexpected = len(optimizer_parameter_ids - expected_parameter_ids)
        raise AssertionError(
            f"optimizer parameter coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    timing = getattr(core, "_floodsimbench_timing_audit", None)
    original_step = optimizer.step

    def timed_optimizer_step(_optimizer: torch.optim.Optimizer, *step_args: Any, **step_kwargs: Any):
        if timing is None or timing["optimizer_step_seconds"] is not None:
            return original_step(*step_args, **step_kwargs)
        _sync_cuda(next(core.parameters()).device)
        now = time.perf_counter()
        if timing["backward_start"] is not None:
            timing["backward_seconds"] = now - timing["backward_start"]
        result = original_step(*step_args, **step_kwargs)
        _sync_cuda(next(core.parameters()).device)
        timing["optimizer_step_seconds"] = time.perf_counter() - now
        return result

    optimizer.step = types.MethodType(timed_optimizer_step, optimizer)
    if reference.is_main():
        print(
            "OPTIMIZER_INFO",
            json.dumps(
                {
                    "model_type": args.model_type,
                    **config,
                    "foreach": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return optimizer


def _warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int,
):
    warmup_updates = max(1, min(int(warmup_updates), int(total_updates)))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0e-3, end_factor=1.0, total_iters=warmup_updates
    )
    if warmup_updates >= total_updates:
        return warmup
    cosine = _REAL_COSINE(optimizer, T_max=total_updates - warmup_updates)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_updates]
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_updates: int,
):
    kind = str(OPTIMIZATION[args.model_type]["scheduler"])
    total_updates = max(1, int(total_updates))
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(OPTIMIZATION[args.model_type]["lr"]),
            total_steps=total_updates,
        )
    if kind == "dpot_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3.0e-5,
            total_steps=total_updates,
            pct_start=0.2,
            div_factor=1.0e4,
            final_div_factor=1.0e4,
        )
    if kind == "warmup_cosine_20":
        return _warmup_cosine(optimizer, total_updates, int(0.2 * total_updates))
    if kind == "gphyt_warmup_cosine":
        return _warmup_cosine(
            optimizer, total_updates, min(5000, int(0.2 * total_updates))
        )
    if kind == "cosine":
        return _REAL_COSINE(optimizer, T_max=total_updates)
    if kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(kind)


def train_one_epoch(
    model: nn.Module,
    loader,
    sampler,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_update: int,
) -> Tuple[float, int, float]:
    """Reference loop plus successful AMP-step accounting and clip=0 semantics."""

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
        group_size = min(args.accum_steps, num_batches - update_index * args.accum_steps)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(group_size):
            batch = reference.move_batch(next(iterator), device)
            should_sync = micro == group_size - 1
            sync_context = (
                contextlib.nullcontext()
                if should_sync or not isinstance(model, DDP)
                else model.no_sync()
            )
            with sync_context:
                with reference.autocast_context(amp_enabled):
                    prediction = model(batch["x"])
                with reference.autocast_context(False):
                    loss = reference.masked_mse(
                        prediction.float(), batch["y"].float(), batch["mask"]
                    )
                scaler.scale(loss / group_size).backward()
            total_loss += float(loss.detach().cpu())
            loss_count += 1
        scaler.unscale_(optimizer)
        max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("gradient norm is NaN/Inf")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        succeeded = (not amp_enabled) or scaler.get_scale() >= old_scale
        args._attempted_optimizer_updates += 1
        if succeeded:
            args._successful_optimizer_updates += 1
            global_update += 1
            scheduler.step()
        else:
            args._amp_skipped_updates += 1
        if reference.is_main() and args.log_every_updates > 0 and (
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
                        "successful_optimizer_updates": args._successful_optimizer_updates,
                        "amp_skipped_updates": args._amp_skipped_updates,
                    }
                ),
                flush=True,
            )
    reduced_total, reduced_count = reference.reduce_pair(total_loss, loss_count, device)
    return reduced_total / max(reduced_count, 1), global_update, time.perf_counter() - start_time


def checkpoint_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _REFERENCE_CHECKPOINT_PAYLOAD(*args, **kwargs)
    run_args = args[6] if len(args) > 6 else kwargs["args"]
    payload["experiment"] = EXPERIMENT_NAME
    payload["optimizer_update_audit"] = {
        "attempted_optimizer_updates": int(run_args._attempted_optimizer_updates),
        "successful_optimizer_updates": int(run_args._successful_optimizer_updates),
        "amp_skipped_updates": int(run_args._amp_skipped_updates),
        "amp_skipped_ratio": (
            float(run_args._amp_skipped_updates)
            / max(int(run_args._attempted_optimizer_updates), 1)
        ),
    }
    payload["dataset"]["name"] = DATASET_NAME
    payload["dataset"]["physical_input"] = "[B,12,1024,1024]"
    payload["dataset"]["prediction"] = "direct_normalized_h_t"
    return payload


def validate_checkpoint(checkpoint: Mapping[str, Any], model_type: str) -> None:
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise RuntimeError("checkpoint belongs to another experiment")
    config = checkpoint.get("model_config", {})
    expected_config = {
        "model_type": model_type,
        "runtime_height": 1024,
        "runtime_width": 1024,
        "input_channels": 12,
        "output_channels": 1,
        "prediction": "direct_normalized_h_t",
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise RuntimeError(f"checkpoint model contract {key}: {config.get(key)!r} != {value!r}")
    contract = checkpoint.get("dataset", {})
    expected_dataset = {
        "name": DATASET_NAME,
        "preprocessing_version": data.PREPROCESSING_VERSION,
        "frames_per_sequence": 72,
        "forecast_samples_per_sequence": 67,
        "selected_train_scenarios_per_epoch": 17,
        "forecast_samples_per_epoch": 1139,
        "crop_size": 1024,
        "loss": "masked_mse_only",
    }
    for key, value in expected_dataset.items():
        if contract.get(key) != value:
            raise RuntimeError(f"checkpoint dataset contract {key}: {contract.get(key)!r} != {value!r}")


def train(*call_args, **call_kwargs):
    """Install the architecture scheduler only within the reference trainer."""

    args = call_args[0] if call_args else call_kwargs["args"]
    original_cosine = torch.optim.lr_scheduler.CosineAnnealingLR

    def scheduler_factory(optimizer, T_max, *unused_args, **unused_kwargs):
        del unused_args, unused_kwargs
        return make_scheduler(optimizer, args, int(T_max))

    torch.optim.lr_scheduler.CosineAnnealingLR = scheduler_factory
    try:
        with collective_pretrained_report_audit():
            result = _REFERENCE_TRAIN(*call_args, **call_kwargs)
    finally:
        torch.optim.lr_scheduler.CosineAnnealingLR = original_cosine
    attempted = int(args._attempted_optimizer_updates)
    skipped = int(args._amp_skipped_updates)
    ratio = skipped / max(attempted, 1)
    if reference.is_main():
        print(
            "OPTIMIZER_UPDATE_AUDIT",
            json.dumps(
                {
                    "expected_optimizer_updates": args.epochs * 143,
                    "attempted_optimizer_updates": attempted,
                    "successful_optimizer_updates": int(args._successful_optimizer_updates),
                    "amp_skipped_updates": skipped,
                    "amp_skipped_ratio": ratio,
                }
            ),
            flush=True,
        )
    if ratio > 0.001:
        raise RuntimeError(f"AMP skipped-update ratio {ratio:.6g} exceeds 0.001")
    return result


def run_smoke(*call_args, **call_kwargs) -> None:
    with collective_pretrained_report_audit():
        domain_balanced.run_smoke(*call_args, **call_kwargs)
    if _LAST_MODEL is None:
        raise AssertionError("smoke did not build a model")
    timing = dict(getattr(_LAST_MODEL, "_floodsimbench_timing_audit", {}))
    timing.pop("backward_start", None)
    if any(timing.get(key) is None for key in ("forward_seconds", "backward_seconds", "optimizer_step_seconds")):
        raise AssertionError(f"incomplete first-step timing audit: {timing}")
    gradient_audit = _gradient_depth_audit(_LAST_MODEL)
    audit_device = next(_LAST_MODEL.parameters()).device
    memory_audit: Dict[str, Any] = {"device": str(audit_device)}
    if audit_device.type == "cuda":
        memory_audit.update(
            {
                "peak_memory_allocated_GB": torch.cuda.max_memory_allocated(audit_device) / (1024 ** 3),
                "peak_memory_reserved_GB": torch.cuda.max_memory_reserved(audit_device) / (1024 ** 3),
            }
        )
    if reference.is_main():
        print(
            "SEVEN_BASELINE_STEP_AUDIT",
            json.dumps(
                {
                    "timing_seconds": timing,
                    "gradient_depth_audit": gradient_audit,
                    "optimizer_parameter_coverage": "all_requires_grad_parameters_exactly_once",
                    "gpu_memory": memory_audit,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def _install_hooks() -> None:
    domain_balanced.EXPERIMENT_NAME = EXPERIMENT_NAME
    reference.data = data
    reference.EXPERIMENT_NAME = EXPERIMENT_NAME
    reference.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    reference.MODEL_TYPES = BASELINE_NAMES
    reference.build_parser = build_parser
    reference.validate_args = validate_args
    reference.model_config_from_args = model_config_from_args
    reference.build_model = build_model
    reference.build_optimizer = build_optimizer
    reference.train_one_epoch = train_one_epoch
    reference.train = train
    reference.checkpoint_payload = checkpoint_payload
    reference.validate_checkpoint = validate_checkpoint
    reference.run_smoke = run_smoke
    reference.ScenarioDistributedBatchSampler = domain_balanced.DomainBalancedDistributedBatchSampler


_install_hooks()


def main() -> None:
    reference.main()


if __name__ == "__main__":
    main()
