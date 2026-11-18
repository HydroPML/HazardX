"""Fault-tolerant and accelerated runner for the unified LandslideCast benchmark."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import experiments_EarthDynamics_LandslideCast_nine_baselines as legacy
from data.landslidecast_2d_dataset import LandslideCast2DDataset, TIMES, validate_splits
from models.adapters.landslide_common import benchmark_loss

ROOT = Path(__file__).resolve().parent
MODELS = legacy.MODELS
PT = legacy.PT
OPT = legacy.OPT
PATHS = legacy.PATHS
AMP_MODELS = legacy.AMP_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=MODELS)
    parser.add_argument("--data-path", default=str(ROOT.parent / "Data2/2d_t10.h5"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/LandslideCast_accelerated"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-updates", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=20,
                        help="Compatibility alias: number of optimizer updates in smoke mode.")
    parser.add_argument("--eval-max-batches", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument(
        "--keep-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also retain immutable epoch_NNNN.pt files. By default every "
            "completed epoch atomically replaces last.pt only."
        ),
    )
    parser.add_argument("--lead-chunk", type=int, default=8, choices=(1, 2, 4, 8))
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prithvi-activation-checkpointing",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grad-scaler-init-scale", type=float, default=0.0)
    parser.add_argument("--check-output-equivalence",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-root", default="")
    parser.add_argument("--pretrained-checkpoint", default="")
    return parser.parse_args()


def resolve(args):
    if args.mode == "train" and (args.epochs != 100 or args.validate_every != 10):
        raise ValueError("benchmark fixes epochs=100 and validation_every=10")
    if min(args.batch_size, args.accum_steps, args.smoke_steps) < 1:
        raise ValueError("batch/accum/smoke values must be positive")
    if args.eval_batch_size < 0 or args.max_train_updates < 0:
        raise ValueError("eval batch size and max updates cannot be negative")
    validate_splits()
    root, checkpoint = PATHS.get(args.model_type, (None, None))
    if root and not args.official_root:
        args.official_root = str(ROOT / "official_sources/LandslideCast" / root)
    if checkpoint and not args.pretrained_checkpoint:
        args.pretrained_checkpoint = str(ROOT / "checkpoints/LandslideCast" / checkpoint)
    required_amp = args.model_type in AMP_MODELS
    if args.amp is not None and bool(args.amp) != required_amp:
        raise ValueError("model precision is fixed by the benchmark table")
    args.amp = required_amp
    # PDE-T produced an FP16 forward overflow on a finite real scenario at
    # epoch 3.  A40 has native BF16 tensor cores; BF16 keeps AMP memory/speed
    # while providing the FP32 exponent range needed by this backbone.
    args.amp_dtype = (
        "bf16" if args.model_type == "PDE-T-SC-S-PT-FT" else "fp16"
    )
    if args.channels_last is None:
        args.channels_last = args.model_type == "ConvLSTM-Scratch"
    if args.fused_optimizer is None:
        args.fused_optimizer = args.model_type == "Prithvi-EO2-300M-PT-FT"
    if args.grad_scaler_init_scale <= 0:
        # Preserve the validated benchmark precision recipe.  A previous
        # PDE-T acceleration probe used 2048 here and skipped optimizer
        # updates after FP16 overflow; scale growth remains dynamic from 1.
        args.grad_scaler_init_scale = 1.0
    if not args.run_id:
        args.run_id = "smoke" if args.mode == "smoke" else "formal_accelerated_v1"
    return args


def distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    return rank, world, local


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_dtype(args):
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def precision_name(args):
    if not args.amp:
        return "FP32"
    return "AMP BF16" if args.amp_dtype == "bf16" else "AMP FP16"


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def loader(path, split, args, rank, world, shuffle):
    dataset = LandslideCast2DDataset(path, split)
    # Keep the original single-GPU RandomSampler/data order.  A distributed
    # sampler is required only when DDP must shard complete scenarios.
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=shuffle,
            seed=args.seed,
            drop_last=False,
        )
        if world > 1
        else None
    )
    batch_size = args.batch_size if split == "train" else (args.eval_batch_size or args.batch_size)
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, data


def build_model(args, out, device):
    model = MODELS[args.model_type](
        official_root=args.official_root,
        checkpoint_path=args.pretrained_checkpoint,
        report_path=str(out / "pretrained_loading_report.json"),
        lead_chunk=args.lead_chunk,
        channels_last=args.channels_last,
    )
    if args.model_type == "Prithvi-EO2-300M-PT-FT":
        if args.prithvi_activation_checkpointing:
            model.enable_activation_checkpointing()
        else:
            model.disable_activation_checkpointing()
    return model.to(device)


def build_optimizer(model, args, device):
    name, learning_rate, weight_decay, betas, _, _ = OPT[args.model_type]
    parameters = (
        model.parameter_groups(learning_rate, 5e-4)
        if args.model_type == "Poseidon-T-PT-FT"
        else (parameter for parameter in model.parameters() if parameter.requires_grad)
    )
    optimizer_class = torch.optim.AdamW if name == "AdamW" else torch.optim.Adam
    kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=betas)
    use_fused = bool(args.fused_optimizer and device.type == "cuda")
    if use_fused:
        kwargs["fused"] = True
    else:
        kwargs["foreach"] = False
    args.fused_optimizer = use_fused
    return optimizer_class(parameters, **kwargs)


def build_scheduler(optimizer, args, updates_per_epoch):
    _, learning_rate, _, _, kind, _ = OPT[args.model_type]
    total = max(1, args.epochs * updates_per_epoch)
    if kind == "none":
        return None
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total
        )
    if kind == "dpot_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-5,
            total_steps=total,
            pct_start=.2,
            div_factor=1e4,
            final_div_factor=1e4,
        )
    if kind in ("warmup_cosine_20", "gphyt_warmup_cosine"):
        warmup = min(
            total - 1,
            max(
                1,
                int(total * .2)
                if kind == "warmup_cosine_20"
                else min(5000, int(total * .2)),
            ),
        )

        def factor(step):
            if step < warmup:
                return .001 + .999 * step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return .5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def audit_optimizer_coverage(model, optimizer):
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if expected != actual:
        raise AssertionError(
            f"optimizer coverage mismatch: missing={len(expected-actual)}, "
            f"unexpected={len(actual-expected)}"
        )


def finalize_loading_report(model, args, out, total, trainable):
    if args.model_type not in PT:
        return
    path = out / "pretrained_loading_report.json"
    payload = json.loads(path.read_text())
    loaded = payload.get("exact_loaded", []) + payload.get("converted", [])
    if not loaded:
        raise RuntimeError(f"{args.model_type}: no pretrained tensors recorded")
    payload.update(
        model=args.model_type,
        checkpoint_path=str(Path(args.pretrained_checkpoint).resolve()),
        loaded_backbone_parameter_count=payload.get("loaded_parameter_count", 0),
        parameters_total=total,
        parameters_trainable=trainable,
        loaded_depth_representatives={
            "shallow": loaded[0],
            "middle": loaded[len(loaded) // 2],
            "deep": loaded[-1],
        },
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _metric_statistics(prediction, target):
    statistics = torch.zeros(20, dtype=torch.float64, device=prediction.device)
    batch_size = prediction.shape[0]
    statistics[0] = batch_size
    relative = []
    for channel in range(2):
        error = prediction[:, :, channel].double() - target[:, :, channel].double()
        truth = target[:, :, channel].double()
        rel = torch.linalg.vector_norm(error.flatten(1), dim=1).div(
            torch.linalg.vector_norm(truth.flatten(1), dim=1) + 1e-12
        )
        last_error = error[:, -1]
        last_truth = truth[:, -1]
        last_rel = torch.linalg.vector_norm(last_error.flatten(1), dim=1).div(
            torch.linalg.vector_norm(last_truth.flatten(1), dim=1) + 1e-12
        )
        base = 4 + channel * 8
        statistics[base] = error.square().sum()
        statistics[base + 1] = error.abs().sum()
        statistics[base + 2] = error.numel()
        statistics[base + 3] = rel.sum()
        statistics[base + 4] = last_error.square().sum()
        statistics[base + 5] = last_error.abs().sum()
        statistics[base + 6] = last_error.numel()
        statistics[base + 7] = last_rel.sum()
        relative.append(rel)
    statistics[2] = relative[0].sum()
    statistics[3] = relative[1].sum()
    statistics[1] = 7.0 * statistics[2] + statistics[3]
    return statistics


@torch.no_grad()
def evaluate(model, data, device, max_batches, args, world):
    model.eval()
    statistics = torch.zeros(20, dtype=torch.float64, device=device)
    for index, batch in enumerate(data):
        if max_batches and index >= max_batches:
            break
        batch = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
            dtype=autocast_dtype(args),
        ):
            prediction = model(batch)
        statistics += _metric_statistics(prediction.float(), batch["target"].float())
    if world > 1:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    samples = statistics[0].item()
    if samples == 0:
        raise RuntimeError("evaluation processed zero samples")
    result = {
        "objective": statistics[1].item() / samples,
        "official_RelL2_h": statistics[2].item() / samples,
        "official_RelL2_u": statistics[3].item() / samples,
    }
    for channel, name in enumerate(("h", "u")):
        base = 4 + channel * 8
        result[f"RMSE_{name}"] = math.sqrt(
            statistics[base].item() / statistics[base + 2].item()
        )
        result[f"MAE_{name}"] = (
            statistics[base + 1].item() / statistics[base + 2].item()
        )
        result[f"RelL2_{name}"] = statistics[base + 3].item() / samples
        result[f"RMSE_{name}@t80"] = math.sqrt(
            statistics[base + 4].item() / statistics[base + 6].item()
        )
        result[f"MAE_{name}@t80"] = (
            statistics[base + 5].item() / statistics[base + 6].item()
        )
        result[f"RelL2_{name}@t80"] = statistics[base + 7].item() / samples
    return result


def gradient_audit(model, model_type):
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("backbone" in name or model_type not in PT)
    ]
    if not candidates:
        raise RuntimeError("no trainable parameters available for gradient audit")
    probes = {
        label: candidates[index]
        for label, index in (
            ("shallow", 0),
            ("middle", len(candidates) // 2),
            ("deep", len(candidates) - 1),
        )
    }
    return {
        label: {
            "parameter": name,
            "present": parameter.grad is not None,
            "finite": parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all()),
            "nonzero": parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad)),
        }
        for label, (name, parameter) in probes.items()
    }


def _safe_training_checkpoint_load(path):
    # Old rolling checkpoints contain a NumPy RNG ndarray.  Keep the
    # weights-only unpickler and narrowly allow only NumPy array construction;
    # arbitrary checkpoint globals remain rejected.
    numpy_reconstruct = np.core.multiarray._reconstruct
    safe_globals = [
        numpy_reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def _local_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, dict):
        numpy_state = (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path,
    raw_model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    batch_in_epoch,
    global_update,
    amp_skipped,
    best,
    args,
    rank,
    world,
    aliases=(),
):
    local_rng = _local_rng_state()
    rng_by_rank = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if rank != 0:
        return
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch - 1 if batch_in_epoch == 0 else epoch,
        "next_epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_update": global_update,
        "amp_skipped_updates": amp_skipped,
        "best_validation": best,
        "rng_by_rank": rng_by_rank,
        "config": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # Rolling aliases (normally last.pt) are hard links to the immutable epoch
    # checkpoint, so they cost no additional model/optimizer storage. Replace
    # atomically so an interrupted update never destroys the previous alias.
    for alias in aliases:
        alias_temporary = alias.with_suffix(alias.suffix + ".tmp")
        alias_temporary.unlink(missing_ok=True)
        os.link(path, alias_temporary)
        os.replace(alias_temporary, alias)


def load_checkpoint(
    path, raw_model, optimizer, scheduler, scaler, rank, device
):
    state = _safe_training_checkpoint_load(path)
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    rng_by_rank = state.get("rng_by_rank")
    if rng_by_rank:
        _restore_rng_state(rng_by_rank[min(rank, len(rng_by_rank) - 1)])
    return state


@torch.no_grad()
def check_output_equivalence(raw_model, execution_model, data, device, args):
    probe = move(next(iter(data)), device)
    raw_model.eval()
    original_chunk = getattr(raw_model, "lead_chunk", None)
    original_channels_last = getattr(raw_model, "channels_last", None)
    if original_chunk is not None:
        raw_model.lead_chunk = 1
    if original_channels_last is not None:
        raw_model.channels_last = False
    # Poseidon grouped and sequential GEMMs can select different FP16 kernels.
    # Its implementation equivalence is therefore audited in FP32; AMP stability
    # is checked separately by the training updates below.
    audit_amp = (
        args.amp
        and args.model_type != "Poseidon-T-PT-FT"
        and device.type == "cuda"
    )
    with torch.autocast(
        device_type=device.type,
        enabled=audit_amp,
        dtype=autocast_dtype(args),
    ):
        reference = raw_model(probe).float()
    if original_chunk is not None:
        raw_model.lead_chunk = original_chunk
    if original_channels_last is not None:
        raw_model.channels_last = original_channels_last
    with torch.autocast(
        device_type=device.type,
        enabled=audit_amp,
        dtype=autocast_dtype(args),
    ):
        candidate = execution_model(probe).float()
    difference = reference - candidate
    maximum = difference.abs().max().item()
    mean_absolute = difference.abs().mean().item()
    relative_delta = (
        torch.linalg.vector_norm(difference.flatten(1), dim=1)
        / (torch.linalg.vector_norm(reference.flatten(1), dim=1) + 1e-12)
    ).mean().item()
    reference_loss = float(
        benchmark_loss(reference, probe["target"].float())[0]
    )
    candidate_loss = float(
        benchmark_loss(candidate, probe["target"].float())[0]
    )
    tolerance = (1e-3, 1e-2) if audit_amp else (1e-5, 1e-6)
    elementwise_close = torch.allclose(
        reference, candidate, rtol=tolerance[0], atol=tolerance[1]
    )
    loss_delta = abs(reference_loss - candidate_loss)
    # BF16 has a 7-bit mantissa (epsilon ~= 7.8e-3), so changing the lead
    # chunk changes GEMM reduction order enough to make the former FP16
    # 1e-3 relative threshold spuriously fail. Keep the loss requirement
    # unchanged and use a still-strict, dtype-aware relative-output bound.
    relative_tolerance = (
        5e-3 if audit_amp and args.amp_dtype == "bf16" else 1e-3
    )
    loss_tolerance = 1e-3
    equivalent = (
        relative_delta <= relative_tolerance and loss_delta <= loss_tolerance
    )
    if not equivalent:
        raise AssertionError(
            "accelerated output mismatch: "
            f"max_abs={maximum}, mean_abs={mean_absolute}, "
            f"relative_delta={relative_delta}, tolerance={tolerance}"
        )
    return {
        "equivalent": equivalent,
        "elementwise_allclose": elementwise_close,
        "max_abs": maximum,
        "mean_abs": mean_absolute,
        "relative_l2_delta": relative_delta,
        "relative_l2_tolerance": relative_tolerance,
        "reference_loss": reference_loss,
        "candidate_loss": candidate_loss,
        "loss_abs_delta": loss_delta,
        "loss_abs_tolerance": loss_tolerance,
        "audit_precision": precision_name(args) if audit_amp else "FP32",
        "rtol": tolerance[0],
        "atol": tolerance[1],
    }


def main():
    args = resolve(parse_args())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rank, world, local = distributed()
    device = torch.device(
        f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    )
    if (
        device.type == "cuda"
        and args.amp_dtype == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("PDE-T requires native CUDA BF16 support")
    effective_batch = args.batch_size * args.accum_steps * world
    if args.mode == "train" and effective_batch != 8:
        raise ValueError(
            f"formal benchmark requires effective batch 8, got {effective_batch}"
        )
    out = Path(args.output_dir) / args.run_id / args.model_type
    out.mkdir(parents=True, exist_ok=True)

    train_set, train_data = loader(
        args.data_path, "train", args, rank, world, True
    )
    _, validation_data = loader(
        args.data_path, "val", args, rank, world, False
    )
    _, test_data = loader(
        args.data_path, "test", args, rank, world, False
    )
    raw_model = build_model(args, out, device)
    total, trainable = parameter_counts(raw_model)
    optimizer = build_optimizer(raw_model, args, device)
    audit_optimizer_coverage(raw_model, optimizer)
    finalize_loading_report(raw_model, args, out, total, trainable)
    updates_per_epoch = max(1, math.ceil(len(train_data) / args.accum_steps))
    scheduler = build_scheduler(optimizer, args, updates_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
        init_scale=args.grad_scaler_init_scale,
    )

    execution_model = raw_model
    if args.compile_model:
        execution_model = torch.compile(raw_model, mode="reduce-overhead")
    equivalence = None
    if args.check_output_equivalence:
        equivalence = check_output_equivalence(
            raw_model, execution_model, validation_data, device, args
        )
    model = execution_model
    if world > 1:
        model = DDP(
            execution_model,
            device_ids=[local],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    if rank == 0:
        first = train_set[0]
        print(
            "DATASET_AUDIT",
            json.dumps(
                {
                    "path": args.data_path,
                    "scenario_count": 12000,
                    "first_train": train_set.names[0],
                    "last_train": train_set.names[-1],
                    "raw_shape": [1, 200, 200, 9, 4],
                    "channels": ["fai", "z", "h", "u"],
                    "times": TIMES.tolist(),
                    "h0_shape": list(first["h0"].shape),
                    "target_shape": list(first["target"].shape),
                    "split_counts": [10000, 1000, 1000],
                    "z_norm_range": [
                        float(first["z_norm"].min()),
                        float(first["z_norm"].max()),
                    ],
                    "h0_norm_range": [
                        float(first["h0_norm"].min()),
                        float(first["h0_norm"].max()),
                    ],
                }
            ),
        )
        train_set.close()
        print(
            "OPTIMIZATION",
            json.dumps(
                {
                    "model": args.model_type,
                    "optimizer": OPT[args.model_type][0],
                    "lr": OPT[args.model_type][1],
                    "weight_decay": OPT[args.model_type][2],
                    "betas": OPT[args.model_type][3],
                    "scheduler": OPT[args.model_type][4],
                    "grad_clip": OPT[args.model_type][5],
                    "precision": precision_name(args),
                    "epochs": args.epochs,
                    "validation_every": args.validate_every,
                    "effective_batch": effective_batch,
                    "num_workers": args.num_workers,
                    "world_size": world,
                    "lead_chunk": args.lead_chunk,
                    "channels_last": args.channels_last,
                    "compile_model": args.compile_model,
                    "prithvi_activation_checkpointing":
                        args.prithvi_activation_checkpointing,
                    "fused_optimizer": args.fused_optimizer,
                    "grad_scaler_init_scale": args.grad_scaler_init_scale,
                    "checkpoint_every_updates": args.checkpoint_every_updates,
                    "keep_epoch_checkpoints": args.keep_epoch_checkpoints,
                    "output_equivalence": equivalence,
                    "parameters_total": total,
                    "parameters_trainable": trainable,
                }
            ),
        )

    start_epoch = 0
    resume_batch = 0
    global_update = 0
    amp_skipped = 0
    best = float("inf")
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(out / "last.pt")
    if resume_path:
        state = load_checkpoint(
            resume_path,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            rank,
            device,
        )
        start_epoch = int(state.get("next_epoch", state.get("epoch", 0)))
        resume_batch = int(state.get("batch_in_epoch", 0))
        global_update = int(state.get("global_update", 0))
        amp_skipped = int(state.get("amp_skipped_updates", 0))
        best = float(state.get("best_validation", float("inf")))

    # Smoke outputs live in an isolated run directory; establish their own
    # one-batch best checkpoint instead of inheriting an absent formal file.
    if args.mode == "smoke":
        best = float("inf")
    # A resumed smoke test must execute one epoch window from its checkpoint.
    max_epochs = start_epoch + 1 if args.mode == "smoke" else args.epochs
    max_updates = (
        args.max_train_updates
        if args.max_train_updates
        else (args.smoke_steps if args.mode == "smoke" else 0)
    )
    optimizer.zero_grad(set_to_none=True)
    updates_this_run = 0
    checkpoint_epoch = start_epoch
    checkpoint_batch = resume_batch
    validation = None
    stop_training = False
    window_start = time.perf_counter()
    window_samples = 0
    window_forward = 0.0
    window_backward = 0.0
    measurements = []
    # The benchmark requires at least one shallow/middle/deep non-zero
    # gradient audit, not non-zero probes on every physical scenario.  A
    # dry/static sample can legitimately produce zero representative grads.
    gradient_nonzero_verified = (
        args.model_type not in PT
        or (args.mode == "train" and global_update > 0)
    )

    for epoch in range(start_epoch, max_epochs):
        if hasattr(train_data.sampler, "set_epoch"):
            train_data.sampler.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(train_data):
            if epoch == start_epoch and step < resume_batch:
                continue
            batch = move(batch, device)
            is_loader_end = step + 1 == len(train_data)
            should_step = (step + 1) % args.accum_steps == 0 or is_loader_end
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else contextlib.nullcontext()
            )
            with sync_context:
                synchronize(device)
                begin = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    enabled=args.amp and device.type == "cuda",
                    dtype=autocast_dtype(args),
                ):
                    prediction = model(batch)
                    loss, relative_h, relative_u = benchmark_loss(
                        prediction.float(), batch["target"].float()
                    )
                    if not bool(torch.isfinite(prediction).all()) or not bool(
                        torch.isfinite(loss)
                    ):
                        raise FloatingPointError(
                            f"{args.model_type}: non-finite prediction or loss"
                        )
                    scaled_loss = loss / args.accum_steps
                synchronize(device)
                forward_time = time.perf_counter() - begin
                begin = time.perf_counter()
                scaler.scale(scaled_loss).backward()
                synchronize(device)
                backward_time = time.perf_counter() - begin
            window_samples += prediction.shape[0] * world
            window_forward += forward_time
            window_backward += backward_time
            checkpoint_epoch = epoch
            checkpoint_batch = step + 1
            if is_loader_end:
                checkpoint_epoch = epoch + 1
                checkpoint_batch = 0
            if not should_step:
                continue

            # unscale_ records non-finite gradients in GradScaler.  Under AMP,
            # scaler.step()/update() must be allowed to consume that record,
            # skip the unsafe optimizer step, and back off the scale.  Raising
            # before scaler.update() turns an expected recoverable FP16
            # overflow into a fatal training error (observed for GPhyT after
            # its dynamic scale grew from 4096 to 8192).
            scaler.unscale_(optimizer)
            audit = (
                gradient_audit(raw_model, args.model_type)
                if args.model_type in PT
                else {}
            )
            if audit and not all(value["present"] for value in audit.values()):
                raise FloatingPointError(
                    f"{args.model_type}: missing "
                    f"shallow/middle/deep gradient {audit}"
                )
            audit_finite = all(
                value["finite"] for value in audit.values()
            ) if audit else True
            clip = OPT[args.model_type][5]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), clip if clip else float("inf")
            )
            grad_norm_finite = bool(torch.isfinite(grad_norm))
            if not args.amp and (not audit_finite or not grad_norm_finite):
                raise FloatingPointError(
                    f"{args.model_type}: non-finite FP32 gradient "
                    f"norm={float(grad_norm)} audit={audit}"
                )
            old_scale = scaler.get_scale()
            synchronize(device)
            begin = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            optimizer_time = time.perf_counter() - begin
            succeeded = (not args.amp) or scaler.get_scale() >= old_scale
            if succeeded:
                if not audit_finite or not grad_norm_finite:
                    raise FloatingPointError(
                        f"{args.model_type}: GradScaler accepted non-finite "
                        f"gradient norm={float(grad_norm)} audit={audit}"
                    )
                if audit and all(value["nonzero"] for value in audit.values()):
                    gradient_nonzero_verified = True
                if scheduler is not None:
                    scheduler.step()
                global_update += 1
                updates_this_run += 1
            else:
                amp_skipped += 1
            elapsed = time.perf_counter() - window_start
            samples_per_second = window_samples / max(elapsed, 1e-12)
            measurement = {
                "samples_per_second": samples_per_second,
                "forward_seconds": window_forward / args.accum_steps,
                "backward_seconds": window_backward / args.accum_steps,
                "optimizer_seconds": optimizer_time,
                "loss": float(loss),
                "peak_allocated": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                "peak_reserved": (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                ),
            }
            if succeeded:
                measurements.append(measurement)
            if rank == 0:
                print(
                    "TRAIN",
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_update": global_update,
                            "run_update": updates_this_run,
                            "optimizer_step_succeeded": succeeded,
                            "amp_skipped_updates": amp_skipped,
                            "amp_overflow_recovered": bool(args.amp and not succeeded),
                            "grad_scale": float(scaler.get_scale()),
                            "grad_norm": (
                                float(grad_norm) if grad_norm_finite else None
                            ),
                            "grad_norm_finite": grad_norm_finite,
                            "shape": list(prediction.shape),
                            "loss": float(loss),
                            "RelL2_h": float(relative_h),
                            "RelL2_u": float(relative_u),
                            "samples_per_second": samples_per_second,
                            "forward_seconds": measurement["forward_seconds"],
                            "backward_seconds": measurement["backward_seconds"],
                            "optimizer_seconds": optimizer_time,
                            "peak_allocated": measurement["peak_allocated"],
                            "peak_reserved": measurement["peak_reserved"],
                            "gradient_audit": audit,
                        }
                    ),
                )
            window_start = time.perf_counter()
            window_samples = 0
            window_forward = 0.0
            window_backward = 0.0

            if (
                succeeded
                and args.checkpoint_every_updates
                and global_update % args.checkpoint_every_updates == 0
            ):
                save_checkpoint(
                    out / "last.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    checkpoint_epoch,
                    checkpoint_batch,
                    global_update,
                    amp_skipped,
                    best,
                    args,
                    rank,
                    world,
                )
            if max_updates and updates_this_run >= max_updates:
                stop_training = True
                break

        if (
            args.model_type in PT
            and not gradient_nonzero_verified
            and updates_this_run >= 20
        ):
            raise FloatingPointError(
                f"{args.model_type}: no update with non-zero "
                "shallow/middle/deep representative gradients in 20 updates"
            )

        resume_batch = 0
        validate_now = (
            args.mode == "smoke"
            or stop_training
            or (checkpoint_batch == 0 and checkpoint_epoch % args.validate_every == 0)
        )
        if validate_now:
            validation = evaluate(
                model,
                validation_data,
                device,
                args.eval_max_batches if args.mode == "smoke" else 0,
                args,
                world,
            )
            improved = validation["objective"] < best
            if improved:
                best = validation["objective"]
                save_checkpoint(
                    out / "best_validation.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    checkpoint_epoch,
                    checkpoint_batch,
                    global_update,
                    amp_skipped,
                    best,
                    args,
                    rank,
                    world,
                )
            if rank == 0:
                print("VALIDATION", json.dumps(validation))
        completed_epoch = checkpoint_batch == 0 and checkpoint_epoch > 0
        checkpoint_epoch_index = (
            checkpoint_epoch - 1 if completed_epoch else checkpoint_epoch
        )
        epoch_checkpoint = out / f"epoch_{checkpoint_epoch_index:04d}.pt"
        checkpoint_path = (
            epoch_checkpoint
            if completed_epoch and args.keep_epoch_checkpoints
            else out / "last.pt"
        )
        aliases = (out / "last.pt",) if checkpoint_path != out / "last.pt" else ()
        save_checkpoint(
            checkpoint_path,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            checkpoint_epoch,
            checkpoint_batch,
            global_update,
            amp_skipped,
            best,
            args,
            rank,
            world,
            aliases=aliases,
        )
        if rank == 0:
            print(
                "CHECKPOINT",
                json.dumps(
                    {
                        "epoch": checkpoint_epoch_index,
                        "next_epoch": checkpoint_epoch,
                        "batch_in_epoch": checkpoint_batch,
                        "path": str(checkpoint_path),
                        "rolling_overwrite": checkpoint_path == out / "last.pt",
                    }
                ),
            )
        if stop_training:
            break

    if world > 1:
        dist.barrier()
    best_path = out / "best_validation.pt"
    if not best_path.exists():
        raise RuntimeError("training finished without a best-validation checkpoint")
    best_state = _safe_training_checkpoint_load(best_path)
    raw_model.load_state_dict(best_state["model"])
    test = evaluate(
        model,
        test_data,
        device,
        args.eval_max_batches if args.mode == "smoke" else 0,
        args,
        world,
    )
    if rank == 0:
        print("TEST", json.dumps(test))
        probe = move(next(iter(validation_data)), device)
        raw_model.eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
            dtype=autocast_dtype(args),
        ):
            before = raw_model(probe).float()
        reload_model = build_model(args, out, device)
        finalize_loading_report(reload_model, args, out, total, trainable)
        reload_model.load_state_dict(best_state["model"])
        reload_model.eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
            dtype=autocast_dtype(args),
        ):
            after = reload_model(probe).float()
        identical = (
            torch.equal(before, after)
            if not args.amp
            else torch.allclose(before, after, rtol=1e-5, atol=1e-6)
        )
        if not identical:
            raise AssertionError("checkpoint reload output mismatch")
        steady = measurements[2:] if len(measurements) > 2 else measurements
        performance = {
            "steady_updates": len(steady),
            "samples_per_second_median": (
                float(np.median([item["samples_per_second"] for item in steady]))
                if steady else None
            ),
            "forward_seconds_median": (
                float(np.median([item["forward_seconds"] for item in steady]))
                if steady else None
            ),
            "backward_seconds_median": (
                float(np.median([item["backward_seconds"] for item in steady]))
                if steady else None
            ),
            "optimizer_seconds_median": (
                float(np.median([item["optimizer_seconds"] for item in steady]))
                if steady else None
            ),
            "peak_allocated": max(
                (item["peak_allocated"] for item in measurements), default=0
            ),
            "peak_reserved": max(
                (item["peak_reserved"] for item in measurements), default=0
            ),
            "loss_all_finite": all(
                math.isfinite(item["loss"]) for item in measurements
            ),
        }
        summary = {
            "model": args.model_type,
            "mode": args.mode,
            "run_id": args.run_id,
            "global_update": global_update,
            "updates_this_run": updates_this_run,
            "amp_skipped_updates": amp_skipped,
            "precision": precision_name(args),
            "gradient_nonzero_verified": gradient_nonzero_verified,
            "validation": validation,
            "test": test,
            "output_equivalence": equivalence,
            "performance": performance,
            "reload_consistent": identical,
            "parameters_total": total,
            "parameters_trainable": trainable,
        }
        (out / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print("COMPLETE", json.dumps(summary))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
