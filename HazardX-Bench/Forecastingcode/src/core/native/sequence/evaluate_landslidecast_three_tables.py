"""Evaluate completed LandslideCast checkpoints with an h <= 100 mask."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import landslidecast_accelerated_runner as runner
from data.landslidecast_2d_dataset import LandslideCast2DDataset


THRESHOLDS = (0.5, 5.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=runner.MODELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cap", type=float, default=None)
    return parser.parse_args()


def make_model_args(args, output_dir):
    model_args = runner.parse_args.__globals__["argparse"].Namespace(
        model_type=args.model_type,
        data_path=args.data_path,
        output_dir=str(output_dir.parent.parent),
        run_id=output_dir.parent.name,
        mode="train",
        epochs=100,
        validate_every=10,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        accum_steps=1,
        num_workers=args.num_workers,
        seed=0,
        max_train_updates=0,
        smoke_steps=20,
        eval_max_batches=0,
        amp=None,
        resume="",
        checkpoint_every_updates=0,
        keep_epoch_checkpoints=False,
        lead_chunk=8,
        channels_last=None,
        compile_model=False,
        prithvi_activation_checkpointing=False,
        fused_optimizer=None,
        grad_scaler_init_scale=0.0,
        check_output_equivalence=False,
        official_root="",
        pretrained_checkpoint="",
    )
    return runner.resolve(model_args)


def safe_load(path):
    return runner._safe_training_checkpoint_load(Path(path))


@torch.no_grad()
def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_args = make_model_args(args, Path(args.checkpoint).parent)
    audit_model_dir = output_path.parent / args.model_type
    audit_model_dir.mkdir(parents=True, exist_ok=True)
    model = runner.build_model(model_args, audit_model_dir, device)
    state = safe_load(args.checkpoint)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    dataset = LandslideCast2DDataset(args.data_path, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    sq = torch.zeros(2, dtype=torch.float64, device=device)
    absolute = torch.zeros(2, dtype=torch.float64, device=device)
    valid_count = torch.zeros(2, dtype=torch.int64, device=device)
    rel_sum = torch.zeros(2, dtype=torch.float64, device=device)
    csi_counts = {
        threshold: torch.zeros(3, dtype=torch.int64, device=device)
        for threshold in THRESHOLDS
    }
    official = torch.zeros(20, dtype=torch.float64, device=device)
    sample_count = 0
    finite = True

    # The second set of scalar accumulators is intentionally separate from the
    # vector accumulator above and is used as an implementation cross-check.
    audit_sq = [0.0, 0.0]
    audit_abs = [0.0, 0.0]
    audit_valid = [0, 0]
    audit_rel = [0.0, 0.0]
    audit_csi = {threshold: [0, 0, 0] for threshold in THRESHOLDS}

    for batch in loader:
        batch = runner.move(batch, device)
        with torch.autocast(
            device_type=device.type,
            enabled=model_args.amp and device.type == "cuda",
            dtype=runner.autocast_dtype(model_args),
        ):
            prediction = model(batch)
        prediction = prediction.float()
        target = batch["target"].float()
        finite = finite and bool(torch.isfinite(prediction).all())
        official += runner._metric_statistics(prediction, target)

        h_truth = target[:, :, 0]
        sample_count += target.shape[0]
        for channel in range(2):
            valid = (
                target[:, :, channel] <= args.cap
                if args.cap is not None
                else torch.ones_like(target[:, :, channel], dtype=torch.bool)
            )
            valid_count[channel] += valid.sum()
            error = (prediction[:, :, channel] - target[:, :, channel]).double()
            masked_error = error[valid]
            sq[channel] += masked_error.square().sum()
            absolute[channel] += masked_error.abs().sum()
            masked_truth = target[:, :, channel].double() * valid
            masked_dense_error = error * valid
            per_sample_rel = torch.linalg.vector_norm(
                masked_dense_error.flatten(1), dim=1
            ) / (torch.linalg.vector_norm(masked_truth.flatten(1), dim=1) + 1e-12)
            rel_sum[channel] += per_sample_rel.sum()

            audit_sq[channel] += float(masked_error.square().sum().cpu())
            audit_abs[channel] += float(masked_error.abs().sum().cpu())
            audit_rel[channel] += float(per_sample_rel.sum().cpu())
            audit_valid[channel] += int(valid.sum().cpu())

        h_prediction = prediction[:, :, 0]
        for threshold in THRESHOLDS:
            truth_event = h_truth > threshold
            pred_event = h_prediction > threshold
            if args.cap is not None:
                truth_event &= h_truth <= args.cap
                pred_event &= h_prediction <= args.cap
            hits = (truth_event & pred_event).sum()
            misses = (truth_event & ~pred_event).sum()
            false_alarms = (~truth_event & pred_event).sum()
            batch_counts = torch.stack((hits, misses, false_alarms))
            csi_counts[threshold] += batch_counts
            for index, value in enumerate(batch_counts.cpu().tolist()):
                audit_csi[threshold][index] += int(value)

    metrics = {}
    for channel, name in enumerate(("h", "u")):
        metrics[f"RMSE_{name}"] = math.sqrt(float(sq[channel]) / int(valid_count[channel]))
        metrics[f"MAE_{name}"] = float(absolute[channel]) / int(valid_count[channel])
        metrics[f"RelL2_{name}"] = float(rel_sum[channel]) / sample_count
    contingency = {}
    for threshold in THRESHOLDS:
        hits, misses, false_alarms = (int(value) for value in csi_counts[threshold])
        denominator = hits + misses + false_alarms
        metrics[f"CSI_{threshold:g}"] = hits / denominator if denominator else 0.0
        contingency[f"CSI_{threshold:g}"] = {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "denominator": denominator,
        }

    official_samples = float(official[0])
    official_recomputed = {
        "RMSE_h": math.sqrt(float(official[4]) / float(official[6])),
        "MAE_h": float(official[5]) / float(official[6]),
        "RelL2_h": float(official[7]) / official_samples,
        "RMSE_u": math.sqrt(float(official[12]) / float(official[14])),
        "MAE_u": float(official[13]) / float(official[14]),
        "RelL2_u": float(official[15]) / official_samples,
    }
    audit_metrics = {}
    for channel, name in enumerate(("h", "u")):
        audit_metrics[f"RMSE_{name}"] = math.sqrt(audit_sq[channel] / audit_valid[channel])
        audit_metrics[f"MAE_{name}"] = audit_abs[channel] / audit_valid[channel]
        audit_metrics[f"RelL2_{name}"] = audit_rel[channel] / sample_count
    for threshold in THRESHOLDS:
        hits, misses, false_alarms = audit_csi[threshold]
        audit_metrics[f"CSI_{threshold:g}"] = hits / (hits + misses + false_alarms)

    max_audit_delta = max(abs(metrics[key] - audit_metrics[key]) for key in metrics)
    report = {
        "model": args.model_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": state.get("epoch"),
        "best_validation": state.get("best_validation"),
        "test_samples": sample_count,
        "cap": args.cap,
        "valid_cells": {name: int(valid_count[channel]) for channel, name in enumerate(("h", "u"))},
        "total_cells": sample_count * 8 * 200 * 200,
        "finite_predictions": finite,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
        "contingency": contingency,
        "official_uncapped_recomputed": official_recomputed,
        "audit_metrics": audit_metrics,
        "max_audit_delta": max_audit_delta,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
