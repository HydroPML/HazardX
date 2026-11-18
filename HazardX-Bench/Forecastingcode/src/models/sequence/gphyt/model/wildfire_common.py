"""Shared loss, metrics, scheduler, and pretrained audit helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialBinaryDiceLoss(nn.Module):
    """Exact SMP ``DiceLoss(mode='binary')`` defaults used by WSTS UNet."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise AssertionError(f"Dice expects matching [B,1,H,W], got {logits.shape}/{target.shape}")
        prediction = F.logsigmoid(logits).exp().reshape(logits.shape[0], 1, -1)
        truth = target.to(logits.dtype).reshape(logits.shape[0], 1, -1)
        # Operation-for-operation port of SMP soft_dice_score ->
        # soft_tversky_score(alpha=.5, beta=.5, smooth=0, eps=1e-7).
        dims = (0, 2)
        prediction_sum = torch.sum(prediction, dim=dims)
        truth_sum = torch.sum(truth, dim=dims)
        difference = torch.linalg.vector_norm(prediction - truth, ord=1, dim=dims)
        intersection = (prediction_sum + truth_sum - difference) / 2
        fp = prediction_sum - intersection
        fn = truth_sum - intersection
        score = intersection / (intersection + .5 * fp + .5 * fn).clamp_min(1e-7)
        loss = 1.0 - score
        # SMP explicitly zeros an empty target channel's contribution.
        loss = loss * (truth.sum(dim=(0, 2)) > 0).to(loss.dtype)
        return loss.mean()


def average_precision(probability: torch.Tensor, target: torch.Tensor) -> float:
    scores = probability.detach().float().reshape(-1).cpu()
    truth = target.detach().bool().reshape(-1).cpu()
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = truth[order].float()
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1)
    return float((precision * ranked).sum() / positives)


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> Dict[str, float]:
    probability = logits.sigmoid()
    pred = probability >= threshold
    truth = target.bool()
    tp = float((pred & truth).sum()); fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    precision = tp / max(tp + fp, 1.0); recall = tp / max(tp + fn, 1.0)
    return {
        "AP": average_precision(probability, target),
        "F1": 2 * precision * recall / max(precision + recall, 1e-12),
        "IoU": tp / max(tp + fp + fn, 1.0), "precision": precision, "recall": recall,
        "Brier": float(((probability - target.float()) ** 2).mean()),
    }


def warmup_cosine_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_loading_report(model: nn.Module, checkpoint_path: str, report_path: str,
                         details: Mapping[str, object]) -> Dict[str, object]:
    payload = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **dict(details),
    }
    path = Path(report_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def assert_optimizer_complete(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if expected != actual:
        raise RuntimeError(f"optimizer parameter mismatch missing={len(expected-actual)} extra={len(actual-expected)}")
