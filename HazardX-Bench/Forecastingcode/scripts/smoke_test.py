"""Check real data, a selected checkpoint, evaluation, and one temporary step.

This does not reproduce full-test scores. It never saves modified model weights.
Use a separate process for each dataset/model to isolate native import namespaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.core.config import load_config
from src.core.runtime import Experiment
from src.data import load_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    for name in (
        "data-root",
        "dem-root",
        "pretrained-root",
        "source-root",
        "output-dir",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config, **vars(args))
    experiment = Experiment(config, args.device)
    experiment.build_model()
    experiment.model.eval()
    batch = load_sample(experiment, "test")
    with torch.inference_mode():
        prediction = experiment.predict(batch)
        metrics = experiment.batch_metrics(prediction, batch)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite evaluation prediction")
    train_batch = load_sample(experiment, "train")
    if experiment.backend == "wildfire":
        # Empty-target Dice batches legitimately produce zero loss. Search real
        # training windows using the unchanged native augmentation.
        for index in range(64):
            if float(train_batch["target"].sum()) > 0:
                break
            train_batch = load_sample(experiment, "train", index + 1)
    step = experiment.training_step(train_batch)
    report = dict(
        dataset=config["dataset"],
        model=config["model"],
        seed=config["seed"],
        checkpoint=config["checkpoint"],
        data_read=True,
        input_shapes={
            key: list(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
        },
        output_shape=list(prediction.shape),
        metrics=metrics,
        training_step=step,
        scope="Representative batch only; full-test scores are not recomputed",
    )
    target = Path(config["paths"]["output_dir"]) / "smoke_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Saved", target)


if __name__ == "__main__":
    main()
