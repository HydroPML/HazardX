"""Shared command-line interface for all four forecasting datasets."""

import argparse, json
from pathlib import Path
from src.core.config import load_config
from src.core.runtime import Experiment


def run(mode):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    for key in [
        "data-root",
        "dem-root",
        "pretrained-root",
        "source-root",
        "output-dir",
    ]:
        p.add_argument("--" + key)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional rollout limit for flood smoke tests; 0 evaluates every step",
    )
    p.add_argument("--protocol", choices=["source", "mozambique"], default="source")
    p.add_argument(
        "--metric-protocol", choices=["original", "cap200-pooled"], default="original"
    )
    a = p.parse_args()
    cfg = load_config(
        a.config, **{k: v for k, v in vars(a).items() if k not in ["config", "resume"]}
    )
    experiment = Experiment(cfg, device=a.device)
    if mode == "train":
        return experiment.train(a.resume)
    experiment.build_model()
    if a.metric_protocol == "cap200-pooled":
        if a.split != "test":
            p.error("The original cap200 evaluator supports the test split only")
        from src.core.cap200 import evaluate

        result = evaluate(experiment, a.max_batches)
    else:
        result = experiment.evaluate(a.max_batches, a.split, a.max_steps, a.protocol)
    target = Path(cfg["paths"]["output_dir"]) / "evaluation.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2))
    print("Saved", target)
