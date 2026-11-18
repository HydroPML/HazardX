"""Reuse the original optional Landslide cap200/pooled-CSI evaluator."""

import argparse
import importlib
import itertools
import json
from pathlib import Path


def evaluate(experiment, max_batches=0):
    if experiment.backend != "landslide":
        raise ValueError("cap200-pooled is a LandslideCast evaluation protocol")
    evaluator = importlib.import_module("evaluate_landslidecast_three_tables")
    target = experiment.output / "cap200_pooled.json"
    args = argparse.Namespace(
        model_type=experiment.args.model_type,
        checkpoint=experiment.config["checkpoint"],
        data_path=experiment.args.data_path,
        output=str(target),
        device=str(experiment.device),
        batch_size=experiment.args.eval_batch_size or experiment.args.batch_size,
        num_workers=experiment.args.num_workers,
        cap=200.0,
    )
    original_parser = evaluator.parse_args
    original_args = evaluator.make_model_args
    original_build = evaluator.runner.build_model
    original_load = evaluator.safe_load
    original_loader = evaluator.DataLoader

    def loader(*values, **options):
        batches = original_loader(*values, **options)
        return itertools.islice(batches, max_batches) if max_batches else batches

    # Adapt only the command-line/path/model-loading boundary. All masks,
    # accumulators, event thresholds, reductions, and cross-checks are native.
    evaluator.parse_args = lambda: args
    evaluator.make_model_args = lambda *_: experiment.args
    evaluator.runner.build_model = lambda *_: experiment.model
    evaluator.safe_load = lambda _: {
        **experiment.checkpoint,
        "model": experiment.model.state_dict(),
    }
    evaluator.DataLoader = loader
    try:
        evaluator.main()
        return json.loads(target.read_text())
    finally:
        evaluator.parse_args = original_parser
        evaluator.make_model_args = original_args
        evaluator.runner.build_model = original_build
        evaluator.safe_load = original_load
        evaluator.DataLoader = original_loader
