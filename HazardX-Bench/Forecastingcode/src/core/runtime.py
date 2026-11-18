"""Uniform experiment interface around the unchanged, audited numerical code.

Each process uses one native namespace. This preserves original state_dict keys
and avoids cross-dataset imports silently replacing model implementations.
"""

from __future__ import annotations
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ENTRY = {
    "floodcast_two": "experiments_EarthDynamics_FloodCastBench_two_baselines_scene_trajectory",
    "floodcast_seven": "experiments_EarthDynamics_FloodCastBench_seven_baselines_scene_trajectory",
    "floodsim_two": "experiments_EarthDynamics_FloodSimBench_two_baselines_crop1024_domain_balanced",
    "floodsim_seven": "experiments_EarthDynamics_FloodSimBench_seven_baselines_context6_crop1024_domain_balanced",
    "floodsim_pde": "experiments_EarthDynamics_FloodSimBench_PDE_T_context6_crop1024_domain_balanced_4gpu_accelerated",
    "landslide": "landslidecast_accelerated_runner",
    "wildfire": "experiments_EarthDynamics_WildfireSpreadTS_nine_baselines",
}


def seed_all(seed, deterministic=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Only the flood experiments explicitly configure these backend flags.
    # Other native entries retain PyTorch's defaults instead of gaining a new
    # deterministic/benchmark policy through the common interface.
    if deterministic is not None:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def activate(backend, reference_root=None):
    if reference_root:
        paths = [Path(reference_root)]
    else:
        group = "flood" if backend.startswith("flood") else "sequence"
        paths = [ROOT / "src/core/native" / group, ROOT / "src/models" / group]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    return importlib.import_module(ENTRY[backend])


@contextlib.contextmanager
def arguments(argv):
    before = sys.argv
    sys.argv = [before[0], *argv]
    try:
        yield
    finally:
        sys.argv = before


def parser_argv(parser, values):
    """Serialize only real parser options; derived checkpoint fields stay metadata."""
    argv = []
    for action in parser._actions:
        if action.dest not in values or not action.option_strings:
            continue
        value = values[action.dest]
        # Checkpoints store resolved internal schedules; native validators restore
        # those schedules from model_type after parsing the public 'auto' option.
        if (
            action.dest == "scheduler"
            and action.choices
            and value not in action.choices
        ):
            value = "auto" if "auto" in action.choices else "cosine"
        if value is None:
            continue
        option = next(
            (s for s in action.option_strings if s.startswith("--")),
            action.option_strings[0],
        )
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(
                option
                if value
                else next(s for s in action.option_strings if s.startswith("--no-"))
            )
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(option)
        else:
            argv.append(option)
            argv.extend(
                str(v).lower() if isinstance(v, bool) else str(v)
                for v in (value if isinstance(value, (list, tuple)) else [value])
            )
    return argv


class Experiment:
    """Build, train, and evaluate one verified dataset/model configuration."""

    def __init__(self, config, device="cuda", reference_root=None):
        self.config = config
        self.backend = config["backend"]
        self.device = torch.device(device)
        self.native = activate(self.backend, reference_root)
        self.output = Path(config["paths"]["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        values = dict(config["native_args"])
        if self.backend.startswith("floodcast"):
            self.native.install_experiment_hooks()
            self.base = importlib.import_module(
                "experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe"
            )
            parser = self.base.build_parser()
            self.args = parser.parse_args(parser_argv(parser, values))
            self.base.validate_args(self.args)
        elif self.backend.startswith("floodsim"):
            self.base = self.native.reference
            parser = self.base.build_parser()
            self.args = parser.parse_args(parser_argv(parser, values))
            # The original PDE training recipe requires four ranks. Inspection
            # validates its exact recipe without requiring DDP for inference.
            original_mode = self.args.mode
            if self.backend == "floodsim_pde":
                self.args.mode = "inspect"
            try:
                self.base.validate_args(self.args)
            finally:
                self.args.mode = original_mode
        elif self.backend == "landslide":
            with arguments(["--model-type", values["model_type"]]):
                self.args = self.native.parse_args()
            vars(self.args).update(values)
            self.args = self.native.resolve(self.args)
        else:
            parser = self.native.parser()
            self.args = parser.parse_args(parser_argv(parser, values))
            self.args._pretrained_report_path = str(
                self.output / "pretrained_loading_report.json"
            )
        self.model = None
        self.normalizer = None
        self.checkpoint = None
        seed_all(config["seed"], values.get("deterministic"))

    def build_model(self, load_weights=True):
        if self.backend.startswith("floodcast"):
            model = self.base.build_model(
                dict(self.config["model_config"]), self.device
            )
        elif self.backend.startswith("floodsim"):
            model = self.base.build_model(
                dict(self.config["model_config"]), self.device
            )
        elif self.backend == "landslide":
            model = self.native.build_model(self.args, self.output, self.device)
        else:
            model = self.native.build_model(self.args).to(self.device)
        if load_weights:
            cp = torch.load(
                self.config["checkpoint"],
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            state = cp["model_state"] if "model_state" in cp else cp["model"]
            model.load_state_dict(state, strict=True)
            if "normalization" in cp:
                normalizer_cls = (
                    self.base.ChannelNormalizer
                    if self.backend.startswith("floodcast")
                    else self.base.metrics_base.ChannelNormalizer
                )
                self.normalizer = normalizer_cls.from_state_dict(cp["normalization"])
                if self.backend.startswith("floodcast"):
                    centered = importlib.import_module(
                        "floodcast_transolver_centered_dem_common"
                    )
                    centered._set_huber_scale(
                        float(self.normalizer.std[0]), float(self.args.huber_delta_m)
                    )
            self.checkpoint = {
                k: v
                for k, v in cp.items()
                if k
                not in [
                    "model",
                    "model_state",
                    "optimizer",
                    "optimizer_state",
                    "rng_state",
                ]
            }
            del cp
        self.model = model
        return model

    def move(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def autocast(self, training=False):
        # Native evaluation is FP32 for FloodSimBench and WildfireSpreadTS.
        enabled = bool(getattr(self.args, "amp", False)) and self.device.type == "cuda"
        if training and self.backend == "wildfire":
            enabled = self.device.type == "cuda"
        if not training and (
            self.backend.startswith("flood") or self.backend == "wildfire"
        ):
            enabled = False
        dtype = (
            torch.bfloat16
            if getattr(self.args, "amp_dtype", "fp16") == "bf16"
            or self.backend == "wildfire"
            else torch.float16
        )
        return torch.autocast(self.device.type, enabled=enabled, dtype=dtype)

    def predict(self, batch, training=False):
        batch = self.move(batch)
        with self.autocast(training):
            if self.backend.startswith("floodcast"):
                delta = self.model(x=batch["x"], static=batch["static"], metadata=batch)
                return self.base.constrain_nonnegative_water_state(
                    batch["x"][:, :1].float() + delta.float(), self.normalizer
                )
            if self.backend.startswith("floodsim"):
                return self.model(batch["x"]).float()
            if self.backend == "landslide":
                return self.model(batch).float()
            return self.native.model_forward(self.model, batch, self.device).float()

    def loss(self, prediction, batch):
        batch = self.move(batch)
        if self.backend.startswith("floodcast"):
            centered = importlib.import_module(
                "floodcast_transolver_centered_dem_common"
            )
            return centered.masked_huber_mse_matched(
                prediction.float(),
                batch["y"].float(),
                batch["mask"],
                delta_norm=float(self.args.huber_delta_m)
                / float(self.normalizer.std[0]),
            )
        if self.backend.startswith("floodsim"):
            return self.base.masked_mse(
                prediction.float(), batch["y"].float(), batch["mask"]
            )
        if self.backend == "landslide":
            return self.native.benchmark_loss(
                prediction.float(), batch["target"].float()
            )[0]
        return self.native.OfficialBinaryDiceLoss()(prediction, batch["target"])

    def batch_metrics(self, prediction, batch):
        batch = self.move(batch)
        if self.backend.startswith("flood"):
            p = self.normalizer.denormalize(prediction, 0)
            t = self.normalizer.denormalize(batch["y"], 0)
            if self.backend.startswith("floodsim"):
                data = importlib.import_module(
                    "data.floodsimbench_crop1024_context6_dataset"
                )
                p = p.clamp_min(0.0)
                p = torch.where(
                    batch["mask"] & (p < data.DEPTH_CLEAN_THRESHOLD_M),
                    torch.zeros_like(p),
                    p,
                )
            metric = (
                self.base.masked_step_metrics
                if self.backend.startswith("floodcast")
                else self.base.metrics_base.masked_step_metrics
            )
            return metric(p, t, batch["mask"])
        if self.backend == "wildfire":
            return self.native.binary_metrics(
                prediction.detach().cpu(),
                batch["target"].detach().cpu(),
                self.config["evaluation"]["threshold"],
            )

        # Use the native evaluator, including its autocast and sample weighting.
        class Fixed(torch.nn.Module):
            def forward(self, ignored):
                return prediction

        return self.native.evaluate(Fixed(), [batch], self.device, 1, self.args, 1)

    def optimizer(self):
        if self.backend.startswith("floodcast"):
            module = importlib.import_module(
                "experiments_EarthDynamics_FloodCastBench_seven_baselines"
                if self.backend.endswith("seven")
                else "floodcast_four_baselines"
            )
            return module.build_optimizer(self.model.parameters(), self.args)
        if self.backend.startswith("floodsim"):
            return self.base.build_optimizer(self.model, self.args)
        if self.backend == "landslide":
            return self.native.build_optimizer(self.model, self.args, self.device)
        return self.native.optimizer(self.model, self.args)[0]

    def training_step(self, batch):
        """One temporary optimizer update; never writes or overwrites a checkpoint."""
        self.model.train()
        optimizer = self.optimizer()
        optimizer.zero_grad(set_to_none=True)
        prediction = self.predict(batch, training=True)
        loss = self.loss(prediction, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        use_scaler = (
            self.device.type == "cuda"
            and getattr(self.args, "amp", False)
            and getattr(self.args, "amp_dtype", "fp16") != "bf16"
            and self.backend != "wildfire"
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_scaler,
            init_scale=float(
                getattr(
                    self.args,
                    "grad_scaler_init_scale",
                    getattr(self.args, "amp_initial_scale", 1.0),
                )
            ),
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [
            p.grad
            for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        if not finite:
            raise FloatingPointError("Non-finite or absent training gradients")
        if self.backend == "wildfire":
            self.native.maybe_clip_gradients(self.model, self.args)
        elif float(getattr(self.args, "grad_clip", 0) or 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        elif self.backend == "landslide":
            clip = self.native.OPT[self.args.model_type][-1]
            if clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        # A tiny gradient may round away in one arbitrary bias. Probe several
        # tensors with the largest gradients to verify a real optimizer update.
        probes = sorted(
            (
                p
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ),
            key=lambda p: float(p.grad.detach().abs().max()),
            reverse=True,
        )[:8]
        before = [p.detach().clone() for p in probes]
        scaler.step(optimizer)
        scaler.update()
        changed = any(
            bool(torch.any(p.detach() != old)) for p, old in zip(probes, before)
        )
        result = dict(
            loss=float(loss.detach()),
            finite_gradients=finite,
            gradient_tensors=len(grads),
            optimizer_step_executed=True,
            parameters_changed=changed,
            gradient_max=max(float(g.detach().abs().max()) for g in grads),
        )
        if not changed:
            # Keep the native loss and numerical precision even when a trained
            # model has gradients too small to change FP32 parameters.
            result["warning"] = (
                "Native finite gradients are below parameter update precision on this batch; optimizer executed without a detectable parameter change."
            )
        optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        return result

    def train(self, resume=None):
        """Enter the unchanged full native training/validation lifecycle."""
        expected = int(self.config["distributed"]["world_size"])
        actual = int(os.environ.get("WORLD_SIZE", "1"))
        if actual != expected:
            raise ValueError(
                f"This experiment requires WORLD_SIZE={expected}; got {actual}. Use torchrun --nproc_per_node={expected} to preserve its effective batch and schedule."
            )
        values = dict(vars(self.args))
        values["resume"] = str(resume or "")
        values["mode"] = "train"
        values["output_dir"] = str(self.output)
        if self.backend == "landslide":
            # Native parse_args owns its parser. Pass only its documented options.
            accepted = {
                "model_type",
                "data_path",
                "output_dir",
                "run_id",
                "mode",
                "epochs",
                "validate_every",
                "batch_size",
                "eval_batch_size",
                "accum_steps",
                "num_workers",
                "seed",
                "max_train_updates",
                "smoke_steps",
                "eval_max_batches",
                "amp",
                "resume",
                "checkpoint_every_updates",
                "keep_epoch_checkpoints",
                "lead_chunk",
                "channels_last",
                "compile_model",
                "prithvi_activation_checkpointing",
                "fused_optimizer",
                "grad_scaler_init_scale",
                "check_output_equivalence",
                "official_root",
                "pretrained_checkpoint",
            }
            argv = []
            for key, value in values.items():
                if key not in accepted or value is None:
                    continue
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    argv.append(flag if value else "--no-" + key.replace("_", "-"))
                else:
                    argv.extend([flag, str(value)])
            with arguments(argv):
                return self.native.main()
        parser = (
            self.native.parser()
            if self.backend == "wildfire"
            else self.base.build_parser()
        )
        with arguments(parser_argv(parser, values)):
            if self.backend.startswith("floodcast"):
                return self.base.main()
            return self.native.main()

    def evaluate(self, max_batches=0, split="test", max_steps=0, protocol="source"):
        """Independent evaluation: no optimizer or training lifecycle is entered."""
        from src.data import evaluation_loader

        self.model.eval()
        loader = evaluation_loader(self, split, protocol)
        if self.backend == "landslide":
            return self.native.evaluate(
                self.model, loader, self.device, max_batches, self.args, 1
            )
        if self.backend == "wildfire":
            return self.native.evaluate(
                self.model,
                loader,
                self.native.OfficialBinaryDiceLoss(),
                self.device,
                self.config["evaluation"]["threshold"],
                max_batches,
            )[0]
        if self.backend.startswith("floodsim"):
            return self.base.evaluate_split(
                self.model,
                loader,
                self.normalizer,
                self.device,
                self.args,
                max_steps=max_steps,
                max_sequences=max_batches,
            )
        results = {
            name: self.base.evaluate_domain(
                self.model,
                ds,
                self.normalizer,
                self.device,
                self.base.EVALUATION_AMP_ENABLED,
                max_steps=max_steps,
                max_sequences=max_batches,
            )
            for name, ds in loader.items()
        }
        if protocol == "source":
            return dict(domains=results, macro=self.base.macro_average(results))
        summary = dict(
            source_key="mozambique_576",
            excluded_source_key="mozambique_1728",
            aggregation="equal_mean_over_contiguous_trajectories",
            debug_max_steps=max_steps,
            protocols={},
        )
        for name, result in results.items():
            states = 288 if name == "1day" else 576
            summary["protocols"][name] = dict(
                domain=result["domain"],
                states_per_trajectory=states,
                prediction_steps_per_trajectory=(
                    min(max_steps, states - 1) if max_steps else states - 1
                ),
                trajectory_count=result["sequence_count"],
                overall_equal_trajectory_mean=result["overall"],
            )
        return dict(domains=results, summary=summary)
