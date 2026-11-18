"""Isolated centered-DEM Transolver experiments for FloodCastBench.

This module leaves every existing experiment untouched and reuses the
validated published-DEM reconstruction in ``floodcast_dem_sensitivity_common``.
Both entry points built on this module use exactly the same data pipeline:

dynamic
    [global-train-standard h, global-train-standard rainfall]

static
    [published DEM minus its region mean, divided by one source-only scale]

The shared terrain scale is the RMS after per-region centering, with Australia,
Pakistan, UK_30 and UK receiving equal 1/4 weights.  Mozambique is transformed
with its own mean and the source-only shared scale, but never contributes to
fitted statistics.

Two loss kinds are supported:

``mse``
    The original per-sample masked MSE.

``masked_huber_mse_matched``
    A physical-threshold Huber evaluated in normalized water-depth space.  Its
    quadratic branch is exactly MSE, while its linear branch is
    ``2*delta*abs(error) - delta**2``.  Thus this is twice the conventional
    Huber definition and preserves the original small-error gradient scale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_Transolver_MSE_fourier_ape as transolver
import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common


DEM_KIND = "terrain_centered_global_scaled"
LOSS_MSE = "mse"
LOSS_HUBER = "masked_huber_mse_matched"
LOSS_KINDS = (LOSS_MSE, LOSS_HUBER)
HUBER_DELTA_M_DEFAULT = 1.0

EXPERIMENT_NAMES = {
    LOSS_MSE: (
        "FloodCastBench_Transolver_MSE_PhysicalFourierAPE_"
        "DEM_Centered_GlobalScaled"
    ),
    LOSS_HUBER: (
        "FloodCastBench_Transolver_Huber_PhysicalFourierAPE_"
        "DEM_Centered_GlobalScaled"
    ),
}

_BASE_VALIDATE_ARGS = baseline.validate_args
_BASE_COMPUTE_NORMALIZER = baseline.compute_training_normalizer
_BASE_LOAD_CHECKPOINT = baseline.load_checkpoint_file
_BASE_LOAD_EVAL_MODEL = baseline.load_eval_model
_BASE_DESCRIBE_DATASET = baseline.describe_dataset
_BASE_MASKED_LOSS = baseline.masked_loss
_BASE_CHECKPOINT_PAYLOAD = baseline.checkpoint_payload
_BASE_MODEL_SUMMARY = transolver.model_summary

_ACTIVE_STORE: Optional[dem_common.DemFeatureStore] = None
_ACTIVE_LOSS_KIND: Optional[str] = None
_ACTIVE_HUBER_DELTA_M: Optional[float] = None
_ACTIVE_TRAIN_WATER_STD: Optional[float] = None
_ACTIVE_HUBER_DELTA_NORM: Optional[float] = None


def _require_store() -> dem_common.DemFeatureStore:
    if _ACTIVE_STORE is None:
        raise RuntimeError("Centered DEM feature store has not been configured")
    return _ACTIVE_STORE


def _configure_store(dem_root: str | Path) -> dem_common.DemFeatureStore:
    global _ACTIVE_STORE
    root = Path(dem_root).expanduser().resolve()
    if (
        _ACTIVE_STORE is None
        or _ACTIVE_STORE.kind != DEM_KIND
        or _ACTIVE_STORE.dem_root != root
    ):
        _ACTIVE_STORE = dem_common.DemFeatureStore(DEM_KIND, root)

    # The generic, already-tested Dataset and rollout resolve their static
    # tensor through these process-local hooks.
    dem_common._ACTIVE_STORE = _ACTIVE_STORE
    dem_common._ACTIVE_KIND = DEM_KIND
    return _ACTIVE_STORE


def _set_huber_scale(train_water_std: float, delta_m: float) -> None:
    global _ACTIVE_TRAIN_WATER_STD, _ACTIVE_HUBER_DELTA_NORM
    if not math.isfinite(train_water_std) or train_water_std <= 0.0:
        raise ValueError(f"Invalid train water std {train_water_std}")
    if not math.isfinite(delta_m) or delta_m <= 0.0:
        raise ValueError(f"Invalid physical Huber delta {delta_m}")
    _ACTIVE_TRAIN_WATER_STD = float(train_water_std)
    _ACTIVE_HUBER_DELTA_NORM = float(delta_m) / float(train_water_std)


def _compute_training_normalizer(
    train_specs: Mapping[str, baseline.DomainSpec],
) -> baseline.ChannelNormalizer:
    """Keep baseline h/rain stats and replace only the static terrain slot."""
    original = _BASE_COMPUTE_NORMALIZER(train_specs)
    provenance = dict(original.provenance)
    provenance["transolver_centered_dem"] = _require_store().metadata()
    provenance["channel_policy"] = {
        "water_depth": "unchanged_baseline_global_train_standard",
        "rainfall": "unchanged_baseline_global_train_standard",
        "terrain_slot": (
            "identity_placeholder; published physical DEM is centered per "
            "region and divided by a source-only equal-domain RMS"
        ),
    }
    normalizer = baseline.ChannelNormalizer(
        mode=original.mode,
        mean=(original.mean[0], original.mean[1], 0.0),
        std=(original.std[0], original.std[1], 1.0),
        provenance=provenance,
    )
    if _ACTIVE_LOSS_KIND == LOSS_HUBER:
        if _ACTIVE_HUBER_DELTA_M is None:
            raise RuntimeError("Huber physical threshold is not configured")
        _set_huber_scale(
            train_water_std=float(normalizer.std[baseline.WATER_CHANNEL]),
            delta_m=_ACTIVE_HUBER_DELTA_M,
        )
    return normalizer


def masked_huber_mse_matched(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta_norm: float,
) -> torch.Tensor:
    """Per-sample masked, MSE-matched Huber followed by equal sample mean."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise AssertionError(
            f"loss shape mismatch pred={prediction.shape}, "
            f"target={target.shape}, mask={mask.shape}"
        )
    if not math.isfinite(delta_norm) or delta_norm <= 0.0:
        raise ValueError(f"delta_norm must be finite and positive, got {delta_norm}")

    prediction_f = prediction.float()
    target_f = target.float()
    valid = (
        mask.bool()
        & torch.isfinite(prediction_f)
        & torch.isfinite(target_f)
    )
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("At least one sample has zero valid target pixels")

    error = torch.where(
        valid,
        prediction_f - target_f,
        torch.zeros_like(prediction_f),
    )
    abs_error = error.abs()
    delta = torch.as_tensor(
        delta_norm,
        dtype=error.dtype,
        device=error.device,
    )
    point_loss = torch.where(
        abs_error <= delta,
        error.square(),
        2.0 * delta * abs_error - delta.square(),
    )
    per_sample = point_loss.flatten(1).sum(dim=1) / counts
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked Huber loss")
    return loss


def _huber_masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Drop-in training/smoke hook for the baseline hard-coded MSE call."""
    del eps
    if kind != "mse":
        raise ValueError(
            "This isolated Huber experiment expects the baseline training "
            f"loop to request its original MSE slot, got {kind!r}"
        )
    if _ACTIVE_HUBER_DELTA_NORM is None:
        raise RuntimeError(
            "Huber normalized threshold is unavailable; training normalizer "
            "must be computed before the first loss call"
        )
    return masked_huber_mse_matched(
        prediction,
        target,
        mask,
        delta_norm=_ACTIVE_HUBER_DELTA_NORM,
    )


def _describe_dataset(
    dataset: dem_common.DemFeatureFloodDomainDataset,
    batch_size: Optional[int],
) -> Dict[str, Any]:
    description = _BASE_DESCRIBE_DATASET(dataset, batch_size)
    store = _require_store()
    description.update(
        {
            "static_source": "published_physical_dem_tiff_not_fm_channel_2",
            "static_feature_names": list(dem_common.STATIC_NAMES[DEM_KIND]),
            "static_channels": store.static_channels,
            "terrain_global_train_scale": store.terrain_global_train_scale,
        }
    )
    print(
        json.dumps(
            {
                "domain": dataset.spec.name,
                "static_source": description["static_source"],
                "static_feature_names": description["static_feature_names"],
                "static_shape": list(dataset.dem_feature.shape),
                "terrain_global_train_scale": (
                    description["terrain_global_train_scale"]
                ),
            },
            ensure_ascii=False,
        )
    )
    return description


def _validate_dem_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    saved = (
        checkpoint.get("normalization", {})
        .get("provenance", {})
        .get("transolver_centered_dem")
    )
    if not isinstance(saved, Mapping):
        raise RuntimeError(
            "Checkpoint has no centered-DEM metadata; existing baseline "
            "checkpoints are intentionally incompatible"
        )
    current = _require_store().metadata()
    if saved.get("kind") != DEM_KIND:
        raise RuntimeError(
            f"Checkpoint DEM kind {saved.get('kind')!r} != {DEM_KIND!r}"
        )
    if saved.get("static_feature_names") != current["static_feature_names"]:
        raise RuntimeError("Checkpoint centered-DEM channels do not match")
    saved_scale = float(saved.get("terrain_global_train_scale"))
    current_scale = float(current["terrain_global_train_scale"])
    if not math.isclose(saved_scale, current_scale, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Centered-DEM scale changed: {saved_scale} != {current_scale}"
        )
    for region in dem_common.DEM_SPECS:
        saved_hash = saved["region_stats"][region]["feature_sha256"]
        current_hash = current["region_stats"][region]["feature_sha256"]
        if saved_hash != current_hash:
            raise RuntimeError(
                f"{region}: current centered DEM differs from checkpoint"
            )


def _load_checkpoint_file(
    path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = _BASE_LOAD_CHECKPOINT(path, device)
    _validate_dem_checkpoint(checkpoint)

    saved_loss = checkpoint.get(
        "loss_kind",
        checkpoint.get("run_config", {}).get("loss"),
    )
    if saved_loss != _ACTIVE_LOSS_KIND:
        raise RuntimeError(
            f"Checkpoint loss {saved_loss!r} != running loss "
            f"{_ACTIVE_LOSS_KIND!r}"
        )
    if _ACTIVE_LOSS_KIND == LOSS_HUBER:
        delta_m = float(checkpoint.get("huber_delta_m", float("nan")))
        water_std = float(checkpoint.get("train_water_std", float("nan")))
        delta_norm = float(checkpoint.get("huber_delta_norm", float("nan")))
        expected_delta_m = float(_ACTIVE_HUBER_DELTA_M)
        expected_water_std = float(
            checkpoint["normalization"]["std"][baseline.WATER_CHANNEL]
        )
        expected_delta_norm = expected_delta_m / expected_water_std
        if not math.isclose(
            delta_m, expected_delta_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("Checkpoint physical Huber threshold mismatch")
        if not math.isclose(
            water_std, expected_water_std, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("Checkpoint water std metadata mismatch")
        if not math.isclose(
            delta_norm, expected_delta_norm, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("Checkpoint normalized Huber threshold mismatch")

        # Fresh training initializes the runtime loss scale while computing
        # the normalizer. Resume loads that normalizer from the checkpoint, so
        # restore the same process-local state explicitly after all checkpoint
        # metadata has passed validation.
        _set_huber_scale(train_water_std=water_std, delta_m=delta_m)

        # The original NaN-safe training loop reserves its ``mse`` slot for
        # the active masked loss implementation and rejects any other value
        # in run_config while resuming.  This experiment deliberately installs
        # the MSE-matched Huber implementation in that slot.  After validating
        # the checkpoint's real loss metadata above, present a shallow,
        # in-memory compatibility view to the legacy training loop.  The file
        # on disk is untouched, and newly written checkpoints still record the
        # user-facing Huber loss from the active CLI arguments.
        checkpoint = dict(checkpoint)
        checkpoint["run_config"] = dict(checkpoint.get("run_config", {}))
        checkpoint["run_config"]["loss"] = "mse"
    return checkpoint


def _huber_checkpoint_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _BASE_CHECKPOINT_PAYLOAD(*args, **kwargs)
    normalizer = kwargs.get("normalizer")
    if normalizer is None:
        raise RuntimeError("Checkpoint payload requires named normalizer")
    train_water_std = float(normalizer.std[baseline.WATER_CHANNEL])
    delta_m = float(_ACTIVE_HUBER_DELTA_M)
    delta_norm = delta_m / train_water_std
    payload.update(
        {
            "loss_kind": LOSS_HUBER,
            "huber_delta_m": delta_m,
            "huber_delta_norm": delta_norm,
            "train_water_std": train_water_std,
        }
    )
    return payload


def _load_huber_eval_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, baseline.ChannelNormalizer, Dict[str, Any]]:
    checkpoint = baseline.load_checkpoint_file(checkpoint_path, device)
    model = baseline.build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = baseline.ChannelNormalizer.from_state_dict(
        checkpoint["normalization"]
    )
    if normalizer.mode != "train_standard":
        raise ValueError("Huber evaluation requires train_standard")
    saved_loss = checkpoint.get(
        "loss_kind",
        checkpoint.get("run_config", {}).get("loss"),
    )
    if saved_loss != LOSS_HUBER:
        raise ValueError("Evaluation requires a centered-DEM Huber checkpoint")
    return model, normalizer, checkpoint


def _replace_loss_parser_choice(
    parser: argparse.ArgumentParser,
    loss_kind: str,
) -> None:
    for action in parser._actions:
        if action.dest == "loss":
            action.choices = (loss_kind,)
            action.default = loss_kind
            return
    raise RuntimeError("Transolver parser has no --loss action")


def run_experiment(
    *,
    loss_kind: str,
    default_output_dir: str,
) -> None:
    """Install isolated hooks and enter the original baseline lifecycle."""
    global _ACTIVE_LOSS_KIND, _ACTIVE_HUBER_DELTA_M
    if loss_kind not in LOSS_KINDS:
        raise ValueError(f"Unsupported centered-DEM loss {loss_kind!r}")
    _ACTIVE_LOSS_KIND = loss_kind

    def build_parser() -> argparse.ArgumentParser:
        parser = transolver.build_parser()
        parser.description = (
            "FloodCastBench residual-conditioned Transolver with published "
            "DEM centered per region and source-only equal-domain scaling; "
            f"loss={loss_kind}"
        )
        parser.add_argument(
            "--dem-root",
            default=dem_common.DEM_ROOT_DEFAULT,
            help="Published DEM directory containing event TIFFs",
        )
        if loss_kind == LOSS_HUBER:
            parser.add_argument(
                "--huber-delta-m",
                type=float,
                default=HUBER_DELTA_M_DEFAULT,
                help="Physical water-depth error threshold in metres",
            )
            _replace_loss_parser_choice(parser, LOSS_HUBER)
        parser.set_defaults(
            output_dir=default_output_dir,
            normalization="train_standard",
            loss=loss_kind,
            position_encoding=transolver.POSITION_ENCODING,
        )
        return parser

    def validate_args(args: argparse.Namespace) -> None:
        global _ACTIVE_HUBER_DELTA_M
        if args.loss != loss_kind:
            raise ValueError(
                f"This experiment requires --loss {loss_kind}, got {args.loss}"
            )
        if loss_kind == LOSS_HUBER:
            if not math.isfinite(args.huber_delta_m) or args.huber_delta_m <= 0:
                raise ValueError("--huber-delta-m must be finite and positive")
            # Reuse every baseline validation rule while locally presenting
            # its original MSE slot.  The user-facing run_config remains Huber.
            actual_loss = args.loss
            args.loss = "mse"
            try:
                _BASE_VALIDATE_ARGS(args)
            finally:
                args.loss = actual_loss
            _ACTIVE_HUBER_DELTA_M = float(args.huber_delta_m)
        else:
            _BASE_VALIDATE_ARGS(args)
        store = _configure_store(args.dem_root)
        if store.static_channels != 1:
            raise AssertionError("Centered DEM experiment must have one channel")

    def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
        config = transolver.model_config_from_args(args)
        config["experiment"] = EXPERIMENT_NAMES[loss_kind]
        return config

    def build_model(
        config: Mapping[str, Any],
        device: torch.device,
    ) -> nn.Module:
        model = transolver.build_model(config, device)
        if getattr(model, "input_channels", None) != 3:
            raise AssertionError(
                "Original Transolver tokenizer must remain [h,rain,terrain]"
            )
        return model

    def model_summary(model: nn.Module) -> Dict[str, Any]:
        summary = _BASE_MODEL_SUMMARY(model)
        summary.update(
            {
                "experiment": EXPERIMENT_NAMES[loss_kind],
                "dynamic_feature_names": ["h_norm", "rainfall_norm"],
                "static_feature_names": ["terrain_centered_global_scaled"],
                "terrain_source": "published_physical_dem_tiff",
                "loss": (
                    "FP32 per-sample masked MSE on normalized reconstructed "
                    "h_next"
                    if loss_kind == LOSS_MSE
                    else (
                        "FP32 per-sample masked MSE-matched Huber on normalized "
                        "reconstructed h_next; physical delta=1m by default"
                    )
                ),
            }
        )
        return summary

    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.compute_training_normalizer = _compute_training_normalizer
    baseline.FloodDomainDataset = dem_common.DemFeatureFloodDomainDataset
    baseline.rollout_sequence = dem_common.rollout_sequence
    baseline.describe_dataset = _describe_dataset
    baseline.load_checkpoint_file = _load_checkpoint_file
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    transolver.model_summary = model_summary

    if loss_kind == LOSS_HUBER:
        baseline.masked_loss = _huber_masked_loss
        baseline.checkpoint_payload = _huber_checkpoint_payload
        baseline.load_eval_model = _load_huber_eval_model
    else:
        baseline.masked_loss = _BASE_MASKED_LOSS
        baseline.checkpoint_payload = _BASE_CHECKPOINT_PAYLOAD
        baseline.load_eval_model = _BASE_LOAD_EVAL_MODEL

    baseline.main()

