"""Three-source-domain split policy for the seven-baseline benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline


TRAIN_DOMAINS = ("australia", "pakistan", "uk_30m")
VALIDATION_KEYS = ("australia_valid", "pakistan_valid")
SOURCE_TEST_KEYS = ("australia_test", "pakistan_test")
OOD_SOURCE_KEYS = ("mozambique_576",)
OOD_PROTOCOL_KEYS = ("mozambique_1day", "mozambique_2day")
EXPECTED_KEYS = set(
    TRAIN_DOMAINS + VALIDATION_KEYS + SOURCE_TEST_KEYS + OOD_SOURCE_KEYS
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def protocol_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def make_three_domain_specs(
    args: Any,
    reference_factory: Callable[[Any], Dict[str, baseline.DomainSpec]],
) -> Dict[str, baseline.DomainSpec]:
    """Build the reference specs, then remove only the exact ``uk`` key."""

    specs = dict(reference_factory(args))
    original_train = tuple(
        key for key, spec in specs.items() if spec.role == "train"
    )
    if original_train != ("australia", "pakistan", "uk_30m", "uk"):
        raise RuntimeError(
            "reference training domains changed; expected "
            "('australia','pakistan','uk_30m','uk'), got "
            f"{original_train}"
        )
    removed = specs.pop("uk", None)
    if removed is None or removed.name != "uk" or removed.role != "train":
        raise RuntimeError("failed to remove the exact reference UK training spec")
    removed_mozambique_1728 = specs.pop("mozambique_1728", None)
    if (
        removed_mozambique_1728 is None
        or removed_mozambique_1728.role != "zero_shot"
    ):
        raise RuntimeError("failed to remove the Mozambique/1728 OOD spec")
    if set(specs) != EXPECTED_KEYS:
        raise RuntimeError(
            f"filtered split keys {sorted(specs)} != expected {sorted(EXPECTED_KEYS)}"
        )
    actual_train = tuple(
        key for key, spec in specs.items() if spec.role == "train"
    )
    if actual_train != TRAIN_DOMAINS or len(actual_train) != 3:
        raise RuntimeError(f"training domains must be exactly {TRAIN_DOMAINS}")
    if "uk" in specs or "uk_30m" not in specs:
        raise AssertionError("exact UK exclusion/UK_30 retention failed")

    baseline.validate_data_provenance(specs)
    split = {
        "format_version": 1,
        "policy": "reference_directory_splits_with_exact_uk_train_exclusion",
        "train": [asdict(specs[key]) for key in TRAIN_DOMAINS],
        "validation": [asdict(specs[key]) for key in VALIDATION_KEYS],
        "source_test": [asdict(specs[key]) for key in SOURCE_TEST_KEYS],
        "ood_test": [
            {
                "protocol": "mozambique_1day",
                "source": asdict(specs["mozambique_576"]),
                "states_per_trajectory": 288,
                "prediction_steps_per_trajectory": 287,
                "trajectory_count": 6,
                "aggregation": "equal_mean_over_six_contiguous_day_segments",
            },
            {
                "protocol": "mozambique_2day",
                "source": asdict(specs["mozambique_576"]),
                "states_per_trajectory": 576,
                "prediction_steps_per_trajectory": 575,
                "trajectory_count": 3,
                "aggregation": "equal_mean_over_three_original_trajectories",
            },
        ],
        "excluded": {
            "key": "uk",
            "path": removed.path,
            "reason": "requested exact training-domain exclusion",
        },
        "excluded_ood_source": {
            "key": "mozambique_1728",
            "path": removed_mozambique_1728.path,
            "reason": "one-/two-day OOD protocols use Mozambique/576 only",
        },
        "normalization_domains": list(TRAIN_DOMAINS),
        "checkpoint_selection_domains": ["australia", "pakistan"],
    }
    split["sha256"] = protocol_hash(split)
    output = Path(args.output_dir).expanduser().resolve() / "data_protocol"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, ensure_ascii=False)
    print("SEVEN_BASELINE_SPLIT", json.dumps(split, ensure_ascii=False))
    setattr(args, "data_split_sha256", split["sha256"])
    return specs


def window_protocol(args: Any) -> Dict[str, Any]:
    protocol = {
        "train_domains": list(TRAIN_DOMAINS),
        "validation_keys": list(VALIDATION_KEYS),
        "source_test_keys": list(SOURCE_TEST_KEYS),
        "ood_source_keys": list(OOD_SOURCE_KEYS),
        "ood_protocol_keys": list(OOD_PROTOCOL_KEYS),
        "training_crop": "deterministic_uniform_random",
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "window_seed": int(args.window_seed),
        "train_windows_per_epoch": int(args.train_windows_per_epoch),
        "tile_batch_size": int(args.tile_batch_size),
        "blend": "clipped_hann",
        "formal_validation": "full_grid_tiled",
        "formal_validation_every_epochs": int(args.validate_every),
        "formal_validation_horizon_steps": int(args.validation_max_steps),
        "formal_validation_horizon": "complete_sequence_to_last_state",
        "formal_validation_states_per_sequence": 288,
        "formal_validation_prediction_steps_per_sequence": 287,
        "formal_validation_checkpoint_selection": True,
        "formal_validation_early_stopping": True,
        "early_stopping_patience_checks": int(args.early_stopping),
        "source_test_horizon_steps": int(args.eval_max_steps),
        "source_test_horizon": "complete_sequence_to_last_state",
        "source_test_states_per_sequence": 288,
        "source_test_prediction_steps_per_sequence": 287,
        "mozambique_ood_protocol": {
            "source_key": "mozambique_576",
            "excluded_source_key": "mozambique_1728",
            "one_day": {
                "states_per_trajectory": 288,
                "prediction_steps_per_trajectory": 287,
                "trajectory_count": 6,
            },
            "two_day": {
                "states_per_trajectory": 576,
                "prediction_steps_per_trajectory": 575,
                "trajectory_count": 3,
            },
        },
        "split_sha256": getattr(args, "data_split_sha256", None),
    }
    protocol["sha256"] = protocol_hash(protocol)
    return protocol
