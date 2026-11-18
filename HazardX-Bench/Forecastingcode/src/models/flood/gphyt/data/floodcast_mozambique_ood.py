"""Audited one-day and two-day Mozambique OOD evaluation protocols.

The published ``Mozambique/576`` split contains three tensors with 576 states
each.  The one-day protocol splits every tensor into two non-overlapping,
contiguous 288-state trajectories and averages the six trajectory metrics.
The two-day protocol evaluates the three original 576-state trajectories.
Targets are used only after each prediction for metrics; every trajectory uses
the installed strict full-grid tiled autoregressive rollout.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline


ONE_DAY_STATES = 288
TWO_DAY_STATES = 576
EXPECTED_SOURCE_FILES = 3
EXPECTED_ONE_DAY_SEGMENTS = 6


def _slice_optional(values: Optional[Any], start: int, end: int) -> Optional[Any]:
    if values is None:
        return None
    try:
        return values[start:end]
    except (IndexError, TypeError) as exc:
        raise RuntimeError(
            f"cannot slice sequence metadata over [{start}:{end}]"
        ) from exc


def _slice_sequence(
    sequence: baseline.LoadedSequence,
    *,
    start: int,
    end: int,
    label: str,
) -> baseline.LoadedSequence:
    if not 0 <= start < end <= int(sequence.data.shape[0]):
        raise ValueError(
            f"invalid Mozambique segment [{start}:{end}] for "
            f"{tuple(sequence.data.shape)}"
        )
    data = sequence.data[start:end]
    if int(data.shape[0]) != ONE_DAY_STATES:
        raise AssertionError(f"one-day segment has {data.shape[0]} states")
    mask = None if sequence.mask is None else sequence.mask[start:end]
    return baseline.LoadedSequence(
        data=data,
        mask=mask,
        timestamps=_slice_optional(sequence.timestamps, start, end),
        dx=sequence.dx,
        dy=sequence.dy,
        source_path=f"{sequence.source_path}::{label}",
        raw_keys=sequence.raw_keys,
    )


class SegmentedMozambiqueDataset:
    """Minimal dataset interface consumed by ``baseline.evaluate_domain``."""

    def __init__(
        self,
        source: baseline.FloodDomainDataset,
        sequences: Sequence[baseline.LoadedSequence],
        *,
        protocol_name: str,
        states_per_segment: int,
    ) -> None:
        self.spec = dataclasses.replace(source.spec, name=protocol_name)
        self.normalizer = source.normalizer
        self._sequences = list(sequences)
        self.files = [Path(item.source_path) for item in self._sequences]
        self.time_length = int(states_per_segment)
        self.pairs_per_sequence = self.time_length - 1
        self.profile = dataclasses.replace(
            source.profile,
            path=f"virtual://{protocol_name}",
            tensor_shape=(
                self.time_length,
                source.profile.height,
                source.profile.width,
                source.profile.channels,
            ),
            time_length=self.time_length,
        )
        if not self._sequences:
            raise ValueError(f"{protocol_name}: no evaluation sequences")
        for sequence in self._sequences:
            if int(sequence.data.shape[0]) != self.time_length:
                raise AssertionError(
                    f"{protocol_name}: mixed segment lengths "
                    f"{sequence.data.shape[0]} and {self.time_length}"
                )

    def __len__(self) -> int:
        return len(self._sequences) * self.pairs_per_sequence

    def get_sequence(self, index: int) -> baseline.LoadedSequence:
        return self._sequences[index]


def build_mozambique_protocol_datasets(
    args: Any,
    specs: Dict[str, baseline.DomainSpec],
    normalizer: baseline.ChannelNormalizer,
) -> Dict[str, SegmentedMozambiqueDataset]:
    if "mozambique_576" not in specs or "mozambique_1728" in specs:
        raise RuntimeError(
            "OOD specs must contain only Mozambique/576 as the protocol source"
        )
    source = baseline.FloodDomainDataset(specs["mozambique_576"], normalizer)
    baseline.check_timestamp_continuity(source)
    if len(source.files) != EXPECTED_SOURCE_FILES:
        raise RuntimeError(
            f"Mozambique/576 must contain {EXPECTED_SOURCE_FILES} files, "
            f"found {len(source.files)}"
        )
    if source.time_length != TWO_DAY_STATES:
        raise RuntimeError(
            f"Mozambique/576 must contain {TWO_DAY_STATES} states per file, "
            f"found {source.time_length}"
        )

    two_day_sequences = [source.get_sequence(index) for index in range(len(source.files))]
    one_day_sequences: List[baseline.LoadedSequence] = []
    for file_index, sequence in enumerate(two_day_sequences):
        for day_index, start in enumerate((0, ONE_DAY_STATES)):
            one_day_sequences.append(
                _slice_sequence(
                    sequence,
                    start=start,
                    end=start + ONE_DAY_STATES,
                    label=f"file_{file_index:02d}_day_{day_index + 1:02d}",
                )
            )
    if len(one_day_sequences) != EXPECTED_ONE_DAY_SEGMENTS:
        raise AssertionError("one-day Mozambique split must produce six segments")

    return {
        "1day": SegmentedMozambiqueDataset(
            source,
            one_day_sequences,
            protocol_name="mozambique_1day",
            states_per_segment=ONE_DAY_STATES,
        ),
        "2day": SegmentedMozambiqueDataset(
            source,
            two_day_sequences,
            protocol_name="mozambique_2day",
            states_per_segment=TWO_DAY_STATES,
        ),
    }


def run_mozambique_day_test(
    checkpoint_path: str | Path,
    args: Any,
    specs: Dict[str, baseline.DomainSpec],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate fixed one-/two-day OOD protocols from Mozambique/576 only."""

    model, normalizer, checkpoint = baseline.load_eval_model(checkpoint_path, device)
    selection_domains = set(checkpoint["checkpoint_selection"]["domains"])
    if selection_domains != {"australia", "pakistan"}:
        raise RuntimeError(
            "Mozambique OOD requires an Australia/Pakistan validation-selected "
            f"checkpoint, got {sorted(selection_domains)}"
        )
    datasets = build_mozambique_protocol_datasets(args, specs, normalizer)
    selected = {
        "1day": ("1day",),
        "2day": ("2day",),
        "both": ("1day", "2day"),
    }[args.mozambique_split]
    debug_steps = int(getattr(args, "ood_debug_max_steps", 0))
    if debug_steps < 0:
        raise ValueError("--ood-debug-max-steps cannot be negative")

    output_dir = baseline.canonical_path(args.output_dir) / "mozambique_zero_shot"
    results: Dict[str, Any] = {}
    summary: Dict[str, Any] = {
        "source_key": "mozambique_576",
        "source_path": specs["mozambique_576"].path,
        "excluded_source_key": "mozambique_1728",
        "aggregation": "equal_mean_over_contiguous_trajectories",
        "debug_max_steps": debug_steps,
        "protocols": {},
    }
    for protocol in selected:
        dataset = datasets[protocol]
        result = baseline.evaluate_domain(
            model=model,
            dataset=dataset,
            normalizer=normalizer,
            device=device,
            amp=baseline.EVALUATION_AMP_ENABLED,
            max_steps=debug_steps,
            max_sequences=0,
        )
        expected_segments = 6 if protocol == "1day" else 3
        expected_states = ONE_DAY_STATES if protocol == "1day" else TWO_DAY_STATES
        if result["sequence_count"] != expected_segments:
            raise AssertionError(
                f"{protocol}: {result['sequence_count']} trajectories, "
                f"expected {expected_segments}"
            )
        expected_predictions = (
            min(debug_steps, expected_states - 1)
            if debug_steps > 0
            else expected_states - 1
        )
        if len(result["lead_metrics"]) != expected_predictions:
            raise AssertionError(
                f"{protocol}: {len(result['lead_metrics'])} leads, "
                f"expected {expected_predictions}"
            )
        results[protocol] = result
        baseline.write_evaluation(result, output_dir)
        summary["protocols"][protocol] = {
            "domain": result["domain"],
            "states_per_trajectory": expected_states,
            "prediction_steps_per_trajectory": expected_predictions,
            "trajectory_count": expected_segments,
            "overall_equal_trajectory_mean": result["overall"],
        }
        print(
            "MOZAMBIQUE_OOD_PROTOCOL",
            json.dumps(
                baseline.json_safe(summary["protocols"][protocol]),
                ensure_ascii=False,
            ),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "one_two_day_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(baseline.json_safe(summary), handle, indent=2, ensure_ascii=False)
    return {"domains": results, "summary": summary}

