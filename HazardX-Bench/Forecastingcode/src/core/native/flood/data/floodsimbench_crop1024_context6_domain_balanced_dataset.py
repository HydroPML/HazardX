"""Domain-balanced Train view of the FloodSimBench context-6 crop dataset.

All physical preprocessing, normalization, evaluation tiling, and forecasting
sample construction are delegated to ``floodsimbench_crop1024_context6_dataset``.
The only data-policy change is the Train view: every epoch exposes exactly one
scenario from each Train domain, selected by a deterministic without-replacement
queue, and all 67 forecasts from that scenario share one deterministic-random
1024 x 1024 crop.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data import floodsimbench_crop1024_context6_dataset as reference


DEFAULT_ROOT = reference.DEFAULT_ROOT
TRAIN_CITIES = reference.TRAIN_CITIES
VALID_CITIES = reference.VALID_CITIES
TEST_CITIES = reference.TEST_CITIES
ALL_CITIES = reference.ALL_CITIES
FRAMES_PER_SCENARIO = reference.FRAMES_PER_SCENARIO
DT_MINUTES = reference.DT_MINUTES
CROP_SIZE = reference.CROP_SIZE
EVAL_STRIDE = reference.EVAL_STRIDE
DEPTH_HISTORY_STEPS = reference.DEPTH_HISTORY_STEPS
RAINFALL_CONTEXT_STEPS = reference.RAINFALL_CONTEXT_STEPS
FORECAST_SAMPLES_PER_SEQUENCE = reference.FORECAST_SAMPLES_PER_SEQUENCE
INPUT_CHANNELS = reference.INPUT_CHANNELS
DEPTH_CLEAN_THRESHOLD_M = reference.DEPTH_CLEAN_THRESHOLD_M
GRID_SPACING_M = reference.GRID_SPACING_M
CHANNEL_ORDER = reference.CHANNEL_ORDER
PREPROCESSING_VERSION = (
    "floodsimbench_1m_context6_domain_balanced_random_train_"
    "fixed_eval_crop1024_v1"
)

SpatialGeometry = reference.SpatialGeometry
EvalTileSequence = reference.EvalTileSequence
LoadedTrainTile = reference.LoadedTrainTile
configure_terrain_store = reference.configure_terrain_store
require_terrain_store = reference.require_terrain_store
forecasting_frame_indices = reference.forecasting_frame_indices
rainfall_at_frame_mm_h = reference.rainfall_at_frame_mm_h
homogeneous_collate = reference.homogeneous_collate


def _stable_domain_salt(domain: str) -> int:
    """Return a process-independent integer; never use Python's salted hash."""
    digest = hashlib.blake2b(domain.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class FloodSimBenchContext6Dataset(reference.FloodSimBenchContext6Dataset):
    """Reference Dataset with a 17-scenario domain-balanced Train view."""

    def __init__(
        self,
        root: str | Path,
        cities: Sequence[str],
        role: str,
        normalizer: Any,
        *,
        seed: int = 1,
    ) -> None:
        super().__init__(root, cities, role, normalizer, seed=seed)
        self._domain_order: Tuple[str, ...] = ()
        self._domain_scenario_indices: Dict[str, Tuple[int, ...]] = {}
        if self.role == "train":
            grouped: Dict[str, List[int]] = defaultdict(list)
            for scenario_index, scenario in enumerate(self.scenarios):
                grouped[str(scenario.domain)].append(scenario_index)
            self._domain_order = tuple(sorted(grouped))
            self._domain_scenario_indices = {
                domain: tuple(
                    sorted(
                        indices,
                        key=lambda index: (
                            str(self.scenarios[index].scenario_id),
                            float(self.scenarios[index].rainfall_mm),
                        ),
                    )
                )
                for domain, indices in grouped.items()
            }
            if len(self._domain_order) != 17:
                raise ValueError(
                    "Domain-balanced Train requires 17 domains, got "
                    f"{len(self._domain_order)}: {self._domain_order}"
                )
            if sum(map(len, self._domain_scenario_indices.values())) != 122:
                raise ValueError("Domain-balanced Train requires 122 scenarios")

    @property
    def domain_order(self) -> Tuple[str, ...]:
        if self.role != "train":
            raise RuntimeError("domain_order is Train-only")
        return self._domain_order

    @property
    def domain_scenario_counts(self) -> Dict[str, int]:
        if self.role != "train":
            raise RuntimeError("domain_scenario_counts is Train-only")
        return {
            domain: len(self._domain_scenario_indices[domain])
            for domain in self._domain_order
        }

    def _queue(self, domain: str, cycle: int) -> Tuple[int, ...]:
        candidates = list(self._domain_scenario_indices[domain])
        rng = random.Random(
            self.seed
            + _stable_domain_salt(domain)
            + 1_000_003 * int(cycle)
            + 104_729
        )
        rng.shuffle(candidates)
        return tuple(candidates)

    def selected_scenario_indices(self, epoch: Optional[int] = None) -> Tuple[int, ...]:
        """One scenario per domain, as a pure function of seed and epoch."""
        if self.role != "train":
            raise RuntimeError("selected_scenario_indices is Train-only")
        epoch_value = self.epoch if epoch is None else int(epoch)
        if epoch_value < 0:
            raise ValueError("epoch must be non-negative")
        selected: List[int] = []
        for domain in self._domain_order:
            queue_length = len(self._domain_scenario_indices[domain])
            cycle, offset = divmod(epoch_value, queue_length)
            selected.append(self._queue(domain, cycle)[offset])
        if len(selected) != len(set(selected)):
            raise AssertionError("One epoch selected a scenario more than once")
        return tuple(selected)

    def selected_epoch_metadata(self, epoch: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.role != "train":
            raise RuntimeError("selected_epoch_metadata is Train-only")
        epoch_value = self.epoch if epoch is None else int(epoch)
        rows: List[Dict[str, Any]] = []
        for domain, scenario_index in zip(
            self._domain_order, self.selected_scenario_indices(epoch_value)
        ):
            scenario = self.scenarios[scenario_index]
            queue_length = len(self._domain_scenario_indices[domain])
            cycle, offset = divmod(epoch_value, queue_length)
            crop_row, crop_col = self.random_coordinate(scenario_index, epoch_value)
            rows.append(
                {
                    "domain": domain,
                    "scenario_index": scenario_index,
                    "scenario_id": scenario.scenario_id,
                    "rainfall_mm": float(scenario.rainfall_mm),
                    "queue_length": queue_length,
                    "queue_cycle": cycle,
                    "queue_offset": offset,
                    "row_start": crop_row,
                    "col_start": crop_col,
                }
            )
        return rows

    @property
    def sequence_count(self) -> int:
        if self.role == "train":
            return len(self._domain_order)
        return len(self.files)

    def __len__(self) -> int:
        return self.sequence_count * FORECAST_SAMPLES_PER_SEQUENCE

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.role != "train":
            return super().__getitem__(index)
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        local_sequence_index = int(index) // FORECAST_SAMPLES_PER_SEQUENCE
        forecast_index = int(index) % FORECAST_SAMPLES_PER_SEQUENCE
        scenario_index = self.selected_scenario_indices()[local_sequence_index]
        scenario = self.scenarios[scenario_index]
        loaded = self._load_train_sequence(scenario_index)
        return self._assemble(
            scenario,
            loaded.depths,
            loaded.terrain,
            loaded.valid,
            forecast_index,
            loaded.row_start,
            loaded.col_start,
            f"{scenario.scenario_id}_domain_balanced_epoch{loaded.epoch}",
        )


def make_channel_normalizer(
    normalizer_class: Any,
    root: str | Path = DEFAULT_ROOT,
    cache_path: Optional[str | Path] = None,
) -> Any:
    """Reuse the exact full-Train statistics from the reference experiment."""
    normalizer = reference.make_channel_normalizer(
        normalizer_class, root, cache_path
    )
    normalizer.provenance = dict(normalizer.provenance)
    normalizer.provenance.update(
        {
            "dataset_variant": "FloodSimBench_context6_crop1024_domain_balanced",
            "preprocessing_version": PREPROCESSING_VERSION,
            "normalization_reuse": (
                "exact reference statistics from all 122 Train scenarios; "
                "never recomputed from the 17 sampled scenarios"
            ),
        }
    )
    return normalizer


def split_inventory(root: str | Path = DEFAULT_ROOT) -> Dict[str, Any]:
    inventory = reference.split_inventory(root)
    train = inventory["splits"]["train"]
    train["available_scenarios"] = train["scenario_count"]
    train["selected_domains_per_epoch"] = 17
    train["selected_scenarios_per_epoch"] = 17
    train["forecast_samples_per_epoch"] = 17 * FORECAST_SAMPLES_PER_SEQUENCE
    inventory["preprocessing_version"] = PREPROCESSING_VERSION
    inventory["train_crop"] = (
        "one deterministic-random full crop for one queued scenario per domain "
        "per epoch"
    )
    inventory["train_sampling"] = (
        "domain-balanced deterministic without-replacement queues"
    )
    return inventory

