"""FloodSimBench 1 m context-6 forecasting tiles for UNet/FNO baselines.

Training uses one deterministic-random 1024 x 1024 crop per scenario and
epoch.  All 67 forecasting samples from that scenario/epoch share the same
crop.  Validation and test use deterministic, non-overlapping 1024 grids.
Only a dimension smaller than 1024 is bottom/right padded; ordinary remainder
is discarded.  No spatial augmentation, resize, downsampling, or stitching is
performed.

Physical channel order is strictly::

    R[t-5], ..., R[t], DEM, h[t-5], ..., h[t-1] -> h[t]

Rainfall is mm/h before Train-only standardization.  Depth and DEM processing
reuse the audited FloodSimBench crop1024 implementation.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data import floodsimbench_markov_dataset as base
from data import floodsimbench_markov_dataset_crop1024 as crop


DEFAULT_ROOT = base.DEFAULT_ROOT
TRAIN_CITIES = base.TRAIN_CITIES
VALID_CITIES = base.VALID_CITIES
TEST_CITIES = base.TEST_CITIES
ALL_CITIES = base.ALL_CITIES
FRAMES_PER_SCENARIO = base.FRAMES_PER_SCENARIO
DT_MINUTES = base.DT_MINUTES

CROP_SIZE = 1024
EVAL_STRIDE = 1024
DEPTH_HISTORY_STEPS = 5
RAINFALL_CONTEXT_STEPS = 6
FORECAST_SAMPLES_PER_SEQUENCE = FRAMES_PER_SCENARIO - DEPTH_HISTORY_STEPS
INPUT_CHANNELS = RAINFALL_CONTEXT_STEPS + 1 + DEPTH_HISTORY_STEPS
DEPTH_CLEAN_THRESHOLD_M = crop.DEPTH_CLEAN_THRESHOLD_M
GRID_SPACING_M = 1.0
PREPROCESSING_VERSION = "floodsimbench_1m_context6_random_train_fixed_eval_crop1024_v1"
CHANNEL_ORDER = (
    "R[t-5]", "R[t-4]", "R[t-3]", "R[t-2]", "R[t-1]", "R[t]",
    "DEM",
    "h[t-5]", "h[t-4]", "h[t-3]", "h[t-2]", "h[t-1]",
)

_ACTIVE_STORE: Optional[base.FloodSimTerrainStore] = None


@dataclass(frozen=True)
class SpatialGeometry:
    domain: str
    height: int
    width: int

    @property
    def padded_height(self) -> int:
        return max(self.height, CROP_SIZE)

    @property
    def padded_width(self) -> int:
        return max(self.width, CROP_SIZE)

    @property
    def pads_bottom(self) -> bool:
        return self.height < CROP_SIZE

    @property
    def pads_right(self) -> bool:
        return self.width < CROP_SIZE

    def fixed_coordinates(self) -> Tuple[Tuple[int, int], ...]:
        rows = tuple(
            range(0, self.padded_height - CROP_SIZE + 1, EVAL_STRIDE)
        )
        cols = tuple(
            range(0, self.padded_width - CROP_SIZE + 1, EVAL_STRIDE)
        )
        if not rows or not cols:
            raise AssertionError(f"{self.domain}: fixed grid is empty")
        return tuple((row, col) for row in rows for col in cols)

    def metadata(self) -> Dict[str, Any]:
        coordinates = self.fixed_coordinates()
        covered_h = coordinates[-1][0] + CROP_SIZE
        covered_w = max(col for _, col in coordinates) + CROP_SIZE
        return {
            "domain": self.domain,
            "raw_shape": [self.height, self.width],
            "padded_shape": [self.padded_height, self.padded_width],
            "padding_bottom": max(CROP_SIZE - self.height, 0),
            "padding_right": max(CROP_SIZE - self.width, 0),
            "padding_reason": "only_dimension_smaller_than_1024",
            "fixed_grid_stride": EVAL_STRIDE,
            "fixed_grid_overlap": 0,
            "fixed_tile_count": len(coordinates),
            "fixed_coordinates": [list(item) for item in coordinates],
            "discarded_bottom_remainder": max(self.height - covered_h, 0),
            "discarded_right_remainder": max(self.width - covered_w, 0),
        }


@dataclass(frozen=True)
class EvalTileSequence:
    scenario: base.FloodSimScenario
    tile_id: int
    row_start: int
    col_start: int

    @property
    def sequence_id(self) -> str:
        return (
            f"{self.scenario.scenario_id}_tile{self.tile_id:03d}_"
            f"r{self.row_start}_c{self.col_start}"
        )


@dataclass
class LoadedTrainTile:
    scenario_index: int
    epoch: int
    row_start: int
    col_start: int
    depths: torch.Tensor
    terrain: torch.Tensor
    valid: torch.Tensor


def rainfall_at_frame_mm_h(rainfall_mm: int | float, frame_index: int) -> float:
    """Existing rectangular-storm convention extended over h5,...,h360."""
    if not 0 <= int(frame_index) < FRAMES_PER_SCENARIO:
        raise IndexError(frame_index)
    time_minutes = (int(frame_index) + 1) * DT_MINUTES
    return float(rainfall_mm) if time_minutes < 60 else 0.0


def forecasting_frame_indices(sample_index: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    if not 0 <= int(sample_index) < FORECAST_SAMPLES_PER_SEQUENCE:
        raise IndexError(sample_index)
    start = int(sample_index)
    target = start + DEPTH_HISTORY_STEPS
    rain = tuple(range(start, target + 1))
    depth = tuple(range(start, target))
    return rain, depth, target


def configure_terrain_store(
    root: str | Path = DEFAULT_ROOT,
) -> base.FloodSimTerrainStore:
    """Configure centered/scaled DEMs without the old full-tile restriction.

    The legacy overlap-crop store rejects domains smaller than 1024 because
    that experiment never padded. This task permits bottom/right padding when
    either dimension is smaller, so the shared DEM transform is used directly.
    """
    global _ACTIVE_STORE
    resolved = Path(root).expanduser().resolve()
    if _ACTIVE_STORE is None or _ACTIVE_STORE.dem_root != resolved:
        _ACTIVE_STORE = base.FloodSimTerrainStore(
            "terrain_centered_global_scaled", resolved
        )
    return _ACTIVE_STORE


def require_terrain_store() -> base.FloodSimTerrainStore:
    if _ACTIVE_STORE is None:
        raise RuntimeError("FloodSimBench context6 terrain store is not configured")
    return _ACTIVE_STORE


def make_channel_normalizer(
    normalizer_class: Any,
    root: str | Path = DEFAULT_ROOT,
    cache_path: Optional[str | Path] = None,
) -> Any:
    stats = base.compute_or_load_training_statistics(root, cache_path)
    return normalizer_class(
        mode="train_standard",
        mean=(stats["water_depth"]["mean"], stats["rainfall"]["mean"], 0.0),
        std=(stats["water_depth"]["std"], stats["rainfall"]["std"], 1.0),
        provenance={
            "dataset": "FloodSimBench",
            "allowed_roles": ["train"],
            "cities": list(TRAIN_CITIES),
            "statistics": stats,
            "terrain": require_terrain_store().metadata(),
            "channel_policy": {
                "water_depth": "train-only population standardization; metres",
                "rainfall": "train-only scalar transition standardization; mm/h",
                "terrain": "per-domain centered, train-domain equal-weight RMS scaled",
            },
            "dataset_variant": "FloodSimBench_context6_crop1024",
            "preprocessing_version": PREPROCESSING_VERSION,
            "normalization_reuse": "exact reference 1 m Train-city statistics",
        },
    )


def _geometries(
    scenarios: Sequence[base.FloodSimScenario],
) -> Dict[str, SpatialGeometry]:
    result: Dict[str, SpatialGeometry] = {}
    for scenario in scenarios:
        layout = crop.tiff_layout(str(scenario.dem_path))
        geometry = SpatialGeometry(scenario.domain, layout.height, layout.width)
        previous = result.setdefault(scenario.domain, geometry)
        if previous != geometry:
            raise ValueError(f"Inconsistent geometry for {scenario.domain}")
        for frame_path in (scenario.frame_paths[0], scenario.frame_paths[-1]):
            frame = crop.tiff_layout(str(frame_path))
            if (frame.height, frame.width) != (layout.height, layout.width):
                raise ValueError(
                    f"{scenario.scenario_id}: water/DEM dimensions disagree"
                )
    return result


def _window_with_optional_padding(
    path: str | Path,
    geometry: SpatialGeometry,
    row_start: int,
    col_start: int,
    *,
    clean_water: bool,
) -> torch.Tensor:
    row_start, col_start = int(row_start), int(col_start)
    if row_start < 0 or col_start < 0:
        raise ValueError("Negative crop coordinate")
    raw_h = min(CROP_SIZE, geometry.height - row_start)
    raw_w = min(CROP_SIZE, geometry.width - col_start)
    if raw_h <= 0 or raw_w <= 0:
        raise ValueError(
            f"{geometry.domain}: crop {(row_start, col_start)} outside raw raster"
        )
    array, nodata = crop.read_declared_tiff_window(
        path, row_start, col_start, height=raw_h, width=raw_w
    )
    if clean_water:
        array = crop.clean_depth(array, nodata)
    else:
        array = np.asarray(array, dtype=np.float32).copy()
        array[crop.invalid_mask(array, nodata)] = np.nan
    output = torch.full((CROP_SIZE, CROP_SIZE), float("nan"), dtype=torch.float32)
    output[:raw_h, :raw_w] = torch.from_numpy(array)
    return output


class FloodSimBenchContext6Dataset(Dataset):
    """Context-6 forecasting Dataset with role-specific spatial sampling."""

    def __init__(
        self,
        root: str | Path,
        cities: Sequence[str],
        role: str,
        normalizer: Any,
        *,
        seed: int = 1,
    ) -> None:
        if role not in {"train", "valid", "test"}:
            raise ValueError(role)
        self.root = Path(root).expanduser().resolve()
        self.cities = tuple(str(city).upper() for city in cities)
        self.role = role
        self.normalizer = normalizer
        self.seed = int(seed)
        self.scenarios, self.ignored_tiffs = base.discover_scenarios(
            self.root, self.cities
        )
        self.store = require_terrain_store()
        self.geometries = _geometries(self.scenarios)
        self._epoch = mp.Value("q", 0, lock=False)
        self._cached_train: Optional[LoadedTrainTile] = None
        if role == "train":
            self.files: List[Any] = list(self.scenarios)
        else:
            self.files = []
            for scenario in self.scenarios:
                geometry = self.geometries[scenario.domain]
                for tile_id, (row, col) in enumerate(geometry.fixed_coordinates()):
                    self.files.append(EvalTileSequence(scenario, tile_id, row, col))
        self.pairs_per_sequence = FORECAST_SAMPLES_PER_SEQUENCE
        self.time_length = FRAMES_PER_SCENARIO
        self.profile = SimpleNamespace(
            tensor_shape=(INPUT_CHANNELS, CROP_SIZE, CROP_SIZE),
            dtype="float32",
            time_length=FRAMES_PER_SCENARIO,
            height=CROP_SIZE,
            width=CROP_SIZE,
            channels=INPUT_CHANNELS,
            has_explicit_mask=True,
            has_timestamps=True,
            has_nan=False,
            has_inf=False,
            terrain_is_static=True,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_cached_train"] = None
        return state

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)
        self._cached_train = None

    @property
    def sequence_count(self) -> int:
        return len(self.scenarios) if self.role == "train" else len(self.files)

    def __len__(self) -> int:
        return self.sequence_count * FORECAST_SAMPLES_PER_SEQUENCE

    def random_coordinate(
        self, scenario_index: int, epoch: Optional[int] = None
    ) -> Tuple[int, int]:
        if self.role != "train":
            raise RuntimeError("random_coordinate is Train-only")
        scenario = self.scenarios[int(scenario_index)]
        geometry = self.geometries[scenario.domain]
        epoch_value = self.epoch if epoch is None else int(epoch)
        rng = random.Random(
            self.seed
            + 1_000_003 * epoch_value
            + 10_007 * int(scenario_index)
            + 97 * int(scenario.rainfall_mm)
        )
        max_row = geometry.padded_height - CROP_SIZE
        max_col = geometry.padded_width - CROP_SIZE
        return rng.randint(0, max_row), rng.randint(0, max_col)

    def _terrain_tile(
        self, scenario: base.FloodSimScenario, row: int, col: int
    ) -> torch.Tensor:
        geometry = self.geometries[scenario.domain]
        full = self.store.for_domain(scenario.domain)[0]
        raw_h = min(CROP_SIZE, geometry.height - row)
        raw_w = min(CROP_SIZE, geometry.width - col)
        output = torch.full((CROP_SIZE, CROP_SIZE), float("nan"), dtype=torch.float32)
        output[:raw_h, :raw_w] = full[row : row + raw_h, col : col + raw_w]
        return output

    def _depth_tile(
        self,
        scenario: base.FloodSimScenario,
        frame_index: int,
        row: int,
        col: int,
    ) -> torch.Tensor:
        return _window_with_optional_padding(
            scenario.frame_paths[int(frame_index)],
            self.geometries[scenario.domain],
            row,
            col,
            clean_water=True,
        )

    def _load_train_sequence(self, scenario_index: int) -> LoadedTrainTile:
        epoch = self.epoch
        cached = self._cached_train
        if (
            cached is not None
            and cached.scenario_index == scenario_index
            and cached.epoch == epoch
        ):
            return cached
        scenario = self.scenarios[scenario_index]
        row, col = self.random_coordinate(scenario_index, epoch)
        terrain = self._terrain_tile(scenario, row, col)
        terrain_valid = torch.isfinite(terrain)
        depths = torch.empty(
            (FRAMES_PER_SCENARIO, CROP_SIZE, CROP_SIZE), dtype=torch.float32
        )
        for frame_index in range(FRAMES_PER_SCENARIO):
            depth = self._depth_tile(scenario, frame_index, row, col)
            if not torch.equal(torch.isfinite(depth), terrain_valid):
                raise ValueError(
                    f"{scenario.scenario_id}@h{(frame_index + 1) * DT_MINUTES}: "
                    "water/DEM masks disagree"
                )
            depths[frame_index].copy_(depth)
        if not bool(terrain_valid.any()):
            raise ValueError(f"{scenario.scenario_id}: random crop has no valid pixels")
        loaded = LoadedTrainTile(
            scenario_index=scenario_index,
            epoch=epoch,
            row_start=row,
            col_start=col,
            depths=depths,
            terrain=terrain,
            valid=terrain_valid,
        )
        self._cached_train = loaded
        return loaded

    def eval_tile(self, sequence_index: int) -> EvalTileSequence:
        if self.role == "train":
            raise RuntimeError("eval_tile is unavailable for Train")
        return self.files[int(sequence_index)]

    def load_eval_depth(self, tile: EvalTileSequence, frame_index: int) -> torch.Tensor:
        return self._depth_tile(
            tile.scenario, frame_index, tile.row_start, tile.col_start
        )

    def load_eval_terrain(self, tile: EvalTileSequence) -> torch.Tensor:
        return self._terrain_tile(tile.scenario, tile.row_start, tile.col_start)

    def _assemble(
        self,
        scenario: base.FloodSimScenario,
        depths: torch.Tensor,
        terrain: torch.Tensor,
        valid: torch.Tensor,
        forecast_index: int,
        row: int,
        col: int,
        sequence_id: str,
    ) -> Dict[str, Any]:
        rain_indices, depth_indices, target_index = forecasting_frame_indices(
            forecast_index
        )
        x = torch.empty((INPUT_CHANNELS, CROP_SIZE, CROP_SIZE), dtype=torch.float32)
        for channel, frame_index in enumerate(rain_indices):
            rain = rainfall_at_frame_mm_h(scenario.rainfall_mm, frame_index)
            rain_n = (rain - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[channel].fill_(rain_n)
        x[RAINFALL_CONTEXT_STEPS].copy_(terrain)
        for offset, frame_index in enumerate(depth_indices):
            x[RAINFALL_CONTEXT_STEPS + 1 + offset].copy_(
                self.normalizer.normalize(depths[frame_index], 0)
            )
        target = self.normalizer.normalize(depths[target_index], 0)
        x = torch.where(valid.unsqueeze(0), x, torch.zeros_like(x))
        target = torch.where(valid, target, torch.zeros_like(target))
        if not torch.isfinite(x).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"{sequence_id}: assembled sample is non-finite")
        return {
            "x": x,
            "y": target.unsqueeze(0).float(),
            "mask": valid.unsqueeze(0),
            "city": scenario.city,
            "domain_name": scenario.domain,
            "scenario_id": scenario.scenario_id,
            "sequence_id": sequence_id,
            "row_start": row,
            "col_start": col,
            "forecast_index": forecast_index,
            "target_frame_index": target_index,
            "target_time_minutes": (target_index + 1) * DT_MINUTES,
            "rainfall_frame_indices": rain_indices,
            "depth_frame_indices": depth_indices,
            "height": CROP_SIZE,
            "width": CROP_SIZE,
            "dx": GRID_SPACING_M,
            "dy": GRID_SPACING_M,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        sequence_index = int(index) // FORECAST_SAMPLES_PER_SEQUENCE
        forecast_index = int(index) % FORECAST_SAMPLES_PER_SEQUENCE
        if self.role == "train":
            scenario = self.scenarios[sequence_index]
            loaded = self._load_train_sequence(sequence_index)
            return self._assemble(
                scenario,
                loaded.depths,
                loaded.terrain,
                loaded.valid,
                forecast_index,
                loaded.row_start,
                loaded.col_start,
                f"{scenario.scenario_id}_epoch{loaded.epoch}",
            )
        tile = self.eval_tile(sequence_index)
        terrain = self.load_eval_terrain(tile)
        valid = torch.isfinite(terrain)
        _, depth_indices, target_index = forecasting_frame_indices(forecast_index)
        needed = (*depth_indices, target_index)
        depths = torch.empty(
            (FRAMES_PER_SCENARIO, CROP_SIZE, CROP_SIZE), dtype=torch.float32
        )
        for frame_index in needed:
            depth = self.load_eval_depth(tile, frame_index)
            if not torch.equal(torch.isfinite(depth), valid):
                raise ValueError(f"{tile.sequence_id}: water/DEM masks disagree")
            depths[frame_index].copy_(depth)
        return self._assemble(
            tile.scenario,
            depths,
            terrain,
            valid,
            forecast_index,
            tile.row_start,
            tile.col_start,
            tile.sequence_id,
        )


def homogeneous_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    shapes = {tuple(sample["x"].shape) for sample in samples}
    if shapes != {(INPUT_CHANNELS, CROP_SIZE, CROP_SIZE)}:
        raise AssertionError(f"Unexpected input shapes {shapes}")
    batch: Dict[str, Any] = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in ("x", "y", "mask")
    }
    for key in samples[0]:
        if key not in batch:
            batch[key] = [sample[key] for sample in samples]
    return batch


def split_inventory(root: str | Path = DEFAULT_ROOT) -> Dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    scenarios, ignored = base.discover_scenarios(root_path)
    geometries = _geometries(scenarios)
    result: Dict[str, Any] = {}
    for role, cities in (
        ("train", TRAIN_CITIES), ("valid", VALID_CITIES), ("test", TEST_CITIES)
    ):
        selected = [item for item in scenarios if item.city in cities]
        if role == "train":
            sequences = len(selected)
        else:
            sequences = sum(
                len(geometries[item.domain].fixed_coordinates()) for item in selected
            )
        result[role] = {
            "cities": list(cities),
            "domains": sorted({item.domain for item in selected}),
            "scenario_count": len(selected),
            "tile_sequences": sequences,
            "forecast_samples": sequences * FORECAST_SAMPLES_PER_SEQUENCE,
        }
    return {
        "root": str(root_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "crop_size": CROP_SIZE,
        "train_crop": "one random full crop per scenario per epoch",
        "eval_grid": "fixed stride=1024 overlap=0; ordinary remainder discarded",
        "augmentation": False,
        "frames_per_sequence": FRAMES_PER_SCENARIO,
        "forecast_samples_per_sequence": FORECAST_SAMPLES_PER_SEQUENCE,
        "input_channels": INPUT_CHANNELS,
        "channel_order": list(CHANNEL_ORDER),
        "splits": result,
        "geometries": {
            domain: geometry.metadata()
            for domain, geometry in sorted(geometries.items())
        },
        "ignored_tiffs": ignored,
    }
