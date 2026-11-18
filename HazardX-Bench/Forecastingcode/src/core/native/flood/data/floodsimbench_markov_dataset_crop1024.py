"""FloodSimBench 1 m overlapping 1024-tile Markov dataset.

This is an independent data implementation shared by the Fourier-APE and
physical-RoPE crop experiments.  It follows ``crop_geotiff_overlap`` from
FloodRisk-DL: complete 1024 x 1024 windows only, stride 924 (100-pixel
overlap), and four variants (identity, vertical flip, horizontal flip, both).
No edge completion, padding, resizing, or domain stitching is performed.

The source observations are h5,...,h360.  Every augmented spatial tile is an
independent 72-frame sequence with 71 Markov transitions.  Rainfall remains a
spatially uniform scalar in mm/h.  Raster NoData becomes NaN; valid negative
water depth and valid depth below 1 mm become physical zero before tiling.
"""

from __future__ import annotations

import json
import math
import mmap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import floodsimbench_markov_dataset as base


DEFAULT_ROOT = base.DEFAULT_ROOT
DT_MINUTES = base.DT_MINUTES
FRAMES_PER_SCENARIO = base.FRAMES_PER_SCENARIO
TRANSITIONS_PER_SCENARIO = base.TRANSITIONS_PER_SCENARIO
STORM_TRANSITIONS = base.STORM_TRANSITIONS
EXPECTED_TIMES = base.EXPECTED_TIMES
CHANNEL_NAMES = base.CHANNEL_NAMES
CHANNEL_UNITS = base.CHANNEL_UNITS
TRAIN_CITIES = base.TRAIN_CITIES
VALID_CITIES = base.VALID_CITIES
TEST_CITIES = base.TEST_CITIES
ALL_CITIES = base.ALL_CITIES
TRAIN_DOMAIN_NAMES = base.TRAIN_DOMAIN_NAMES
FloodSimScenario = base.FloodSimScenario
discover_scenarios = base.discover_scenarios
rainfall_rate_mm_h = base.rainfall_rate_mm_h
rainfall_series_mm_h = base.rainfall_series_mm_h

CROP_SIZE = 1024
OVERLAP = 100
STRIDE = CROP_SIZE - OVERLAP
AUGMENTATION_NAMES = (
    "original",
    "flip_up_down",
    "flip_left_right",
    "flip_up_down_left_right",
)
DEPTH_CLEAN_THRESHOLD_M = 1e-3
GRID_SPACING_M = 1.0
PREPROCESSING_VERSION = "floodsimbench_1m_crop1024_overlap100_flip4_v1"

Image.MAX_IMAGE_PIXELS = None


def _parse_nodata(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)):
        if not raw:
            return None
        raw = raw[0]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def invalid_mask(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    bad = ~np.isfinite(array)
    if nodata is not None and math.isfinite(nodata):
        bad |= array == np.float32(nodata)
    return bad


@dataclass(frozen=True)
class TiffLayout:
    path: Path
    height: int
    width: int
    nodata: Optional[float]
    fast_float32: bool
    byte_order: bytes
    strip_offsets: Tuple[int, ...]
    strip_byte_counts: Tuple[int, ...]


@lru_cache(maxsize=4096)
def tiff_layout(path_value: str) -> TiffLayout:
    path = Path(path_value)
    with Image.open(path) as image:
        width, height = image.size
        nodata = _parse_nodata(image.tag_v2.get(42113))
        compression = image.tag_v2.get(259)
        bits = tuple(image.tag_v2.get(258, ()))
        sample_format = tuple(image.tag_v2.get(339, ()))
        samples_per_pixel = int(image.tag_v2.get(277, 1))
        rows_per_strip = int(image.tag_v2.get(278, 0))
        orientation = image.tag_v2.get(274)
        offsets = tuple(int(value) for value in image.tag_v2.get(273, ()))
        byte_counts = tuple(int(value) for value in image.tag_v2.get(279, ()))
        byte_order = image.tag_v2.prefix
        fast = (
            image.mode == "F"
            and int(compression) == 1
            and bits == (32,)
            and sample_format == (3,)
            and samples_per_pixel == 1
            and rows_per_strip == 1
            and orientation in (None, 1)
            and byte_order in (b"II", b"MM")
            and len(offsets) == height
            and len(byte_counts) == height
            and all(count == width * 4 for count in byte_counts)
        )
    return TiffLayout(
        path=path,
        height=height,
        width=width,
        nodata=nodata,
        fast_float32=fast,
        byte_order=byte_order,
        strip_offsets=offsets,
        strip_byte_counts=byte_counts,
    )


def read_declared_tiff_window(
    path: str | Path,
    row_start: int,
    col_start: int,
    height: int = CROP_SIZE,
    width: int = CROP_SIZE,
) -> Tuple[np.ndarray, Optional[float]]:
    """Read one exact TIFF window, with a zero-copy mmap fast path."""
    source = Path(path)
    layout = tiff_layout(str(source))
    row_start, col_start = int(row_start), int(col_start)
    height, width = int(height), int(width)
    if (
        row_start < 0
        or col_start < 0
        or row_start + height > layout.height
        or col_start + width > layout.width
    ):
        raise ValueError(
            f"Window {(row_start, col_start, height, width)} is outside "
            f"{source} shape {(layout.height, layout.width)}"
        )
    if not layout.fast_float32:
        with Image.open(source) as image:
            window = image.crop(
                (col_start, row_start, col_start + width, row_start + height)
            )
            array = np.array(window, dtype=np.float32, copy=True)
        return array, layout.nodata

    dtype = np.dtype("<f4" if layout.byte_order == b"II" else ">f4")
    output = np.empty((height, width), dtype=np.float32)
    with source.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            for local_row, source_row in enumerate(
                range(row_start, row_start + height)
            ):
                offset = layout.strip_offsets[source_row] + col_start * 4
                stop = offset + width * 4
                if stop > len(mapped):
                    raise ValueError(f"Invalid TIFF strip window in {source}")
                output[local_row] = np.frombuffer(
                    mapped, dtype=dtype, count=width, offset=offset
                )
        finally:
            mapped.close()
    return output, layout.nodata


def clean_depth(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Convert NoData to NaN and clean valid water depth in physical metres."""
    output = np.asarray(array, dtype=np.float32).copy()
    bad = invalid_mask(output, nodata)
    output[bad] = np.nan
    valid = ~bad
    output[valid & (output < 0.0)] = 0.0
    output[valid & (output < DEPTH_CLEAN_THRESHOLD_M)] = 0.0
    return output


def load_clean_depth_tile(
    path: str | Path, row_start: int, col_start: int
) -> torch.Tensor:
    array, nodata = read_declared_tiff_window(path, row_start, col_start)
    return torch.from_numpy(clean_depth(array, nodata))


def apply_augmentation(tensor: torch.Tensor, augmentation_id: int) -> torch.Tensor:
    augmentation_id = int(augmentation_id)
    if augmentation_id == 0:
        return tensor
    if augmentation_id == 1:
        return torch.flip(tensor, dims=(-2,))
    if augmentation_id == 2:
        return torch.flip(tensor, dims=(-1,))
    if augmentation_id == 3:
        return torch.flip(tensor, dims=(-2, -1))
    raise ValueError(f"Unknown augmentation_id {augmentation_id}")


def tile_coordinates(height: int, width: int) -> Tuple[Tuple[int, int], ...]:
    """Exact FloodRisk-DL full-window coordinates; no edge completion."""
    return tuple(
        (row, col)
        for row in range(0, int(height) - CROP_SIZE + 1, STRIDE)
        for col in range(0, int(width) - CROP_SIZE + 1, STRIDE)
    )


@dataclass(frozen=True)
class DomainTileGeometry:
    domain: str
    dem_path: Path
    raw_shape: Tuple[int, int]
    coordinates: Tuple[Tuple[int, int], ...]

    @property
    def num_rows(self) -> int:
        return len({row for row, _ in self.coordinates})

    @property
    def num_cols(self) -> int:
        return len({col for _, col in self.coordinates})

    @property
    def base_tiles(self) -> int:
        return len(self.coordinates)

    def metadata(self) -> Dict[str, Any]:
        h, w = self.raw_shape
        covered_h = self.coordinates[-1][0] + CROP_SIZE
        covered_w = max(col for _, col in self.coordinates) + CROP_SIZE
        return {
            "domain": self.domain,
            "raw_shape_1m": [h, w],
            "crop_size": CROP_SIZE,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "base_tiles": self.base_tiles,
            "augmented_tiles": self.base_tiles * len(AUGMENTATION_NAMES),
            "covered_extent": [covered_h, covered_w],
            "discarded_bottom": h - covered_h,
            "discarded_right": w - covered_w,
            "padding": False,
            "edge_completion": False,
        }


@dataclass(frozen=True)
class FloodSimTileSequence:
    scenario: FloodSimScenario
    tile_id: int
    row_start: int
    col_start: int
    augmentation_id: int

    @property
    def domain(self) -> str:
        return self.scenario.domain

    @property
    def city(self) -> str:
        return self.scenario.city

    @property
    def rainfall_mm(self) -> int:
        return self.scenario.rainfall_mm

    @property
    def sequence_id(self) -> str:
        return (
            f"{self.scenario.scenario_id}_tile{self.tile_id:03d}_"
            f"r{self.row_start}_c{self.col_start}_aug{self.augmentation_id}"
        )

    @property
    def stem(self) -> str:
        return self.sequence_id

    @property
    def name(self) -> str:
        return self.sequence_id

    def __str__(self) -> str:
        return self.sequence_id


@dataclass(frozen=True)
class FloodSimLoadedTileSequence:
    tile: FloodSimTileSequence
    source_path: str
    dx: float
    dy: float

    @property
    def scenario(self) -> FloodSimScenario:
        return self.tile.scenario


class FloodSimTerrainStoreCrop1024(base.FloodSimTerrainStore):
    """Original 1 m DEM transform plus deterministic per-domain windows."""

    def __init__(self, kind: str, dem_root: str | Path):
        super().__init__(kind, dem_root)
        self.geometries: Dict[str, DomainTileGeometry] = {}
        for domain, path in sorted(self._dem_paths.items()):
            h, w = map(int, self.region_stats[domain]["shape"])
            coordinates = tile_coordinates(h, w)
            if not coordinates:
                raise ValueError(f"{domain} shape {(h, w)} has no 1024 tile")
            self.geometries[domain] = DomainTileGeometry(
                domain=domain,
                dem_path=path,
                raw_shape=(h, w),
                coordinates=coordinates,
            )

    def for_tile(self, tile: FloodSimTileSequence) -> torch.Tensor:
        geometry = self.geometries[tile.domain]
        expected = geometry.coordinates[tile.tile_id]
        if expected != (tile.row_start, tile.col_start):
            raise AssertionError(
                f"{tile.domain} tile {tile.tile_id}: {expected} != "
                f"{(tile.row_start, tile.col_start)}"
            )
        terrain = self.for_domain(tile.domain)[
            :,
            tile.row_start : tile.row_start + CROP_SIZE,
            tile.col_start : tile.col_start + CROP_SIZE,
        ]
        if tuple(terrain.shape) != (1, CROP_SIZE, CROP_SIZE):
            raise AssertionError(f"Wrong DEM tile shape {tuple(terrain.shape)}")
        return apply_augmentation(terrain, tile.augmentation_id)

    def metadata(self) -> Dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "grid_spacing_m": GRID_SPACING_M,
                "tiling": {
                    "crop_size": CROP_SIZE,
                    "overlap": OVERLAP,
                    "stride": STRIDE,
                    "augmentations": list(AUGMENTATION_NAMES),
                    "padding": False,
                    "edge_completion": False,
                    "domain_stitching": False,
                },
                "domain_tiles": {
                    domain: geometry.metadata()
                    for domain, geometry in sorted(self.geometries.items())
                },
            }
        )
        return result


FloodSimTerrainStore = FloodSimTerrainStoreCrop1024
_ACTIVE_STORE: Optional[FloodSimTerrainStoreCrop1024] = None


def configure_terrain_store(
    root: str | Path = DEFAULT_ROOT,
) -> FloodSimTerrainStoreCrop1024:
    global _ACTIVE_STORE
    resolved = Path(root).expanduser().resolve()
    if _ACTIVE_STORE is None or _ACTIVE_STORE.dem_root != resolved:
        _ACTIVE_STORE = FloodSimTerrainStoreCrop1024(
            "terrain_centered_global_scaled", resolved
        )
    return _ACTIVE_STORE


def require_terrain_store() -> FloodSimTerrainStoreCrop1024:
    if _ACTIVE_STORE is None:
        raise RuntimeError("FloodSimBench crop1024 terrain store is not configured")
    return _ACTIVE_STORE


def _tile_sequences_for_scenarios(
    scenarios: Sequence[FloodSimScenario],
    store: FloodSimTerrainStoreCrop1024,
) -> List[FloodSimTileSequence]:
    sequences: List[FloodSimTileSequence] = []
    for scenario in scenarios:
        geometry = store.geometries[scenario.domain]
        first_layout = tiff_layout(str(scenario.frame_paths[0]))
        if (first_layout.height, first_layout.width) != geometry.raw_shape:
            raise ValueError(
                f"{scenario.scenario_id}: water shape "
                f"{(first_layout.height, first_layout.width)} != DEM "
                f"{geometry.raw_shape}"
            )
        for tile_id, (row, col) in enumerate(geometry.coordinates):
            for augmentation_id in range(len(AUGMENTATION_NAMES)):
                sequences.append(
                    FloodSimTileSequence(
                        scenario=scenario,
                        tile_id=tile_id,
                        row_start=row,
                        col_start=col,
                        augmentation_id=augmentation_id,
                    )
                )
    return sequences


class FloodSimBenchMarkovDataset(Dataset):
    """One virtual sequence per scenario x base tile x augmentation."""

    def __init__(self, spec: Any, normalizer: Any, noise_std: float = 0.0):
        self.spec = spec
        self.normalizer = normalizer
        self.noise_std = float(noise_std)
        if not math.isclose(float(spec.dx), 1.0) or not math.isclose(
            float(spec.dy), 1.0
        ):
            raise ValueError(
                f"crop1024 dataset requires dx=dy=1, got {spec.dx}, {spec.dy}"
            )
        cities = tuple(part for part in str(spec.name).split("+") if part)
        scenarios, self.ignored_tiffs = discover_scenarios(spec.path, cities)
        self.store = require_terrain_store()
        self.files = _tile_sequences_for_scenarios(scenarios, self.store)
        self.raw_scenarios = scenarios
        self.pairs_per_sequence = TRANSITIONS_PER_SCENARIO
        self.time_length = FRAMES_PER_SCENARIO
        first = self.files[0]
        self.profile = SimpleNamespace(
            path=first.sequence_id,
            tensor_shape=(self.time_length, CROP_SIZE, CROP_SIZE, 3),
            dtype="float32",
            time_length=self.time_length,
            height=CROP_SIZE,
            width=CROP_SIZE,
            channels=3,
            has_explicit_mask=True,
            has_timestamps=True,
            has_nan=True,
            has_inf=False,
            terrain_is_static=True,
            raw_keys=None,
        )
        self.dem_feature = self.store.for_tile(first)

    def __len__(self) -> int:
        return len(self.files) * self.pairs_per_sequence

    def get_sequence(self, sequence_index: int) -> FloodSimLoadedTileSequence:
        tile = self.files[sequence_index]
        return FloodSimLoadedTileSequence(
            tile=tile,
            source_path=tile.sequence_id,
            dx=GRID_SPACING_M,
            dy=GRID_SPACING_M,
        )

    def _load_depth(self, tile: FloodSimTileSequence, frame_index: int) -> torch.Tensor:
        depth = load_clean_depth_tile(
            tile.scenario.frame_paths[frame_index],
            tile.row_start,
            tile.col_start,
        )
        return apply_augmentation(depth, tile.augmentation_id)

    def _load_pair(
        self, tile: FloodSimTileSequence, transition: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t = self._load_depth(tile, transition)
        h_next = self._load_depth(tile, transition + 1)
        terrain = self.store.for_tile(tile)[0]
        if not (
            h_t.shape == h_next.shape == terrain.shape == (CROP_SIZE, CROP_SIZE)
        ):
            raise AssertionError(
                f"{tile.sequence_id}: depth/target/DEM shapes disagree"
            )
        current_mask = torch.isfinite(h_t)
        next_mask = torch.isfinite(h_next)
        terrain_mask = torch.isfinite(terrain)
        if not torch.equal(current_mask, next_mask) or not torch.equal(
            current_mask, terrain_mask
        ):
            raise ValueError(
                f"{tile.sequence_id}@{transition}: DEM/water masks are not aligned"
            )
        return h_t, h_next, terrain, current_mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_index = index // self.pairs_per_sequence
        transition = index % self.pairs_per_sequence
        tile = self.files[sequence_index]
        h_t, h_next, terrain, valid = self._load_pair(tile, transition)
        if not bool(valid.any()):
            raise ValueError(f"{tile.sequence_id}@{transition}: no valid pixels")
        if bool((h_t[valid] < 0).any()) or bool((h_next[valid] < 0).any()):
            raise AssertionError("Negative valid depth survived preprocessing")
        for depth in (h_t, h_next):
            if bool(((depth > 0) & (depth < DEPTH_CLEAN_THRESHOLD_M)).any()):
                raise AssertionError("Positive sub-millimetre depth survived cleanup")

        rain_value = rainfall_rate_mm_h(tile.rainfall_mm, transition)
        rain_norm_value = (
            rain_value - self.normalizer.mean[1]
        ) / self.normalizer.std[1]
        h_t_n = self.normalizer.normalize(h_t, 0)
        h_next_n = self.normalizer.normalize(h_next, 0)
        rain_n = torch.full_like(h_t_n, rain_norm_value)
        h_t_n = torch.where(valid, h_t_n, torch.zeros_like(h_t_n))
        h_next_n = torch.where(valid, h_next_n, torch.zeros_like(h_next_n))
        rain_n = torch.where(valid, rain_n, torch.zeros_like(rain_n))
        terrain_n = torch.where(valid, terrain, torch.zeros_like(terrain))
        x = torch.stack((h_t_n, rain_n), dim=0).float()
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * valid.unsqueeze(0)
        return {
            "x": x,
            "y": h_next_n.unsqueeze(0).float(),
            "static": terrain_n.unsqueeze(0).float(),
            "mask": valid.unsqueeze(0),
            "domain": self.spec.name,
            "domain_id": self.spec.domain_id,
            "domain_name": tile.domain,
            "tile_id": tile.tile_id,
            "row_start": tile.row_start,
            "col_start": tile.col_start,
            "augmentation_id": tile.augmentation_id,
            "augmentation_name": AUGMENTATION_NAMES[tile.augmentation_id],
            "sample_id": tile.sequence_id,
            "source_path": str(tile.scenario.frame_paths[transition]),
            "time_index": transition,
            "t_start_min": (transition + 1) * DT_MINUTES,
            "t_end_min": (transition + 2) * DT_MINUTES,
            "rainfall_mm_h": rain_value,
            "dt_minutes": DT_MINUTES,
            "dx": GRID_SPACING_M,
            "dy": GRID_SPACING_M,
            "height": CROP_SIZE,
            "width": CROP_SIZE,
        }


def homogeneous_collate(
    samples: Sequence[Mapping[str, Any]], base_collate: Any
) -> Dict[str, Any]:
    batch = base_collate(samples)
    for key in (
        "domain_name",
        "tile_id",
        "row_start",
        "col_start",
        "augmentation_id",
        "augmentation_name",
        "t_start_min",
        "t_end_min",
        "rainfall_mm_h",
    ):
        batch[key] = [sample[key] for sample in samples]
    return batch


def make_channel_normalizer(
    normalizer_class: Any,
    root: str | Path = DEFAULT_ROOT,
    cache_path: Optional[str | Path] = None,
) -> Any:
    """Reuse the exact 1 m Train-only statistics from the reference run."""
    stats = base.compute_or_load_training_statistics(root, cache_path)
    provenance = {
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
        "dataset_variant": "FloodSimBench_crop1024",
        "preprocessing_version": PREPROCESSING_VERSION,
        "normalization_reuse": (
            "exact reference 1 m Train-city statistics; overlap and flip do "
            "not refit or reweight normalization"
        ),
        "crop1024": require_terrain_store().metadata()["tiling"],
    }
    return normalizer_class(
        mode="train_standard",
        mean=(stats["water_depth"]["mean"], stats["rainfall"]["mean"], 0.0),
        std=(stats["water_depth"]["std"], stats["rainfall"]["std"], 1.0),
        provenance=provenance,
    )


compute_or_load_training_statistics = base.compute_or_load_training_statistics


def rollout_sequence(
    model: torch.nn.Module,
    sequence: FloodSimLoadedTileSequence,
    spec: Any,
    normalizer: Any,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
) -> Dict[str, Any]:
    """Independent true-h5 tile rollout with physical cleanup between steps."""
    import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline

    del spec
    tile = sequence.tile
    steps = (
        TRANSITIONS_PER_SCENARIO
        if max_steps <= 0
        else min(max_steps, TRANSITIONS_PER_SCENARIO)
    )
    store = require_terrain_store()
    static_cpu = store.for_tile(tile)[0]
    initial_state = apply_augmentation(
        load_clean_depth_tile(
            tile.scenario.frame_paths[0], tile.row_start, tile.col_start
        ),
        tile.augmentation_id,
    )
    static_valid = torch.isfinite(static_cpu)
    initial_valid = torch.isfinite(initial_state)
    if not torch.equal(static_valid, initial_valid):
        raise ValueError(f"{tile.sequence_id}: initial water/DEM masks differ")
    h_state = normalizer.normalize(initial_state, 0)
    h_state = torch.where(initial_valid, h_state, torch.zeros_like(h_state))
    static = torch.where(static_valid, static_cpu, torch.zeros_like(static_cpu))
    h_state = h_state[None, None].float().to(device)
    static = static[None, None].float().to(device)
    current_truth = initial_state
    lead_metrics: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for transition in range(steps):
            target = apply_augmentation(
                load_clean_depth_tile(
                    tile.scenario.frame_paths[transition + 1],
                    tile.row_start,
                    tile.col_start,
                ),
                tile.augmentation_id,
            )
            valid = (
                static_valid
                & torch.isfinite(current_truth)
                & torch.isfinite(target)
            )
            if not torch.equal(valid, static_valid):
                raise ValueError(
                    f"{tile.sequence_id}@{transition}: rollout masks differ"
                )
            rain_value = rainfall_rate_mm_h(tile.rainfall_mm, transition)
            rain_norm = (rain_value - normalizer.mean[1]) / normalizer.std[1]
            rain = torch.full_like(target, rain_norm)
            rain = torch.where(valid, rain, torch.zeros_like(rain))[None, None]
            rain = rain.to(device)
            x = torch.cat((h_state, rain), dim=1)
            metadata = {
                "height": [CROP_SIZE],
                "width": [CROP_SIZE],
                "dt_minutes": [DT_MINUTES],
                "dx": [GRID_SPACING_M],
                "dy": [GRID_SPACING_M],
            }
            with baseline.autocast_context(amp):
                dh = model(x=x, static=static, metadata=metadata)
            raw_next_normalized = h_state.float() + dh.float()
            prediction_phys = normalizer.denormalize(raw_next_normalized, 0)
            valid_device = valid[None, None].to(device)
            prediction_phys = torch.where(
                valid_device,
                prediction_phys.clamp_min(0.0),
                torch.full_like(prediction_phys, float("nan")),
            )
            prediction_phys = torch.where(
                valid_device & (prediction_phys < DEPTH_CLEAN_THRESHOLD_M),
                torch.zeros_like(prediction_phys),
                prediction_phys,
            )
            if not torch.isfinite(prediction_phys[valid_device]).all():
                raise FloatingPointError(
                    f"{tile.sequence_id}: invalid prediction at step {transition + 1}"
                )
            target_device = target[None, None].to(device)
            metrics = baseline.masked_step_metrics(
                prediction_phys, target_device, valid_device
            )
            metrics.update(
                {
                    "lead_step": transition + 1,
                    "lead_minutes": (transition + 1) * DT_MINUTES,
                    "valid_pixels": int(valid.sum()),
                    "rainfall_mm_h": rain_value,
                }
            )
            lead_metrics.append(metrics)
            next_normalized = normalizer.normalize(prediction_phys, 0)
            h_state = torch.where(
                valid_device,
                next_normalized.detach(),
                torch.zeros_like(next_normalized),
            )
            current_truth = target

    names = [
        "rmse",
        "mae",
        "relative_l2",
        "nse",
        "corr",
        *(f"csi_{threshold:g}" for threshold in baseline.CSI_THRESHOLDS),
    ]
    overall = {
        name: baseline.mean_finite(step[name] for step in lead_metrics)
        for name in names
    }
    return {
        "sample_id": tile.sequence_id,
        "source_path": str(tile.scenario.frame_paths[0].parent),
        "dataset_version": "FloodSimBench_crop1024",
        "domain_name": tile.domain,
        "tile_id": tile.tile_id,
        "row_start": tile.row_start,
        "col_start": tile.col_start,
        "augmentation_id": tile.augmentation_id,
        "augmentation_name": AUGMENTATION_NAMES[tile.augmentation_id],
        "states_available": FRAMES_PER_SCENARIO,
        "prediction_steps": steps,
        "duration_minutes": steps * DT_MINUTES,
        "start_time_minutes": DT_MINUTES,
        "end_time_minutes": DT_MINUTES + steps * DT_MINUTES,
        "dx_m": GRID_SPACING_M,
        "dy_m": GRID_SPACING_M,
        "rollout_postprocessing": (
            "denormalize_m->valid_clamp_min_0->valid_lt_1mm_zero->"
            "restore_mask->train_stats_renormalize"
        ),
        "overall": overall,
        "lead_metrics": lead_metrics,
    }


def inventory(root: str | Path = DEFAULT_ROOT) -> Dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    scenarios, ignored = discover_scenarios(root_path)
    store = require_terrain_store()
    split_cities = {
        "train": TRAIN_CITIES,
        "valid": VALID_CITIES,
        "test": TEST_CITIES,
    }
    splits: Dict[str, Any] = {}
    for split, cities in split_cities.items():
        selected = [scenario for scenario in scenarios if scenario.city in cities]
        base_sequences = sum(
            store.geometries[scenario.domain].base_tiles for scenario in selected
        )
        augmented_sequences = base_sequences * len(AUGMENTATION_NAMES)
        splits[split] = {
            "cities": list(cities),
            "domains": sorted({scenario.domain for scenario in selected}),
            "domain_count": len({scenario.domain for scenario in selected}),
            "scenario_count": len(selected),
            "base_tile_sequences": base_sequences,
            "augmented_tile_sequences": augmented_sequences,
            "markov_pairs": augmented_sequences * TRANSITIONS_PER_SCENARIO,
        }
    return {
        "root": str(root_path),
        "dataset_version": "FloodSimBench_crop1024",
        "preprocessing_version": PREPROCESSING_VERSION,
        "complete_scenarios": len(scenarios),
        "ignored_tiffs": ignored,
        "tiling": {
            "crop_size": CROP_SIZE,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "augmentations": list(AUGMENTATION_NAMES),
            "padding": False,
            "edge_completion": False,
            "domain_stitching": False,
        },
        "domain_tiles": {
            domain: geometry.metadata()
            for domain, geometry in sorted(store.geometries.items())
        },
        "splits": splits,
    }


def describe_dataset(
    dataset: FloodSimBenchMarkovDataset, batch_size: Optional[int]
) -> Dict[str, Any]:
    domains = sorted({tile.domain for tile in dataset.files})
    description = {
        "domain": dataset.spec.name,
        "role": dataset.spec.role,
        "path": dataset.spec.path,
        "dataset_version": "FloodSimBench_crop1024",
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_scenarios": len(dataset.raw_scenarios),
        "independent_augmented_tile_sequences": len(dataset.files),
        "markov_pairs": len(dataset),
        "states_per_tile_sequence": FRAMES_PER_SCENARIO,
        "markov_pairs_per_tile_sequence": TRANSITIONS_PER_SCENARIO,
        "domains": domains,
        "domain_count": len(domains),
        "tile_shape": [CROP_SIZE, CROP_SIZE],
        "crop_size": CROP_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "augmentations": list(AUGMENTATION_NAMES),
        "channels": 3,
        "channel_names": list(CHANNEL_NAMES),
        "channel_units": list(CHANNEL_UNITS),
        "dimension_order": "x=[2,1024,1024], static=[1,1024,1024], y=[1,1024,1024]",
        "rainfall": "P mm/h for pair t_start=5..55 min, else 0 mm/h",
        "initial_state": "observed h5; no synthetic h0",
        "depth_cleanup": "NoData->NaN; valid <0->0; valid <1 mm->0 before crop",
        "dt_minutes": DT_MINUTES,
        "dx_m": GRID_SPACING_M,
        "dy_m": GRID_SPACING_M,
        "micro_batch_size": batch_size,
        "file_format": "source GeoTIFF; dynamic mmap-backed windows; no tile files",
        "position_coordinates": (
            "regular post-augmentation tensor grid; never flipped with data"
        ),
        "padding": False,
        "edge_completion": False,
        "domain_stitching": False,
        "terrain_global_train_scale": dataset.store.terrain_global_train_scale,
        "ignored_tiffs": dataset.ignored_tiffs,
    }
    print(json.dumps(description, ensure_ascii=False))
    return description


def validate_rainfall_contract() -> Dict[str, Any]:
    result = base.validate_rainfall_contract()
    result.update(
        {
            "dataset_version": "FloodSimBench_crop1024",
            "tile_shape": [CROP_SIZE, CROP_SIZE],
            "dx_m": GRID_SPACING_M,
            "dy_m": GRID_SPACING_M,
            "depth_cleanup_threshold_m": DEPTH_CLEAN_THRESHOLD_M,
        }
    )
    return result


def validate_augmentation_group(
    dataset: FloodSimBenchMarkovDataset, transition: int = 0
) -> Dict[str, Any]:
    """Check the first tile's four variants, including invariant grid metadata."""
    samples = [
        dataset[augmentation_id * TRANSITIONS_PER_SCENARIO + transition]
        for augmentation_id in range(4)
    ]
    reference = samples[0]
    dims = {1: (-2,), 2: (-1,), 3: (-2, -1)}
    for augmentation_id, flip_dims in dims.items():
        sample = samples[augmentation_id]
        for key in ("x", "y", "static", "mask"):
            if not torch.equal(
                sample[key], torch.flip(reference[key], dims=flip_dims)
            ):
                raise AssertionError(
                    f"augmentation {augmentation_id} did not synchronously flip {key}"
                )
        for key in ("tile_id", "row_start", "col_start", "height", "width", "dx", "dy"):
            if sample[key] != reference[key]:
                raise AssertionError(
                    f"position/grid metadata {key} changed under augmentation"
                )
        if sample["rainfall_mm_h"] != reference["rainfall_mm_h"]:
            raise AssertionError("rainfall scalar changed under augmentation")
    return {
        "domain_name": reference["domain_name"],
        "tile_id": reference["tile_id"],
        "row_start": reference["row_start"],
        "col_start": reference["col_start"],
        "augmentation_ids": [sample["augmentation_id"] for sample in samples],
        "synchronous_fields": ["h_t", "h_t+5", "DEM", "valid_mask"],
        "rainfall_scalar_invariant": True,
        "position_grid_metadata_invariant": True,
    }

