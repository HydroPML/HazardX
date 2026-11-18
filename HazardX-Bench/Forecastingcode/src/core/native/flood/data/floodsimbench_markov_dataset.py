"""FloodSimBench GeoTIFF inventory, Markov dataset, and streaming rollout.

This module is intentionally independent from the two model entry points.  It
is the single implementation of FloodSimBench parsing used by both the
Fourier-APE Transolver and physical-RoPE EarthDynamics experiments.

The on-disk sequence starts at t=5 minutes and contains 72 observed states.
No synthetic t=0 state is introduced: the 72 rasters yield 71 Markov
transitions from h5->h10 through h355->h360.  Rainfall is a spatially uniform
one-hour rectangular storm in mm/h: P for pair start times 5..55 minutes and
zero afterwards.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ROOT = Path("/mnt/HDD2/qingsong/qinqsong/Datasets/FM/FloodSimBench")
DT_MINUTES = 5
FRAMES_PER_SCENARIO = 72
TRANSITIONS_PER_SCENARIO = FRAMES_PER_SCENARIO - 1
STORM_TRANSITIONS = 11
EXPECTED_TIMES = tuple(30_000 * i for i in range(1, 73))
CHANNEL_NAMES = ("water_depth", "rainfall", "terrain")
CHANNEL_UNITS = ("m", "mm/h", "dimensionless_centered_global_scaled_DEM")
TRAIN_CITIES = ("HOU", "DAL", "OKC", "NYC", "ATL", "ORL")
VALID_CITIES = ("AUS", "MIA")
TEST_CITIES = ("LA", "SF")
ALL_CITIES = TRAIN_CITIES + VALID_CITIES + TEST_CITIES
TRAIN_DOMAIN_NAMES = (
    "HOU001", "HOU002", "HOU003", "HOU004", "HOU005", "HOU006", "HOU007",
    "DAL001", "DAL002", "OKC001", "OKC002", "NYC001", "NYC002",
    "ATL001", "ATL002", "ORL001", "ORL002",
)

_DOMAIN_RE = re.compile(r"^(?P<city>[A-Z]+)(?P<tile>\d{3})$")
_WATER_RE = re.compile(
    r"^(?P<domain>[A-Z]+\d{3})_(?P<rain>\d+)mm_"
    r"WaterDepth_(?P<time>\d+)\.tif$"
)


@dataclass(frozen=True)
class FloodSimScenario:
    domain: str
    city: str
    rainfall_mm: int
    dem_path: Path
    frame_paths: Tuple[Path, ...]
    times: Tuple[int, ...]

    @property
    def scenario_id(self) -> str:
        return f"{self.domain}_{self.rainfall_mm}mm"

    # The reference sampler/diagnostics treat sequence entries like Paths.
    @property
    def stem(self) -> str:
        return self.scenario_id

    @property
    def name(self) -> str:
        return self.scenario_id

    def __str__(self) -> str:
        return self.scenario_id


@dataclass(frozen=True)
class FloodSimLoadedScenario:
    scenario: FloodSimScenario
    source_path: str
    dx: float
    dy: float


def _city_of_domain(domain: str) -> str:
    match = _DOMAIN_RE.fullmatch(domain)
    if match is None:
        raise ValueError(f"Invalid FloodSimBench domain name {domain!r}")
    return match.group("city")


def rainfall_rate_mm_h(rainfall_mm: int | float, transition: int) -> float:
    """Return forcing for [t,t+5); transition 0 is the h5->h10 pair."""
    if not 0 <= int(transition) < TRANSITIONS_PER_SCENARIO:
        raise IndexError(transition)
    t_start_min = (int(transition) + 1) * DT_MINUTES
    return float(rainfall_mm) if t_start_min < 60 else 0.0


def rainfall_series_mm_h(rainfall_mm: int | float) -> Tuple[float, ...]:
    return tuple(
        rainfall_rate_mm_h(rainfall_mm, step)
        for step in range(TRANSITIONS_PER_SCENARIO)
    )


def discover_scenarios(
    root: str | Path = DEFAULT_ROOT,
    cities: Optional[Iterable[str]] = None,
    *,
    require_complete: bool = True,
) -> Tuple[List[FloodSimScenario], List[str]]:
    """Strictly parse actual WaterDepth scenarios, ignoring ``6hr_max``."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"FloodSimBench root does not exist: {root_path}")
    selected = None if cities is None else {str(city).upper() for city in cities}
    scenarios: List[FloodSimScenario] = []
    ignored: List[str] = []
    for domain_dir in sorted(root_path.iterdir(), key=lambda path: path.name):
        domain_match = _DOMAIN_RE.fullmatch(domain_dir.name)
        if not domain_dir.is_dir() or domain_match is None:
            continue
        domain = domain_dir.name
        city = domain_match.group("city")
        if selected is not None and city not in selected:
            continue
        dem_path = domain_dir / f"{domain}_DEM.tif"
        if not dem_path.is_file():
            raise FileNotFoundError(f"Missing DEM for {domain}: {dem_path}")
        grouped: Dict[int, Dict[int, Path]] = {}
        for path in sorted(domain_dir.glob("*.tif"), key=lambda item: item.name):
            if path == dem_path:
                continue
            match = _WATER_RE.fullmatch(path.name)
            if match is None or match.group("domain") != domain:
                ignored.append(str(path))
                continue
            rain = int(match.group("rain"))
            time_value = int(match.group("time"))
            if time_value in grouped.setdefault(rain, {}):
                raise ValueError(f"Duplicate {domain}/{rain}mm time {time_value}")
            grouped[rain][time_value] = path
        for rain, by_time in sorted(grouped.items()):
            times = tuple(sorted(by_time))
            if require_complete and times != EXPECTED_TIMES:
                missing = sorted(set(EXPECTED_TIMES) - set(times))
                extra = sorted(set(times) - set(EXPECTED_TIMES))
                raise ValueError(
                    f"Incomplete {domain}_{rain}mm: {len(times)} frames, "
                    f"missing={missing}, extra={extra}"
                )
            scenarios.append(
                FloodSimScenario(
                    domain=domain,
                    city=city,
                    rainfall_mm=rain,
                    dem_path=dem_path,
                    frame_paths=tuple(by_time[t] for t in times),
                    times=times,
                )
            )
    if selected is not None:
        found = {scenario.city for scenario in scenarios}
        missing_cities = selected - found
        if missing_cities:
            raise FileNotFoundError(
                f"No FloodSimBench scenarios for cities {sorted(missing_cities)}"
            )
    if not scenarios:
        raise FileNotFoundError(f"No FloodSimBench WaterDepth scenarios in {root_path}")
    return scenarios, ignored


def inventory(root: str | Path = DEFAULT_ROOT) -> Dict[str, Any]:
    scenarios, ignored = discover_scenarios(root)
    by_split: Dict[str, Dict[str, Any]] = {}
    split_cities = {
        "train": TRAIN_CITIES,
        "valid": VALID_CITIES,
        "test": TEST_CITIES,
    }
    for split, cities in split_cities.items():
        chosen = [scenario for scenario in scenarios if scenario.city in cities]
        by_split[split] = {
            "cities": list(cities),
            "domains": sorted({scenario.domain for scenario in chosen}),
            "domain_count": len({scenario.domain for scenario in chosen}),
            "scenario_count": len(chosen),
            "markov_pairs": len(chosen) * TRANSITIONS_PER_SCENARIO,
            "rainfall_scenarios_by_domain": {
                domain: sorted(
                    scenario.rainfall_mm
                    for scenario in chosen
                    if scenario.domain == domain
                )
                for domain in sorted({scenario.domain for scenario in chosen})
            },
        }
    city_sets = [set(value) for value in split_cities.values()]
    if any(city_sets[i] & city_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("FloodSimBench city splits overlap")
    return {
        "root": str(Path(root).expanduser().resolve()),
        "complete_scenarios": len(scenarios),
        "markov_pairs": len(scenarios) * TRANSITIONS_PER_SCENARIO,
        "ignored_tiffs": ignored,
        "splits": by_split,
    }


def read_tiff(path: str | Path) -> Tuple[torch.Tensor, Optional[float]]:
    """Read one float GeoTIFF with Pillow, retaining its GDAL nodata tag."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        nodata_raw = image.tag_v2.get(42113)
        array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim != 2:
        raise ValueError(f"Expected single-band TIFF {path}, got {array.shape}")
    nodata = None
    if nodata_raw is not None:
        try:
            nodata = float(nodata_raw)
        except (TypeError, ValueError):
            nodata = None
    return torch.from_numpy(array), nodata


def valid_values(values: torch.Tensor, nodata: Optional[float]) -> torch.Tensor:
    valid = torch.isfinite(values)
    if nodata is not None and math.isfinite(nodata):
        valid &= values != nodata
    return valid


def _array_stats(values: torch.Tensor, nodata: Optional[float]) -> Tuple[int, float, float]:
    valid = valid_values(values, nodata)
    finite = values[valid].double()
    if finite.numel() == 0:
        raise ValueError("Raster contains no valid values")
    return int(finite.numel()), float(finite.sum()), float(finite.square().sum())


class FloodSimTerrainStore:
    """Per-domain centered DEMs with one train-only equal-domain RMS scale."""

    def __init__(self, kind: str, dem_root: str | Path):
        if kind != "terrain_centered_global_scaled":
            raise ValueError(f"Unsupported FloodSimBench terrain kind {kind!r}")
        self.kind = kind
        self.dem_root = Path(dem_root).expanduser().resolve()
        scenarios, _ = discover_scenarios(self.dem_root)
        self._dem_paths = {
            scenario.domain: scenario.dem_path for scenario in scenarios
        }
        self.region_stats: Dict[str, Dict[str, Any]] = {}
        second_moments: List[float] = []
        for domain, path in sorted(self._dem_paths.items()):
            dem, nodata = read_tiff(path)
            count, total, total2 = _array_stats(dem, nodata)
            mean = total / count
            variance = max(total2 / count - mean * mean, 0.0)
            std = math.sqrt(variance)
            if not math.isfinite(std) or std <= 0.0:
                raise ValueError(f"Invalid DEM std for {domain}: {std}")
            self.region_stats[domain] = {
                "dem_path": str(path),
                "shape": list(dem.shape),
                "valid_pixels": count,
                "terrain_mean_m": mean,
                "terrain_population_std_m": std,
                "role_for_shared_scale": (
                    "train" if domain in TRAIN_DOMAIN_NAMES else "held_out"
                ),
            }
            if domain in TRAIN_DOMAIN_NAMES:
                second_moments.append(variance)
            del dem
        if set(TRAIN_DOMAIN_NAMES) != {
            name for name in self.region_stats if name in TRAIN_DOMAIN_NAMES
        }:
            raise AssertionError("Training DEM inventory is incomplete")
        self.terrain_global_train_scale = math.sqrt(
            sum(second_moments) / len(second_moments)
        )
        if not math.isfinite(self.terrain_global_train_scale) or self.terrain_global_train_scale <= 0:
            raise ValueError("Invalid train-only global centered DEM scale")
        self._cached_domain: Optional[str] = None
        self._cached_feature: Optional[torch.Tensor] = None

    @property
    def static_channels(self) -> int:
        return 1

    def for_domain(self, domain_name: str) -> torch.Tensor:
        # Exact domain is preferred.  A city-level spec resolves to its first
        # domain only for reference diagnostics; dataset samples pass exact IDs.
        domain = domain_name
        if domain not in self._dem_paths:
            matches = sorted(name for name in self._dem_paths if name.startswith(domain_name))
            if not matches:
                raise KeyError(domain_name)
            domain = matches[0]
        if domain == self._cached_domain and self._cached_feature is not None:
            return self._cached_feature
        dem, nodata = read_tiff(self._dem_paths[domain])
        valid = valid_values(dem, nodata)
        mean = float(self.region_stats[domain]["terrain_mean_m"])
        feature = (dem - mean) / self.terrain_global_train_scale
        feature = torch.where(valid, feature, torch.full_like(feature, float("nan")))
        self._cached_domain = domain
        self._cached_feature = feature.unsqueeze(0).float()
        return self._cached_feature

    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dem_root": str(self.dem_root),
            "static_feature_names": ["terrain_centered_global_scaled"],
            "static_channels": 1,
            "source_domains_for_shared_scale": list(TRAIN_DOMAIN_NAMES),
            "source_domain_weights": [1.0 / len(TRAIN_DOMAIN_NAMES)] * len(TRAIN_DOMAIN_NAMES),
            "shared_scale_pixel_weighting": "equal_train_domains_then_equal_valid_cells",
            "terrain_global_train_scale": self.terrain_global_train_scale,
            "region_stats": self.region_stats,
        }


_ACTIVE_STORE: Optional[FloodSimTerrainStore] = None


def configure_terrain_store(root: str | Path = DEFAULT_ROOT) -> FloodSimTerrainStore:
    global _ACTIVE_STORE
    resolved = Path(root).expanduser().resolve()
    if _ACTIVE_STORE is None or _ACTIVE_STORE.dem_root != resolved:
        _ACTIVE_STORE = FloodSimTerrainStore(
            "terrain_centered_global_scaled", resolved
        )
    return _ACTIVE_STORE


def require_terrain_store() -> FloodSimTerrainStore:
    if _ACTIVE_STORE is None:
        raise RuntimeError("FloodSimBench terrain store has not been configured")
    return _ACTIVE_STORE


class FloodSimBenchMarkovDataset(Dataset):
    """Lazy GeoTIFF Markov pairs, grouped by one or more city prefixes."""

    def __init__(self, spec: Any, normalizer: Any, noise_std: float = 0.0):
        self.spec = spec
        self.normalizer = normalizer
        self.noise_std = float(noise_std)
        cities = tuple(part for part in str(spec.name).split("+") if part)
        self.files, self.ignored_tiffs = discover_scenarios(spec.path, cities)
        self.pairs_per_sequence = TRANSITIONS_PER_SCENARIO
        self.time_length = FRAMES_PER_SCENARIO
        self.store = require_terrain_store()
        first = self.files[0]
        with Image.open(first.frame_paths[0]) as image:
            width, height = image.size
            dtype = str(np.asarray(image).dtype)
        self.profile = SimpleNamespace(
            path=first.scenario_id,
            tensor_shape=(self.time_length, height, width, 3),
            dtype=dtype,
            time_length=self.time_length,
            height=height,
            width=width,
            channels=3,
            has_explicit_mask=True,
            has_timestamps=True,
            has_nan=True,
            has_inf=False,
            terrain_is_static=True,
            raw_keys=None,
        )
        self.dem_feature = self.store.for_domain(first.domain)

    def __len__(self) -> int:
        return len(self.files) * self.pairs_per_sequence

    def get_sequence(self, sequence_index: int) -> FloodSimLoadedScenario:
        scenario = self.files[sequence_index]
        return FloodSimLoadedScenario(
            scenario=scenario,
            source_path=scenario.scenario_id,
            dx=float(self.spec.dx if self.spec.dx is not None else 1.0),
            dy=float(self.spec.dy if self.spec.dy is not None else 1.0),
        )

    def _load_pair(
        self, scenario: FloodSimScenario, transition: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t, current_nodata = read_tiff(scenario.frame_paths[transition])
        h_next, next_nodata = read_tiff(scenario.frame_paths[transition + 1])
        terrain = self.store.for_domain(scenario.domain)[0]
        valid = (
            valid_values(h_t, current_nodata)
            & valid_values(h_next, next_nodata)
            & torch.isfinite(terrain)
        )
        return h_t, h_next, terrain, valid

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_index = index // self.pairs_per_sequence
        transition = index % self.pairs_per_sequence
        scenario = self.files[sequence_index]
        h_t, h_next, terrain, valid = self._load_pair(scenario, transition)
        if not bool(valid.any()):
            raise ValueError(f"{scenario.scenario_id}@{transition}: no valid pixels")
        rain_value = rainfall_rate_mm_h(scenario.rainfall_mm, transition)
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
            "sample_id": scenario.scenario_id,
            "source_path": str(scenario.frame_paths[transition]),
            "time_index": transition,
            "t_start_min": (transition + 1) * DT_MINUTES,
            "t_end_min": (transition + 2) * DT_MINUTES,
            "rainfall_mm_h": rain_value,
            "dt_minutes": DT_MINUTES,
            "dx": float(self.spec.dx if self.spec.dx is not None else 1.0),
            "dy": float(self.spec.dy if self.spec.dy is not None else 1.0),
            "height": int(h_t.shape[0]),
            "width": int(h_t.shape[1]),
        }


def _statistics_fingerprint(scenarios: Sequence[FloodSimScenario]) -> str:
    digest = hashlib.sha256()
    digest.update(b"observed_h5_to_h360_71_pairs_v2")
    for scenario in scenarios:
        digest.update(scenario.scenario_id.encode())
        for path in (scenario.dem_path, *scenario.frame_paths):
            stat = path.stat()
            digest.update(str(path).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _rainfall_statistics(scenarios: Sequence[FloodSimScenario]) -> Tuple[float, float, int]:
    values = np.asarray(
        [value for scenario in scenarios for value in rainfall_series_mm_h(scenario.rainfall_mm)],
        dtype=np.float64,
    )
    return float(values.mean()), float(values.std(ddof=0)), int(values.size)


def compute_or_load_training_statistics(
    root: str | Path = DEFAULT_ROOT,
    cache_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Exact train-only population stats; stream TIFFs and cache the result."""
    root_path = Path(root).expanduser().resolve()
    scenarios, _ = discover_scenarios(root_path, TRAIN_CITIES)
    fingerprint = _statistics_fingerprint(scenarios)
    cache = (
        Path(cache_path).expanduser().resolve()
        if cache_path is not None
        else root_path / ".floodsimbench_train_statistics.json"
    )
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("format_version") == 2 and saved.get("fingerprint") == fingerprint:
            return saved

    water_count = 0
    water_sum = 0.0
    water_sum2 = 0.0
    for scenario_index, scenario in enumerate(scenarios):
        dem, dem_nodata = read_tiff(scenario.dem_path)
        dem_valid = valid_values(dem, dem_nodata)
        for frame_path in scenario.frame_paths:
            water, nodata = read_tiff(frame_path)
            valid = valid_values(water, nodata) & dem_valid
            finite = water[valid].double()
            water_count += int(finite.numel())
            water_sum += float(finite.sum())
            water_sum2 += float(finite.square().sum())
            del water, valid, finite
        del dem, dem_valid
        if scenario_index % 4 == 0:
            gc.collect()
    water_mean = water_sum / water_count
    water_var = max(water_sum2 / water_count - water_mean * water_mean, 1e-12)
    rain_mean, rain_std, rain_count = _rainfall_statistics(scenarios)
    result = {
        "format_version": 2,
        "fingerprint": fingerprint,
        "root": str(root_path),
        "allowed_split": "train",
        "cities": list(TRAIN_CITIES),
        "domains": sorted({scenario.domain for scenario in scenarios}),
        "scenario_count": len(scenarios),
        "state_policy": "each_of_72_observed_frames_once; no_synthetic_h0",
        "rainfall_policy": "scalar_over_all_71_markov_transitions_per_scenario",
        "water_depth": {
            "unit": "m", "count": water_count,
            "mean": water_mean, "std": math.sqrt(water_var),
        },
        "rainfall": {
            "unit": "mm/h", "count": rain_count,
            "mean": rain_mean, "std": rain_std,
        },
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(cache)
    return result


def make_channel_normalizer(
    normalizer_class: Any,
    root: str | Path = DEFAULT_ROOT,
    cache_path: Optional[str | Path] = None,
) -> Any:
    stats = compute_or_load_training_statistics(root, cache_path)
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
        },
    )


def rollout_sequence(
    model: torch.nn.Module,
    sequence: FloodSimLoadedScenario,
    spec: Any,
    normalizer: Any,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
) -> Dict[str, Any]:
    """Strict streaming h5->h360 autoregressive rollout (never teacher forced)."""
    # Local import avoids coupling the shared data parser to one experiment.
    import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline

    scenario = sequence.scenario
    steps = TRANSITIONS_PER_SCENARIO if max_steps <= 0 else min(max_steps, TRANSITIONS_PER_SCENARIO)
    static_cpu = require_terrain_store().for_domain(scenario.domain)[0]
    static_valid = torch.isfinite(static_cpu)
    initial_state, initial_nodata = read_tiff(scenario.frame_paths[0])
    initial_valid = static_valid & valid_values(initial_state, initial_nodata)
    h_state = normalizer.normalize(initial_state, 0)
    h_state = torch.where(initial_valid, h_state, torch.zeros_like(h_state))
    static = torch.where(static_valid, static_cpu, torch.zeros_like(static_cpu))
    h_state = h_state[None, None].float().to(device)
    static = static[None, None].float().to(device)
    lead_metrics: List[Dict[str, Any]] = []
    current_truth = initial_state
    current_nodata = initial_nodata
    model.eval()
    with torch.no_grad():
        for transition in range(steps):
            target, target_nodata = read_tiff(scenario.frame_paths[transition + 1])
            valid = (
                static_valid
                & valid_values(current_truth, current_nodata)
                & valid_values(target, target_nodata)
            )
            rain_value = rainfall_rate_mm_h(scenario.rainfall_mm, transition)
            rain_norm = (rain_value - normalizer.mean[1]) / normalizer.std[1]
            rain = torch.full_like(target, rain_norm)
            rain = torch.where(valid, rain, torch.zeros_like(rain))[None, None].to(device)
            x = torch.cat((h_state, rain), dim=1)
            metadata = {
                "height": [int(target.shape[0])], "width": [int(target.shape[1])],
                "dt_minutes": [DT_MINUTES], "dx": [sequence.dx], "dy": [sequence.dy],
            }
            with baseline.autocast_context(amp):
                dh = model(x=x, static=static, metadata=metadata)
            next_state = baseline.constrain_nonnegative_water_state(
                h_state.float() + dh.float(), normalizer
            )
            if not torch.isfinite(next_state).all():
                raise FloatingPointError(
                    f"{scenario.scenario_id}: non-finite prediction at step {transition + 1}"
                )
            prediction_phys = normalizer.denormalize(next_state, 0)
            target_device = target[None, None].to(device)
            valid_device = valid[None, None].to(device)
            metrics = baseline.masked_step_metrics(prediction_phys, target_device, valid_device)
            metrics.update({
                "lead_step": transition + 1,
                "lead_minutes": (transition + 1) * DT_MINUTES,
                "valid_pixels": int(valid.sum()),
                "rainfall_mm_h": rain_value,
            })
            lead_metrics.append(metrics)
            h_state = torch.where(
                valid_device, next_state.detach(), torch.zeros_like(next_state)
            )
            current_truth, current_nodata = target, target_nodata
    names = [
        "rmse", "mae", "relative_l2", "nse", "corr",
        *(f"csi_{threshold:g}" for threshold in baseline.CSI_THRESHOLDS),
    ]
    overall = {
        name: baseline.mean_finite(step[name] for step in lead_metrics)
        for name in names
    }
    return {
        "sample_id": scenario.scenario_id,
        "source_path": str(scenario.frame_paths[0].parent),
        "states_available": FRAMES_PER_SCENARIO,
        "prediction_steps": steps,
        "duration_minutes": steps * DT_MINUTES,
        "start_time_minutes": DT_MINUTES,
        "end_time_minutes": DT_MINUTES + steps * DT_MINUTES,
        "overall": overall,
        "lead_metrics": lead_metrics,
    }


def describe_dataset(dataset: FloodSimBenchMarkovDataset, batch_size: Optional[int]) -> Dict[str, Any]:
    domains = sorted({scenario.domain for scenario in dataset.files})
    description = {
        "domain": dataset.spec.name,
        "role": dataset.spec.role,
        "path": dataset.spec.path,
        "sequence_files": len(dataset.files),
        "scenario_count": len(dataset.files),
        "markov_pairs": len(dataset),
        "states_per_scenario": dataset.time_length,
        "observed_frames_per_scenario": FRAMES_PER_SCENARIO,
        "domains": domains,
        "domain_count": len(domains),
        "channels": 3,
        "channel_names": list(CHANNEL_NAMES),
        "channel_units": list(CHANNEL_UNITS),
        "dimension_order": "x=[2,H,W], static=[1,H,W], y=[1,H,W]",
        "rainfall": "P mm/h for pair t_start=5..55 min, else 0 mm/h",
        "initial_state": "observed h5; no synthetic h0",
        "dt_minutes": DT_MINUTES,
        "dx_m": dataset.spec.dx,
        "dy_m": dataset.spec.dy,
        "micro_batch_size": batch_size,
        "file_format": "single-band float32 GeoTIFF",
        "terrain_static": True,
        "terrain_global_train_scale": dataset.store.terrain_global_train_scale,
        "ignored_tiffs": dataset.ignored_tiffs,
    }
    print(json.dumps(description, ensure_ascii=False))
    return description


def validate_rainfall_contract() -> Dict[str, Any]:
    values = rainfall_series_mm_h(82)
    if values[:11] != (82.0,) * 11 or values[11:] != (0.0,) * 60:
        raise AssertionError("82mm rectangular storm transition values are wrong")
    if rainfall_rate_mm_h(82, 10) != 82.0:  # h55 -> h60
        raise AssertionError("h55->h60 rainfall must be 82 mm/h")
    if rainfall_rate_mm_h(82, 11) != 0.0:  # h60 -> h65
        raise AssertionError("h60->h65 rainfall must be zero")
    represented_depth = sum(value * DT_MINUTES / 60.0 for value in values)
    expected_represented_depth = 82.0 * 55.0 / 60.0
    if not math.isclose(
        represented_depth, expected_represented_depth, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"82mm represented storm depth is {represented_depth}")
    scenarios, _ = discover_scenarios(DEFAULT_ROOT, ("HOU",))
    hou = next(s for s in scenarios if s.domain == "HOU001" and s.rainfall_mm == 82)
    if len(hou.frame_paths) != 72:
        raise AssertionError("HOU001_82mm must contain 72 observed frames")
    return {
        "scenario": hou.scenario_id,
        "markov_pairs": TRANSITIONS_PER_SCENARIO,
        "initial_state": "observed_h5",
        "rainfall_first_11_mm_h": list(values[:11]),
        "rainfall_last_60_unique_mm_h": sorted(set(values[11:])),
        "h55_to_h60_mm_h": values[10],
        "h60_to_h65_mm_h": values[11],
        "represented_depth_h5_to_h60_mm": represented_depth,
        "omitted_depth_h0_to_h5_mm": 82.0 * DT_MINUTES / 60.0,
        "file_storm_total_mm": 82.0,
    }
