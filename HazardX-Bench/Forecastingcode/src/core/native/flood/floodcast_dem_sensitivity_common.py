"""Shared, isolated DEM feature pipeline for FloodCastBench sensitivity runs.

This module deliberately leaves the existing experiment files untouched.  It
installs narrowly scoped hooks into the NaN-safe Physical-RoPE experiment:

* h, rainfall, h_next, MSE, optimizer, scheduler, sampling and validation stay
  exactly as in the baseline;
* the FM terrain channel is replaced by a DEM rebuilt from the published TIFF;
* the one-static-channel model is generalized to one or three static channels;
* autoregressive rollout uses the same rebuilt static feature as training.

The four source domains are equally weighted when a shared DEM/slope scale is
fitted.  Mozambique is transform-only and never contributes to fitted scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe_physical_rope as physical
from gphyt.model import model_specs
from gphyt.model.tokenizer.tokenizer import Detokenizer
from gphyt.model.transformer.physical_position_models import (
    SPACE_REF_METERS,
    build_physical_position_gphyt,
)


DEM_ROOT_DEFAULT = (
    "/mnt/HDD2/qingsong/qinqsong/Datasets/FloodCastBench/Relevant data/DEM"
)
SLOPE_REFERENCE = 0.01
SOURCE_REGIONS = ("australia", "pakistan", "uk_30m", "uk")
EXPERIMENT_KINDS = (
    "terrain_region_zscore",
    "terrain_region_zscore_slope",
    "terrain_centered_global_scaled",
)

# crop=(top, bottom_exclusive, left, right_exclusive); None means array edge.
DEM_SPECS: Mapping[str, Mapping[str, Any]] = {
    "australia": {
        "file": "Australia_DEM.tif",
        "crop": (None, None, None, None),
        "factor": 2,
        "shape": (536, 536),
        "dx": 60.0,
        "dy": 60.0,
    },
    "pakistan": {
        "file": "Pakistan_DEM.tif",
        "crop": (37, -33, 170, -57),
        "factor": 16,
        "shape": (810, 441),
        "dx": 480.0,
        "dy": 480.0,
    },
    "uk": {
        "file": "UK_DEM.tif",
        "crop": (5, -5, 5, -5),
        "factor": 2,
        "shape": (85, 137),
        "dx": 60.0,
        "dy": 60.0,
    },
    "uk_30m": {
        "file": "UK_DEM.tif",
        "crop": (5, -5, 5, -5),
        "factor": 1,
        "shape": (170, 275),
        "dx": 30.0,
        "dy": 30.0,
    },
    "mozambique": {
        "file": "Mozambique_DEM.tif",
        "crop": (150, -50, 50, -50),
        "factor": 16,
        "shape": (151, 138),
        "dx": 480.0,
        "dy": 480.0,
    },
}

STATIC_NAMES: Mapping[str, Tuple[str, ...]] = {
    "terrain_region_zscore": ("terrain_region_zscore",),
    "terrain_region_zscore_slope": (
        "terrain_region_zscore",
        "slope_x_asinh_global_scaled",
        "slope_y_asinh_global_scaled",
    ),
    "terrain_centered_global_scaled": (
        "terrain_centered_global_scaled",
    ),
}


def canonical_region(domain_name: str) -> str:
    """Map train/valid/test split names onto one physical DEM region."""
    name = str(domain_name).lower()
    if name.startswith("mozambique"):
        return "mozambique"
    if name in DEM_SPECS:
        return name
    raise KeyError(f"No physical DEM configured for domain {domain_name!r}")


def _crop(array: np.ndarray, crop: Sequence[Optional[int]]) -> np.ndarray:
    top, bottom, left, right = crop
    return array[
        slice(top, bottom),
        slice(left, right),
    ]


def dynamics_inter(array: np.ndarray, factor: int) -> torch.Tensor:
    """Reproduce dynamics ``inter()`` on the endpoint-aligned target grid.

    The original SciPy ``interp2d`` creates ``floor(H/factor)`` by
    ``floor(W/factor)`` coordinates spanning both original endpoints.  For a
    regular grid, bilinear ``interpolate(..., align_corners=True)`` is the same
    operation and remains available in current SciPy/PyTorch environments.
    """
    values = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if int(factor) == 1:
        return values.clone()
    height, width = (int(v) for v in values.shape)
    output_size = (height // int(factor), width // int(factor))
    return F.interpolate(
        values.unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=True,
    )[0, 0]


def _repair_dynamics_nodata(dem: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Match ``process_dem``: values below -2000 become max(valid)+30 m."""
    repaired = dem.float().clone()
    bad = (~torch.isfinite(repaired)) | (repaired < -2000.0)
    good = ~bad
    if not bool(good.any()):
        raise ValueError("Processed DEM contains no valid elevation")
    fill_value = repaired[good].max() + 30.0
    repaired[bad] = fill_value
    if not torch.isfinite(repaired).all():
        raise FloatingPointError("DEM nodata repair left NaN/Inf")
    return repaired, int(bad.sum().item())


def _population_stats(values: torch.Tensor) -> Tuple[float, float]:
    finite = values[torch.isfinite(values)].double()
    if finite.numel() == 0:
        raise ValueError("Cannot compute DEM statistics without finite values")
    mean = finite.mean()
    std = finite.std(correction=0)
    if not bool(torch.isfinite(std)) or float(std) <= 0.0:
        raise ValueError(f"Invalid DEM population std {float(std)}")
    return float(mean), float(std)


def _physical_slopes(
    terrain_m: torch.Tensor,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference dz/dx and dz/dy in raster column/row directions."""
    z = terrain_m.float()
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError(f"DEM must be a 2-D grid of at least 2x2, got {z.shape}")
    slope_x = torch.empty_like(z)
    slope_y = torch.empty_like(z)
    slope_x[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * float(dx))
    slope_x[:, 0] = (z[:, 1] - z[:, 0]) / float(dx)
    slope_x[:, -1] = (z[:, -1] - z[:, -2]) / float(dx)
    slope_y[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2.0 * float(dy))
    slope_y[0, :] = (z[1, :] - z[0, :]) / float(dy)
    slope_y[-1, :] = (z[-1, :] - z[-2, :]) / float(dy)
    if not torch.isfinite(slope_x).all() or not torch.isfinite(slope_y).all():
        raise FloatingPointError("Slope calculation produced NaN/Inf")
    return slope_x, slope_y


def _sha256_tensor(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class DemFeatureStore:
    """Build and cache all region-specific physical DEM features once."""

    def __init__(self, kind: str, dem_root: str | Path):
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(f"Unknown DEM experiment {kind!r}")
        self.kind = kind
        self.dem_root = Path(dem_root).expanduser().resolve()
        if not self.dem_root.is_dir():
            raise FileNotFoundError(f"DEM root does not exist: {self.dem_root}")
        self.terrain_m: Dict[str, torch.Tensor] = {}
        self.features: Dict[str, torch.Tensor] = {}
        self.region_stats: Dict[str, Dict[str, Any]] = {}
        self.terrain_global_train_scale: Optional[float] = None
        self.slope_transform_scale: Optional[float] = None
        self._build()

    def _load_region(self, region: str) -> torch.Tensor:
        spec = DEM_SPECS[region]
        path = self.dem_root / str(spec["file"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {region} DEM: {path}")
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            raw = np.asarray(image, dtype=np.float32)
        cropped = _crop(raw, spec["crop"])
        downsampled = dynamics_inter(cropped, int(spec["factor"]))
        repaired, filled_cells = _repair_dynamics_nodata(downsampled)
        expected = tuple(int(v) for v in spec["shape"])
        if tuple(repaired.shape) != expected:
            raise AssertionError(
                f"{region}: rebuilt DEM shape {tuple(repaired.shape)} != "
                f"FM shape {expected}"
            )
        mean, std = _population_stats(repaired)
        self.region_stats[region] = {
            "dem_path": str(path),
            "crop": list(spec["crop"]),
            "downsample_factor": int(spec["factor"]),
            "interpolation": "bilinear_endpoint_aligned_align_corners_true",
            "shape": list(expected),
            "dx_m": float(spec["dx"]),
            "dy_m": float(spec["dy"]),
            "nodata_rule": "value_below_-2000_to_valid_max_plus_30m",
            "filled_output_cells": filled_cells,
            "terrain_mean_m": mean,
            "terrain_population_std_m": std,
            "processed_dem_sha256": _sha256_tensor(repaired),
        }
        return repaired

    def _build(self) -> None:
        for region in DEM_SPECS:
            self.terrain_m[region] = self._load_region(region)

        centered: Dict[str, torch.Tensor] = {}
        zscore: Dict[str, torch.Tensor] = {}
        for region, terrain in self.terrain_m.items():
            mean = float(self.region_stats[region]["terrain_mean_m"])
            std = float(self.region_stats[region]["terrain_population_std_m"])
            centered[region] = terrain.float() - mean
            zscore[region] = centered[region] / std

        if self.kind == "terrain_region_zscore":
            self.features = {
                region: value.unsqueeze(0).float()
                for region, value in zscore.items()
            }
        elif self.kind == "terrain_centered_global_scaled":
            domain_second_moments = torch.tensor(
                [
                    centered[region].double().square().mean()
                    for region in SOURCE_REGIONS
                ],
                dtype=torch.float64,
            )
            scale = torch.sqrt(domain_second_moments.mean())
            if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
                raise ValueError(f"Invalid global centered DEM scale {float(scale)}")
            self.terrain_global_train_scale = float(scale)
            self.features = {
                region: (value / self.terrain_global_train_scale)
                .unsqueeze(0)
                .float()
                for region, value in centered.items()
            }
        else:
            transformed_slopes: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for region, terrain in self.terrain_m.items():
                spec = DEM_SPECS[region]
                slope_x, slope_y = _physical_slopes(
                    terrain,
                    dx=float(spec["dx"]),
                    dy=float(spec["dy"]),
                )
                tx = torch.asinh(slope_x / SLOPE_REFERENCE)
                ty = torch.asinh(slope_y / SLOPE_REFERENCE)
                transformed_slopes[region] = (tx, ty)
                self.region_stats[region].update(
                    {
                        "slope_x_rms": float(
                            torch.sqrt(slope_x.double().square().mean())
                        ),
                        "slope_y_rms": float(
                            torch.sqrt(slope_y.double().square().mean())
                        ),
                    }
                )
            # Each source domain receives 1/4 total weight.  Within a domain,
            # x/y components and all grid cells receive equal weight.
            domain_second_moments = torch.tensor(
                [
                    0.5
                    * (
                        transformed_slopes[region][0].double().square().mean()
                        + transformed_slopes[region][1].double().square().mean()
                    )
                    for region in SOURCE_REGIONS
                ],
                dtype=torch.float64,
            )
            scale = torch.sqrt(domain_second_moments.mean())
            if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
                raise ValueError(f"Invalid global transformed-slope scale {scale}")
            self.slope_transform_scale = float(scale)
            self.features = {
                region: torch.stack(
                    (
                        zscore[region],
                        transformed_slopes[region][0] / self.slope_transform_scale,
                        transformed_slopes[region][1] / self.slope_transform_scale,
                    ),
                    dim=0,
                ).float()
                for region in DEM_SPECS
            }

        expected_channels = len(STATIC_NAMES[self.kind])
        for region, feature in self.features.items():
            expected_shape = (
                expected_channels,
                *tuple(int(v) for v in DEM_SPECS[region]["shape"]),
            )
            if tuple(feature.shape) != expected_shape:
                raise AssertionError(
                    f"{region}: feature shape {tuple(feature.shape)} != "
                    f"{expected_shape}"
                )
            if feature.dtype != torch.float32:
                raise AssertionError(f"{region}: DEM feature must be float32")
            if not torch.isfinite(feature).all():
                raise FloatingPointError(f"{region}: DEM feature contains NaN/Inf")
            self.region_stats[region]["feature_sha256"] = _sha256_tensor(feature)

    @property
    def static_channels(self) -> int:
        return len(STATIC_NAMES[self.kind])

    def for_domain(self, domain_name: str) -> torch.Tensor:
        return self.features[canonical_region(domain_name)]

    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dem_root": str(self.dem_root),
            "static_feature_names": list(STATIC_NAMES[self.kind]),
            "static_channels": self.static_channels,
            "source_domains_for_shared_scale": list(SOURCE_REGIONS),
            "source_domain_weights": [0.25, 0.25, 0.25, 0.25],
            "shared_scale_pixel_weighting": (
                "equal_domains_then_equal_cells_and_xy_components_within_domain"
            ),
            "slope_reference": (
                SLOPE_REFERENCE
                if self.kind == "terrain_region_zscore_slope"
                else None
            ),
            "slope_transform_scale": self.slope_transform_scale,
            "terrain_global_train_scale": self.terrain_global_train_scale,
            "region_stats": self.region_stats,
        }


_ACTIVE_STORE: Optional[DemFeatureStore] = None
_ACTIVE_KIND: Optional[str] = None
_ORIGINAL_VALIDATE_ARGS = baseline.validate_args
_ORIGINAL_COMPUTE_NORMALIZER = baseline.compute_training_normalizer
_ORIGINAL_LOAD_CHECKPOINT = baseline.load_checkpoint_file
_ORIGINAL_DESCRIBE_DATASET = baseline.describe_dataset


def _require_store() -> DemFeatureStore:
    if _ACTIVE_STORE is None:
        raise RuntimeError("DEM feature store has not been configured")
    return _ACTIVE_STORE


def _configure_store(kind: str, dem_root: str | Path) -> DemFeatureStore:
    global _ACTIVE_STORE, _ACTIVE_KIND
    root = Path(dem_root).expanduser().resolve()
    if (
        _ACTIVE_STORE is None
        or _ACTIVE_STORE.kind != kind
        or _ACTIVE_STORE.dem_root != root
    ):
        _ACTIVE_STORE = DemFeatureStore(kind, root)
    _ACTIVE_KIND = kind
    return _ACTIVE_STORE


class DemFeatureFloodDomainDataset(baseline.FloodDomainDataset):
    """Baseline Markov pairs with only the static terrain branch replaced."""

    def __init__(
        self,
        spec: baseline.DomainSpec,
        normalizer: baseline.ChannelNormalizer,
        noise_std: float = 0.0,
    ):
        super().__init__(spec, normalizer, noise_std)
        self.dem_feature = _require_store().for_domain(spec.name)
        if tuple(self.dem_feature.shape[-2:]) != (
            self.profile.height,
            self.profile.width,
        ):
            raise AssertionError(
                f"{spec.name}: DEM H/W {tuple(self.dem_feature.shape[-2:])} "
                f"!= FM H/W {(self.profile.height, self.profile.width)}"
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_index = index // self.pairs_per_sequence
        t = index % self.pairs_per_sequence
        sequence = self._load_sequence(sequence_index)
        data = sequence.data
        h_t = data[t, :, :, self.spec.water_channel]
        rain_t = data[t, :, :, self.spec.rain_channel]
        h_next = data[t + 1, :, :, self.spec.water_channel]

        valid = self._explicit_pair_mask(sequence, t)
        for values in (h_t, rain_t, h_next):
            valid &= baseline._valid_values(values, self.spec.nodata_value)
        valid &= torch.isfinite(self.dem_feature).all(dim=0)
        if not bool(valid.any()):
            raise ValueError(
                f"{self.spec.name}/{self.files[sequence_index].name}@{t}: "
                "no valid pixels in Markov pair"
            )

        # These are intentionally the untouched baseline train_standard
        # transforms and normalized h_next target.
        h_t_n = self.normalizer.normalize(h_t, baseline.WATER_CHANNEL)
        rain_n = self.normalizer.normalize(rain_t, baseline.RAIN_CHANNEL)
        h_next_n = self.normalizer.normalize(h_next, baseline.WATER_CHANNEL)
        h_t_n = torch.where(valid, h_t_n, torch.zeros_like(h_t_n))
        rain_n = torch.where(valid, rain_n, torch.zeros_like(rain_n))
        h_next_n = torch.where(valid, h_next_n, torch.zeros_like(h_next_n))
        static = torch.where(
            valid.unsqueeze(0),
            self.dem_feature,
            torch.zeros_like(self.dem_feature),
        ).float()
        x = torch.stack((h_t_n, rain_n), dim=0).float()
        y = h_next_n.unsqueeze(0).float()
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * valid.unsqueeze(0)

        height, width = int(data.shape[1]), int(data.shape[2])
        return {
            "x": x,
            "y": y,
            "static": static,
            "mask": valid.unsqueeze(0),
            "domain": self.spec.name,
            "domain_id": self.spec.domain_id,
            "sample_id": self.files[sequence_index].stem,
            "source_path": str(self.files[sequence_index]),
            "time_index": t,
            "dt_minutes": self.spec.dt_minutes,
            "dx": sequence.dx,
            "dy": sequence.dy,
            "height": height,
            "width": width,
        }


def _dem_frame_mask(
    sequence: baseline.LoadedSequence,
    spec: baseline.DomainSpec,
    t: int,
    include_next: bool,
) -> torch.Tensor:
    """Frame mask that does not read the obsolete FM terrain channel."""
    data = sequence.data
    values = [
        data[t, :, :, spec.water_channel],
        data[t, :, :, spec.rain_channel],
    ]
    if include_next:
        values.append(data[t + 1, :, :, spec.water_channel])
    valid = torch.ones(data.shape[1:3], dtype=torch.bool)
    if sequence.mask is not None:
        valid &= sequence.mask[t]
        if include_next:
            valid &= sequence.mask[t + 1]
    for value in values:
        valid &= baseline._valid_values(value, spec.nodata_value)
    valid &= torch.isfinite(_require_store().for_domain(spec.name)).all(dim=0)
    return valid


def rollout_sequence(
    model: nn.Module,
    sequence: baseline.LoadedSequence,
    spec: baseline.DomainSpec,
    normalizer: baseline.ChannelNormalizer,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
) -> Dict[str, Any]:
    """Baseline rollout with the identical DEM feature used by training."""
    data = sequence.data
    if data.ndim != 4:
        raise AssertionError(f"Expected [T,H,W,C], got {tuple(data.shape)}")
    total_steps = int(data.shape[0]) - 1
    if max_steps > 0:
        total_steps = min(total_steps, int(max_steps))
    if total_steps < 1:
        raise ValueError(f"{sequence.source_path}: no rollout target steps")

    static_cpu = _require_store().for_domain(spec.name)
    if tuple(static_cpu.shape[-2:]) != tuple(data.shape[1:3]):
        raise AssertionError("Rollout DEM and FM H/W do not match")
    initial_valid = _dem_frame_mask(sequence, spec, 0, include_next=False)
    h_state = normalizer.normalize(
        data[0, :, :, spec.water_channel], baseline.WATER_CHANNEL
    )
    h_state = torch.where(initial_valid, h_state, torch.zeros_like(h_state))
    static_cpu = torch.where(
        initial_valid.unsqueeze(0), static_cpu, torch.zeros_like(static_cpu)
    )
    h_state = h_state.unsqueeze(0).unsqueeze(0).float().to(device)
    static = static_cpu.unsqueeze(0).float().to(device)

    lead_metrics = []
    model.eval()
    with torch.no_grad():
        for t in range(total_steps):
            valid = _dem_frame_mask(sequence, spec, t, include_next=True)
            rain = normalizer.normalize(
                data[t, :, :, spec.rain_channel], baseline.RAIN_CHANNEL
            )
            rain = torch.where(valid, rain, torch.zeros_like(rain))
            rain = rain.unsqueeze(0).unsqueeze(0).float().to(device)
            x = torch.cat((h_state, rain), dim=1)
            metadata = {
                "height": [int(data.shape[1])],
                "width": [int(data.shape[2])],
                "dt_minutes": [spec.dt_minutes],
                "dx": [sequence.dx],
                "dy": [sequence.dy],
            }
            with baseline.autocast_context(amp):
                dh = model(x=x, static=static, metadata=metadata)
            next_state = baseline.constrain_nonnegative_water_state(
                h_state.float() + dh.float(), normalizer
            )
            if not torch.isfinite(next_state).all():
                raise FloatingPointError(
                    f"{spec.name}/{Path(sequence.source_path).name}: "
                    f"non-finite prediction at lead {t + 1}"
                )
            prediction_phys = normalizer.denormalize(
                next_state, baseline.WATER_CHANNEL
            )
            target_phys = data[
                t + 1, :, :, spec.water_channel
            ].unsqueeze(0).unsqueeze(0).to(device)
            valid_device = valid.unsqueeze(0).unsqueeze(0).to(device)
            metrics = baseline.masked_step_metrics(
                prediction_phys, target_phys, valid_device
            )
            metrics.update(
                {
                    "lead_step": t + 1,
                    "lead_minutes": (t + 1) * spec.dt_minutes,
                    "valid_pixels": int(valid.sum().item()),
                }
            )
            lead_metrics.append(metrics)
            next_valid = _dem_frame_mask(
                sequence,
                spec,
                min(t + 1, int(data.shape[0]) - 1),
                include_next=False,
            ).unsqueeze(0).unsqueeze(0).to(device)
            h_state = torch.where(
                next_valid, next_state.detach(), torch.zeros_like(next_state)
            )

    metric_names = [
        "rmse",
        "mae",
        "relative_l2",
        "nse",
        "corr",
        *(f"csi_{threshold:g}" for threshold in baseline.CSI_THRESHOLDS),
    ]
    overall = {
        key: baseline.mean_finite(step[key] for step in lead_metrics)
        for key in metric_names
    }
    return {
        "sample_id": Path(sequence.source_path).stem,
        "source_path": sequence.source_path,
        "states_available": int(data.shape[0]),
        "prediction_steps": total_steps,
        "duration_minutes": total_steps * spec.dt_minutes,
        "overall": overall,
        "lead_metrics": lead_metrics,
    }


class NativeGridPhysicalPositionStaticEarthDynamics(nn.Module):
    """Physical-RoPE baseline generalized only in static input channel count."""

    def __init__(
        self,
        patch_height: int = 16,
        patch_width: int = 16,
        model_size: str = "GPT_S",
        attention_mode: str = "spatial",
        tokenizer_mode: str = "linear",
        detokenizer_mode: str = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        use_derivatives: bool = False,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        direct_derivative_output: bool = True,
        position_encoding: str = "physical_rope",
        space_ref: float = SPACE_REF_METERS,
        static_channels: int = 1,
        dem_feature_kind: str = "terrain_region_zscore",
    ) -> None:
        super().__init__()
        if not direct_derivative_output:
            raise ValueError("Direct derivative output must remain enabled")
        if patch_height < 1 or patch_width < 1 or static_channels < 1:
            raise ValueError("Patch dimensions/static_channels must be positive")
        if attention_mode not in {"spatial", "full"}:
            raise ValueError(f"Unsupported attention mode {attention_mode!r}")
        if position_encoding != "physical_rope":
            raise ValueError("These sensitivity experiments require physical_rope")
        if not math.isfinite(space_ref) or space_ref <= 0:
            raise ValueError("space_ref must be finite and positive")
        if dem_feature_kind not in EXPERIMENT_KINDS:
            raise ValueError(f"Unknown DEM feature kind {dem_feature_kind!r}")
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.static_channels = int(static_channels)
        self.input_channels = 2 + self.static_channels
        self.position_encoding = position_encoding
        self.space_ref = float(space_ref)
        self.dem_feature_kind = dem_feature_kind
        patch_size = (1, self.patch_height, self.patch_width)
        gpt_spec = getattr(model_specs, model_size)()
        model_config = {
            "architecture": "gphyt",
            "img_size": (1, self.patch_height, self.patch_width),
            "transformer": {
                "input_channels": self.input_channels,
                "model_size": model_size,
                "att_mode": attention_mode,
                "dropout": dropout,
                "position_encoding": position_encoding,
                "space_ref": self.space_ref,
                "patch_size": list(patch_size),
                "stochastic_depth_rate": stochastic_depth_rate,
                "use_derivatives": use_derivatives,
                "integrator": None,
            },
            "tokenizer": {
                "tokenizer_mode": tokenizer_mode,
                "detokenizer_mode": detokenizer_mode,
                "tokenizer_overlap": tokenizer_overlap,
                "detokenizer_overlap": detokenizer_overlap,
            },
        }
        self.backbone = build_physical_position_gphyt(
            model_config,
            hidden_dim=gpt_spec.hidden_dim,
            mlp_dim=gpt_spec.mlp_dim,
            num_heads=gpt_spec.num_heads,
            num_layers=gpt_spec.num_layers,
            conv_channels=gpt_spec.conv_channels,
        )
        self.backbone.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=gpt_spec.hidden_dim,
            out_channels=1,
            mode=detokenizer_mode,
            conv_net_channels=gpt_spec.conv_channels,
            overlap=detokenizer_overlap,
            img_size=(1, self.patch_height, self.patch_width),
        )
        self.direct_derivative_output = True

    def padding_for(self, height: int, width: int) -> Tuple[int, int]:
        return (
            (-int(height)) % self.patch_height,
            (-int(width)) % self.patch_width,
        )

    @staticmethod
    def _metadata_values(
        metadata: Optional[Mapping[str, Any]],
        key: str,
        required: bool,
    ) -> Any:
        value = None if metadata is None else metadata.get(key)
        if required and value is None:
            raise ValueError(f"metadata[{key!r}] is required")
        return value

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise AssertionError(f"x must be [B,2,H,W], got {tuple(x.shape)}")
        if (
            static is None
            or static.ndim != 4
            or static.shape[1] != self.static_channels
        ):
            raise AssertionError(
                f"static must be [B,{self.static_channels},H,W], got "
                f"{None if static is None else tuple(static.shape)}"
            )
        if x.shape[0] != static.shape[0] or x.shape[-2:] != static.shape[-2:]:
            raise AssertionError("x/static batch or H/W mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(static).all():
            raise FloatingPointError("Model input contains NaN/Inf")
        batch_size, _, height, width = x.shape
        for key, expected in (("height", height), ("width", width)):
            values = self._metadata_values(metadata, key, required=False)
            if values is not None:
                values = torch.as_tensor(values).reshape(-1)
                if values.numel() not in {1, batch_size} or bool(
                    (values != expected).any()
                ):
                    raise AssertionError(f"metadata {key} mismatch")
        dx = self._metadata_values(metadata, "dx", required=True)
        dy = self._metadata_values(metadata, "dy", required=True)

        fields = torch.cat((x, static), dim=1)
        pad_h, pad_w = self.padding_for(height, width)
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), value=0.0)
        fields = fields.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        derivative_fields = self.backbone(fields, dx=dx, dy=dy)
        if derivative_fields.shape[-1] != 1:
            raise AssertionError("Backbone must return one derivative channel")
        dh = (
            derivative_fields[:, 0, :height, :width, :]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        expected_shape = (batch_size, 1, height, width)
        if tuple(dh.shape) != expected_shape:
            raise AssertionError(f"dh shape {tuple(dh.shape)} != {expected_shape}")
        if not torch.isfinite(dh).all():
            raise FloatingPointError("Model output dh contains NaN/Inf")
        return dh


def _compute_training_normalizer(
    train_specs: Mapping[str, baseline.DomainSpec],
) -> baseline.ChannelNormalizer:
    """Keep exact baseline h/rain stats; mark static features pre-normalized."""
    original = _ORIGINAL_COMPUTE_NORMALIZER(train_specs)
    provenance = dict(original.provenance)
    provenance["dem_sensitivity"] = _require_store().metadata()
    provenance["channel_policy"] = {
        "water_depth": "unchanged_baseline_global_train_standard",
        "rainfall": "unchanged_baseline_global_train_standard",
        "terrain_slot": (
            "identity_placeholder; physical DEM feature is already transformed "
            "by the experiment-specific static pipeline"
        ),
    }
    return baseline.ChannelNormalizer(
        mode=original.mode,
        mean=(original.mean[0], original.mean[1], 0.0),
        std=(original.std[0], original.std[1], 1.0),
        provenance=provenance,
    )


def _describe_dataset(
    dataset: DemFeatureFloodDomainDataset,
    batch_size: Optional[int],
) -> Dict[str, Any]:
    description = _ORIGINAL_DESCRIBE_DATASET(dataset, batch_size)
    description.update(
        {
            "static_source": "published_physical_dem_tiff_not_fm_channel_2",
            "static_feature_names": list(STATIC_NAMES[_require_store().kind]),
            "static_channels": _require_store().static_channels,
        }
    )
    print(
        json.dumps(
            {
                "domain": dataset.spec.name,
                "static_source": description["static_source"],
                "static_feature_names": description["static_feature_names"],
                "static_shape": list(dataset.dem_feature.shape),
            },
            ensure_ascii=False,
        )
    )
    return description


def _load_checkpoint_file(
    path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = _ORIGINAL_LOAD_CHECKPOINT(path, device)
    saved = (
        checkpoint.get("normalization", {})
        .get("provenance", {})
        .get("dem_sensitivity")
    )
    if not isinstance(saved, Mapping):
        raise RuntimeError(
            "Checkpoint has no DEM-sensitivity metadata; old baseline "
            "checkpoints are intentionally incompatible"
        )
    current = _require_store().metadata()
    if saved.get("kind") != current["kind"]:
        raise RuntimeError(
            f"Checkpoint DEM kind {saved.get('kind')!r} != {current['kind']!r}"
        )
    if saved.get("static_feature_names") != current["static_feature_names"]:
        raise RuntimeError("Checkpoint static feature channels do not match")
    for region in DEM_SPECS:
        saved_hash = saved["region_stats"][region]["feature_sha256"]
        current_hash = current["region_stats"][region]["feature_sha256"]
        if saved_hash != current_hash:
            raise RuntimeError(
                f"{region}: current DEM feature differs from checkpoint metadata"
            )
    return checkpoint


def run_experiment(kind: str, default_output_dir: str) -> None:
    """Install the isolated hooks and enter the unchanged baseline main loop."""
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(kind)

    def build_parser() -> argparse.ArgumentParser:
        parser = physical.build_parser()
        parser.description = (
            f"FloodCastBench Physical-RoPE DEM sensitivity: {kind}; "
            "h/rain/loss/optimizer/scheduler/sampling remain baseline-identical"
        )
        parser.add_argument(
            "--dem-root",
            default=DEM_ROOT_DEFAULT,
            help="Directory containing Australia/Pakistan/UK/Mozambique DEM TIFFs",
        )
        parser.set_defaults(
            output_dir=default_output_dir,
            normalization="train_standard",
            loss="mse",
            position_encoding="physical_rope",
        )
        return parser

    def validate_args(args: argparse.Namespace) -> None:
        _ORIGINAL_VALIDATE_ARGS(args)
        _configure_store(kind, args.dem_root)

    def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
        config = physical.model_config_from_args(args)
        config.update(
            {
                "static_channels": len(STATIC_NAMES[kind]),
                "dem_feature_kind": kind,
            }
        )
        return config

    def build_model(
        config: Mapping[str, Any],
        device: torch.device,
    ) -> nn.Module:
        if config.get("dem_feature_kind") != kind:
            raise RuntimeError(
                f"Model/checkpoint DEM kind {config.get('dem_feature_kind')!r} "
                f"does not match running experiment {kind!r}"
            )
        model = NativeGridPhysicalPositionStaticEarthDynamics(**dict(config))
        if model.static_channels != len(STATIC_NAMES[kind]):
            raise AssertionError("Model static channel count is inconsistent")
        if model.backbone.coordinate_embedding is not None:
            raise AssertionError("Physical RoPE must not use absolute coordinates")
        if any(
            block.attention.rope is None
            or block.attention.rope.head_dim != block.attention.head_dim
            for block in model.backbone.attention_blocks
        ):
            raise AssertionError("Physical RoPE must rotate post-head Q/K")
        return model.to(device)

    baseline.build_parser = build_parser
    baseline.validate_args = validate_args
    baseline.compute_training_normalizer = _compute_training_normalizer
    baseline.FloodDomainDataset = DemFeatureFloodDomainDataset
    baseline.rollout_sequence = rollout_sequence
    baseline.describe_dataset = _describe_dataset
    baseline.load_checkpoint_file = _load_checkpoint_file
    baseline.model_config_from_args = model_config_from_args
    baseline.build_model = build_model
    baseline.main()

