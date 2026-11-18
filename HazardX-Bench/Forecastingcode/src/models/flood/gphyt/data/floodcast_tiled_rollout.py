"""Strict full-grid FloodCastBench rollout through 128/96 tiled inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
from gphyt.data.floodcast_window_utils import tiled_model_delta


_WINDOW_SIZE = 128
_STRIDE = 96
_TILE_BATCH_SIZE = 1
_BLEND_FLOOR = 0.05

# Audited over the ten fixed Pakistan training tensors only.  The rollout cap
# is deliberately one observed 5-minute change above the largest training
# state, so validation/test values are never used to tune the guard.
GPHYT_PAKISTAN_TRAIN_WATER_MAX_M = 26.636995315551758
GPHYT_PAKISTAN_TRAIN_MAX_ABS_5MIN_CHANGE_M = 8.002134323120117
GPHYT_PAKISTAN_ROLLOUT_WATER_CAP_M = (
    GPHYT_PAKISTAN_TRAIN_WATER_MAX_M
    + GPHYT_PAKISTAN_TRAIN_MAX_ABS_5MIN_CHANGE_M
)
_GPHYT_MODEL_NAME = "GPhyT-S-PT-FT"


def _rollout_model_name(model: nn.Module) -> str:
    """Resolve the public seven-baseline name without depending on globals."""

    current = model
    if hasattr(current, "module") and isinstance(current.module, nn.Module):
        current = current.module
    return str(getattr(current, "model_name", ""))


def project_gphyt_pakistan_rollout_state(
    model: nn.Module,
    normalized_state: torch.Tensor,
    normalizer: baseline.ChannelNormalizer,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Project only unstable GPhyT Pakistan feedback to training support.

    GPhyT is trained with one-step teacher forcing.  Its direct ``h_next``
    mapping can therefore have closed-loop gain above one even when every
    one-step training value is finite.  The fixed Pakistan run demonstrated
    exponential feedback growth from tens of metres to NaN/Inf.  Keep the
    formal rollout autoregressive, but prevent the next model input from
    leaving a training-derived physical envelope.

    Non-finite values are rejected before projection; this is a stability
    constraint, not a NaN replacement policy.
    """

    if not torch.isfinite(normalized_state).all():
        raise FloatingPointError("rollout state is non-finite before projection")
    normalization_domains = tuple(
        str(value) for value in normalizer.provenance.get("domains", ())
    )
    enabled = (
        _rollout_model_name(model) == _GPHYT_MODEL_NAME
        and normalization_domains == ("pakistan",)
    )
    if not enabled:
        return normalized_state, {
            "enabled": False,
            "policy": "none",
            "clipped_pixels": 0,
            "clipped_fraction": 0.0,
        }

    water_mean = float(normalizer.mean[baseline.WATER_CHANNEL])
    water_std = float(normalizer.std[baseline.WATER_CHANNEL])
    cap_normalized = (
        float(GPHYT_PAKISTAN_ROLLOUT_WATER_CAP_M) - water_mean
    ) / water_std
    clipped = normalized_state > cap_normalized
    clipped_pixels = int(clipped.sum().detach().cpu())
    projected = torch.clamp_max(normalized_state, cap_normalized)
    if not torch.isfinite(projected).all():
        raise FloatingPointError("GPhyT rollout projection produced NaN/Inf")
    total_pixels = int(normalized_state.numel())
    pre_projection_max_m = float(
        (normalized_state.max().detach().cpu() * water_std) + water_mean
    )
    post_projection_max_m = float(
        (projected.max().detach().cpu() * water_std) + water_mean
    )
    return projected, {
        "enabled": True,
        "policy": "pakistan_training_support_upper_projection",
        "normalization_domains": list(normalization_domains),
        "training_water_max_m": float(GPHYT_PAKISTAN_TRAIN_WATER_MAX_M),
        "training_max_abs_5min_change_m": float(
            GPHYT_PAKISTAN_TRAIN_MAX_ABS_5MIN_CHANGE_M
        ),
        "water_cap_m": float(GPHYT_PAKISTAN_ROLLOUT_WATER_CAP_M),
        "water_cap_normalized": float(cap_normalized),
        "clipped_pixels": clipped_pixels,
        "clipped_fraction": (
            float(clipped_pixels) / float(total_pixels) if total_pixels else 0.0
        ),
        "pre_projection_max_m": pre_projection_max_m,
        "post_projection_max_m": post_projection_max_m,
        "non_finite_policy": "raise",
    }


def configure_tiled_rollout(
    *,
    window_size: int,
    stride: int,
    tile_batch_size: int,
    blend_floor: float = 0.05,
) -> None:
    global _WINDOW_SIZE, _STRIDE, _TILE_BATCH_SIZE, _BLEND_FLOOR
    if window_size < 1 or stride < 1 or tile_batch_size < 1:
        raise ValueError("window_size/stride/tile_batch_size must be positive")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size because it leaves gaps")
    if not math.isfinite(blend_floor) or not 0.0 < blend_floor <= 1.0:
        raise ValueError("blend_floor must be in (0,1]")
    _WINDOW_SIZE = int(window_size)
    _STRIDE = int(stride)
    _TILE_BATCH_SIZE = int(tile_batch_size)
    _BLEND_FLOOR = float(blend_floor)


def tiled_rollout_sequence(
    model: nn.Module,
    sequence: baseline.LoadedSequence,
    spec: baseline.DomainSpec,
    normalizer: baseline.ChannelNormalizer,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
) -> Dict[str, Any]:
    """Stitch every full next state before feeding it to the next lead time."""

    data = sequence.data
    if data.ndim != 4:
        raise AssertionError(f"Expected [T,H,W,C], got {tuple(data.shape)}")
    total_steps = int(data.shape[0]) - 1
    if max_steps > 0:
        total_steps = min(total_steps, int(max_steps))
    if total_steps < 1:
        raise ValueError(f"{sequence.source_path}: no rollout target steps")

    static_cpu = dem_common._require_store().for_domain(spec.name)
    if tuple(static_cpu.shape[-2:]) != tuple(data.shape[1:3]):
        raise AssertionError("Rollout DEM and FloodCastBench H/W do not match")
    initial_valid = dem_common._dem_frame_mask(
        sequence, spec, 0, include_next=False
    )
    h_state = normalizer.normalize(
        data[0, :, :, spec.water_channel], baseline.WATER_CHANNEL
    )
    h_state = torch.where(initial_valid, h_state, torch.zeros_like(h_state))
    static_cpu = torch.where(
        initial_valid.unsqueeze(0), static_cpu, torch.zeros_like(static_cpu)
    )
    h_state = h_state.unsqueeze(0).unsqueeze(0).float().to(device)
    static = static_cpu.unsqueeze(0).float().to(device)

    lead_metrics: List[Dict[str, Any]] = []
    tile_statistics: List[Dict[str, Any]] = []
    stabilization_steps = 0
    stabilization_pixels = 0
    model.eval()
    with torch.no_grad():
        for time_index in range(total_steps):
            valid = dem_common._dem_frame_mask(
                sequence, spec, time_index, include_next=True
            )
            rain = normalizer.normalize(
                data[time_index, :, :, spec.rain_channel], baseline.RAIN_CHANNEL
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
                delta, tile_info = tiled_model_delta(
                    model,
                    x=x,
                    static=static,
                    metadata=metadata,
                    window_size=_WINDOW_SIZE,
                    stride=_STRIDE,
                    tile_batch_size=_TILE_BATCH_SIZE,
                    blend_floor=_BLEND_FLOOR,
                )
            next_state = baseline.constrain_nonnegative_water_state(
                h_state.float() + delta.float(), normalizer
            )
            next_state, stabilization = project_gphyt_pakistan_rollout_state(
                model, next_state, normalizer
            )
            stabilization_steps += int(stabilization["clipped_pixels"] > 0)
            stabilization_pixels += int(stabilization["clipped_pixels"])
            if tuple(next_state.shape[-2:]) != tuple(data.shape[1:3]):
                raise AssertionError("stitched state does not match the physical grid")
            if not torch.isfinite(next_state).all():
                raise FloatingPointError(
                    f"{spec.name}/{Path(sequence.source_path).name}: "
                    f"non-finite prediction at lead {time_index + 1}"
                )

            prediction_phys = normalizer.denormalize(
                next_state, baseline.WATER_CHANNEL
            )
            target_phys = data[
                time_index + 1, :, :, spec.water_channel
            ].unsqueeze(0).unsqueeze(0).to(device)
            valid_device = valid.unsqueeze(0).unsqueeze(0).to(device)
            metrics = baseline.masked_step_metrics(
                prediction_phys, target_phys, valid_device
            )
            metrics.update(
                {
                    "lead_step": time_index + 1,
                    "lead_minutes": (time_index + 1) * spec.dt_minutes,
                    "valid_pixels": int(valid.sum().item()),
                    "tile_count": int(tile_info["tile_count"]),
                    "tile_inference_seconds": float(
                        tile_info["inference_seconds"]
                    ),
                    "rollout_stabilization_clipped_pixels": int(
                        stabilization["clipped_pixels"]
                    ),
                    "rollout_stabilization_clipped_fraction": float(
                        stabilization["clipped_fraction"]
                    ),
                }
            )
            lead_metrics.append(metrics)
            tile_info = dict(tile_info)
            tile_info["state_projection"] = stabilization
            tile_statistics.append(tile_info)

            # Only the stitched prediction is propagated.  Ground truth is
            # read above solely for metrics and is never assigned to h_state.
            next_valid = dem_common._dem_frame_mask(
                sequence,
                spec,
                min(time_index + 1, int(data.shape[0]) - 1),
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
        "tiling": {
            "window_size": _WINDOW_SIZE,
            "stride": _STRIDE,
            "tile_batch_size": _TILE_BATCH_SIZE,
            "strict_full_grid_state_feedback": True,
            "steps": tile_statistics,
        },
        "rollout_stabilization": {
            "enabled": bool(
                tile_statistics
                and tile_statistics[0]["state_projection"]["enabled"]
            ),
            "policy": (
                tile_statistics[0]["state_projection"]["policy"]
                if tile_statistics
                else "none"
            ),
            "water_cap_m": (
                tile_statistics[0]["state_projection"].get("water_cap_m")
                if tile_statistics
                else None
            ),
            "steps_with_projection": int(stabilization_steps),
            "total_clipped_pixels": int(stabilization_pixels),
            "non_finite_policy": "raise",
        },
    }
