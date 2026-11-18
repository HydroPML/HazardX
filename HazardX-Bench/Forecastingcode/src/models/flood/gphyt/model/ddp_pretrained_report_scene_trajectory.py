"""DDP-safe publication of audited pretrained-loading reports.

The existing model wrappers remain unchanged.  Each rank builds the selected
PT-FT model with a private report directory, then this module verifies that
all rank reports are identical before rank zero atomically publishes the
canonical report used by the experiment record.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn

from gphyt.model.model_registry_seven import PT_FT_NAMES


_REQUIRED_REPORT_KEYS = (
    "exact_loaded",
    "converted",
    "newly_initialized",
    "missing",
    "unexpected",
    "shape_mismatch",
    "loaded_parameter_count",
    "backbone_parameter_count",
    "backbone_loading_ratio",
    "checkpoint_sha256",
)

_COLLECTIVE_AUDIT_ENABLED = False


@contextmanager
def collective_pretrained_report_audit() -> Iterator[None]:
    """Enable collectives only while every training rank builds the model."""

    global _COLLECTIVE_AUDIT_ENABLED
    previous = _COLLECTIVE_AUDIT_ENABLED
    _COLLECTIVE_AUDIT_ENABLED = True
    try:
        yield
    finally:
        _COLLECTIVE_AUDIT_ENABLED = previous


def install_ddp_report_training_scope() -> None:
    """Wrap the installed trainer so post-train rank-zero eval never gathers."""

    import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline

    previous_train = baseline.train

    def train(*args, **kwargs):
        with collective_pretrained_report_audit():
            return previous_train(*args, **kwargs)

    baseline.train = train


def _rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def _read_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"rank-local pretrained report was not written: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError(f"pretrained report must be a JSON object: {path}")
    missing = [key for key in _REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise KeyError(f"pretrained report {path} lacks keys {missing}")
    ratio = float(report["backbone_loading_ratio"])
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"invalid backbone loading ratio {ratio} in {path}")
    if not str(report["checkpoint_sha256"]):
        raise ValueError(f"empty checkpoint SHA256 in {path}")
    return report


def _canonical_json(report: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(report), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _atomic_write_json(report: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(report),
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_model_with_ddp_safe_report(
    original_build_model: Callable[[Mapping[str, Any], torch.device], nn.Module],
    config: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build one model and safely consolidate its rank-local PT report."""

    values = dict(config)
    model_name = str(values.get("model_type", ""))
    if model_name not in PT_FT_NAMES:
        return original_build_model(values, device)

    canonical_root = Path(str(values["report_dir"])).expanduser().resolve()
    rank, world_size = _rank_and_world()
    rank_root = canonical_root / ".rank_reports" / f"rank_{rank}"
    rank_config = dict(values)
    rank_config["report_dir"] = str(rank_root)

    model = original_build_model(rank_config, device)
    rank_report_path = rank_root / model_name / "pretrained_loading_report.json"
    rank_report = _read_report(rank_report_path)
    serialized = _canonical_json(rank_report)

    collective_audit = world_size > 1 and _COLLECTIVE_AUDIT_ENABLED
    gathered: list[str | None] = [None for _ in range(world_size)]
    if collective_audit:
        dist.all_gather_object(gathered, serialized)
    else:
        gathered[rank] = serialized
    if collective_audit:
        reference = gathered[0]
        disagreements = [
            index for index, value in enumerate(gathered) if value != reference
        ]
        if disagreements:
            raise RuntimeError(
                "DDP pretrained loading reports disagree with rank 0: "
                f"ranks={disagreements}"
            )

    canonical_path = canonical_root / model_name / "pretrained_loading_report.json"
    if rank == 0:
        _atomic_write_json(rank_report, canonical_path)
    if collective_audit:
        dist.barrier()

    if rank == 0:
        published = _read_report(canonical_path)
        if _canonical_json(published) != serialized:
            raise RuntimeError("canonical pretrained report differs from rank-local report")
    print(
        "DDP_PRETRAINED_REPORT_AUDIT",
        json.dumps(
            {
                "model_type": model_name,
                "rank": rank,
                "world_size": world_size,
                "rank_report": str(rank_report_path),
                "canonical_report": str(canonical_path),
                "checkpoint_sha256": rank_report["checkpoint_sha256"],
                "backbone_loading_ratio": rank_report["backbone_loading_ratio"],
                "collective_audit_enabled": collective_audit,
                "all_ranks_identical": True if collective_audit else None,
                "atomic_rank0_publish": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return model
