"""NaN-safe FloodCastBench native-grid multi-domain EarthDynamics experiment.

This is a new, standalone experiment entry point.  It deliberately does not
modify ``experiments_EarthDynamics_mb0_1.py`` or any of the existing GPhyT
implementation files.

The model is the same GPhyT/EarthDynamics backbone used by
``experiments_EarthDynamics_mb0_1.py``.  The changes are confined to:

* a common adapter for the FloodCastBench ``[T, H, W, 3]`` tensors;
* dynamic native-grid patching with minimal bottom/right padding;
* homogeneous per-domain micro-batches and cross-domain gradient accumulation;
* one-step Markov training and strictly autoregressive validation/test;
* training-only standardization and FP32 MSE loss;
* FP32 validation/test plus the physical constraint ``h_next >= 0``;
* complete checkpoints and source-domain macro validation;
* a provenance guard that keeps Mozambique out of every training decision.

Channel convention observed in the supplied data:
    0 = water depth h, 1 = rainfall forcing, 2 = static terrain.

The GPhyT backbone is configured without its internal numerical integrator,
and its detokenizer directly emits the single-channel derivative ``dh``.  The experiment applies the
required Euler update explicitly: ``h_next = h_t + dh``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from gphyt.model import model_specs
from gphyt.model.tokenizer.tokenizer import Detokenizer
from gphyt.model.transformer.model import get_model as build_gphyt_model


DT_MINUTES = 5
CHANNEL_NAMES = ("water_depth", "rainfall", "terrain")
WATER_CHANNEL = 0
RAIN_CHANNEL = 1
TERRAIN_CHANNEL = 2
SOURCE_TRAIN_DOMAINS = ("australia", "pakistan", "uk_30m", "uk")
SOURCE_EVAL_DOMAINS = ("australia", "pakistan")
MOZAMBIQUE_DOMAINS = ("mozambique_576", "mozambique_1728")
CSI_THRESHOLDS = (0.01, 0.05, 0.1, 0.5)
# Training may use AMP, but every autoregressive validation/test rollout is FP32.
EVALUATION_AMP_ENABLED = False


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def natural_key(path: Path) -> Tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def list_sequence_files(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix == ".pt"),
        key=natural_key,
    )
    if not files:
        raise FileNotFoundError(f"No .pt sequence files found recursively in {root}")
    return files


def canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_data_provenance(specs: Mapping[str, "DomainSpec"]) -> None:
    """Fail closed if train/validation/test/Mozambique paths overlap."""
    train = {
        name: canonical_path(specs[name].path)
        for name in SOURCE_TRAIN_DOMAINS
        if name in specs
    }
    forbidden = {
        name: canonical_path(spec.path)
        for name, spec in specs.items()
        if spec.role != "train"
    }
    for train_name, train_path in train.items():
        for other_name, other_path in forbidden.items():
            if paths_overlap(train_path, other_path):
                raise RuntimeError(
                    "Data leakage guard rejected overlapping paths: "
                    f"train[{train_name}]={train_path} and "
                    f"{specs[other_name].role}[{other_name}]={other_path}"
                )

    moz_paths = {
        canonical_path(specs[name].path)
        for name in MOZAMBIQUE_DOMAINS
        if name in specs
    }
    for name, spec in specs.items():
        path = canonical_path(spec.path)
        if name not in MOZAMBIQUE_DOMAINS and any(
            paths_overlap(path, moz_path) for moz_path in moz_paths
        ):
            raise RuntimeError(
                f"Mozambique leakage guard rejected {name} path {path}"
            )


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    domain_id: int
    path: str
    role: str
    dt_minutes: int = DT_MINUTES
    dx: Optional[float] = None
    dy: Optional[float] = None
    water_channel: int = WATER_CHANNEL
    rain_channel: int = RAIN_CHANNEL
    terrain_channel: int = TERRAIN_CHANNEL
    time_axis: int = 0
    channel_axis: int = -1
    nodata_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dt_minutes != DT_MINUTES:
            raise ValueError(
                f"{self.name}: FloodCastBench cadence must remain 5 min, "
                f"got {self.dt_minutes}"
            )
        if self.role not in {"train", "valid", "test", "zero_shot"}:
            raise ValueError(f"{self.name}: invalid data role {self.role!r}")


@dataclass
class LoadedSequence:
    data: torch.Tensor
    mask: Optional[torch.Tensor]
    timestamps: Optional[Any]
    dx: Optional[float]
    dy: Optional[float]
    source_path: str
    raw_keys: Optional[List[str]]


@dataclass(frozen=True)
class SequenceProfile:
    path: str
    tensor_shape: Tuple[int, int, int, int]
    dtype: str
    time_length: int
    height: int
    width: int
    channels: int
    has_explicit_mask: bool
    has_timestamps: bool
    has_nan: bool
    has_inf: bool
    terrain_is_static: bool
    raw_keys: Optional[List[str]]


def _pick_mapping_value(
    obj: Mapping[str, Any], candidates: Sequence[str], required: bool
) -> Any:
    for key in candidates:
        if key in obj:
            return obj[key]
    if required:
        raise KeyError(
            f"Could not find any of {list(candidates)} in mapping keys "
            f"{sorted(str(k) for k in obj)}"
        )
    return None


def _coerce_mask(
    mask: Optional[Any], time_length: int, height: int, width: int
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask_tensor = torch.as_tensor(mask).bool().squeeze()
    if mask_tensor.shape == (height, width):
        return mask_tensor.unsqueeze(0).expand(time_length, -1, -1)
    if mask_tensor.shape == (time_length, height, width):
        return mask_tensor
    if mask_tensor.shape == (height, width, time_length):
        return mask_tensor.permute(2, 0, 1).contiguous()
    raise ValueError(
        "Unsupported explicit mask shape "
        f"{tuple(mask_tensor.shape)} for sequence {(time_length, height, width)}"
    )


def load_sequence_file(path: Path, spec: DomainSpec) -> LoadedSequence:
    """Load one trusted local PyTorch sequence and normalize it to [T,H,W,C]."""
    obj = torch.load(path, map_location="cpu")
    raw_keys: Optional[List[str]] = None
    mask = timestamps = None
    dx, dy = spec.dx, spec.dy
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, Mapping):
        raw_keys = sorted(str(key) for key in obj)
        tensor = _pick_mapping_value(
            obj, ("data", "tensor", "sequence", "states", "pde"), required=True
        )
        mask = _pick_mapping_value(
            obj, ("mask", "valid_mask", "nodata_mask"), required=False
        )
        timestamps = _pick_mapping_value(
            obj, ("timestamps", "times", "time"), required=False
        )
        if dx is None and "dx" in obj:
            dx = float(obj["dx"])
        if dy is None and "dy" in obj:
            dy = float(obj["dy"])
    else:
        raise TypeError(
            f"{path}: expected Tensor or mapping, got {type(obj).__name__}"
        )

    tensor = torch.as_tensor(tensor)
    if tensor.ndim != 4:
        raise ValueError(
            f"{path}: expected a four-dimensional sequence, got {tuple(tensor.shape)}"
        )
    time_axis = spec.time_axis % tensor.ndim
    channel_axis = spec.channel_axis % tensor.ndim
    if time_axis == channel_axis:
        raise ValueError(f"{path}: time_axis and channel_axis must be different")
    tensor = torch.movedim(tensor, (time_axis, channel_axis), (0, 3)).contiguous()
    if tensor.shape[-1] <= max(
        spec.water_channel, spec.rain_channel, spec.terrain_channel
    ):
        raise ValueError(
            f"{path}: only {tensor.shape[-1]} channels, but channel mapping is "
            f"h={spec.water_channel}, rain={spec.rain_channel}, "
            f"terrain={spec.terrain_channel}"
        )
    if not tensor.is_floating_point():
        tensor = tensor.float()
    mask_tensor = _coerce_mask(
        mask, int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])
    )
    return LoadedSequence(
        data=tensor,
        mask=mask_tensor,
        timestamps=timestamps,
        dx=dx,
        dy=dy,
        source_path=str(path),
        raw_keys=raw_keys,
    )


def profile_sequence(sequence: LoadedSequence, spec: DomainSpec) -> SequenceProfile:
    data = sequence.data
    terrain = data[:, :, :, spec.terrain_channel]
    terrain_is_static = bool(
        torch.equal(terrain[0], terrain[-1])
        and (
            data.shape[0] < 2
            or torch.equal(terrain[0], terrain[min(1, data.shape[0] - 1)])
        )
    )
    probe = data[[0, -1]]
    return SequenceProfile(
        path=sequence.source_path,
        tensor_shape=tuple(int(v) for v in data.shape),
        dtype=str(data.dtype),
        time_length=int(data.shape[0]),
        height=int(data.shape[1]),
        width=int(data.shape[2]),
        channels=int(data.shape[3]),
        has_explicit_mask=sequence.mask is not None,
        has_timestamps=sequence.timestamps is not None,
        has_nan=bool(torch.isnan(probe).any().item()),
        has_inf=bool(torch.isinf(probe).any().item()),
        terrain_is_static=terrain_is_static,
        raw_keys=sequence.raw_keys,
    )


def check_numeric_file_continuity(files: Sequence[Path], domain: str) -> None:
    numeric: List[int] = []
    for path in files:
        if path.stem.isdigit():
            numeric.append(int(path.stem))
        else:
            warnings.warn(
                f"{domain}: {path.name} has no parseable numeric index; "
                "lexical/natural ordering is used"
            )
            return
    if len(numeric) != len(set(numeric)):
        warnings.warn(f"{domain}: duplicate numeric file indices: {numeric}")
    if numeric:
        expected = list(range(min(numeric), max(numeric) + 1))
        if numeric != expected:
            warnings.warn(
                f"{domain}: file index gap/order mismatch; got {numeric}, "
                f"expected contiguous {expected}"
            )


class ChannelNormalizer:
    """One global training-only channel transform shared by every domain."""

    def __init__(
        self,
        mode: str = "none",
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std: Sequence[float] = (1.0, 1.0, 1.0),
        provenance: Optional[Mapping[str, Any]] = None,
    ):
        if mode not in {"none", "train_standard"}:
            raise ValueError(f"Unsupported normalization mode {mode!r}")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Normalizer requires exactly three channel statistics")
        if any((not math.isfinite(v)) or v <= 0 for v in std):
            raise ValueError(f"All normalization std values must be positive: {std}")
        self.mode = mode
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.provenance = dict(provenance or {})

    def normalize(self, tensor: torch.Tensor, channel: int) -> torch.Tensor:
        return (tensor - self.mean[channel]) / self.std[channel]

    def denormalize(self, tensor: torch.Tensor, channel: int) -> torch.Tensor:
        return tensor * self.std[channel] + self.mean[channel]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": list(self.mean),
            "std": list(self.std),
            "provenance": self.provenance,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ChannelNormalizer":
        return cls(
            mode=str(state["mode"]),
            mean=state["mean"],
            std=state["std"],
            provenance=state.get("provenance", {}),
        )


def constrain_nonnegative_water_state(
    normalized_state: torch.Tensor,
    normalizer: ChannelNormalizer,
) -> torch.Tensor:
    """Apply physical ``h >= 0`` in FP32 while retaining normalized storage.

    The forward value is exactly clamped.  A straight-through gradient avoids
    the zero-gradient dead zone that a hard training clamp would create for
    below-zero predictions; evaluation is numerically identical to clamp_min.
    """
    state_fp32 = normalized_state.float()
    normalized_zero = (
        0.0 - normalizer.mean[WATER_CHANNEL]
    ) / normalizer.std[WATER_CHANNEL]
    constrained = torch.clamp_min(state_fp32, normalized_zero)
    return state_fp32 + (constrained - state_fp32).detach()


def _valid_values(values: torch.Tensor, nodata: Optional[float]) -> torch.Tensor:
    valid = torch.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    return valid


class FloodDomainDataset(Dataset):
    """A lazy Markov-pair dataset with a one-sequence cache per worker."""

    def __init__(
        self,
        spec: DomainSpec,
        normalizer: ChannelNormalizer,
        noise_std: float = 0.0,
    ):
        self.spec = spec
        self.normalizer = normalizer
        self.noise_std = float(noise_std)
        self.files = list_sequence_files(canonical_path(spec.path))
        check_numeric_file_continuity(self.files, spec.name)

        representative = load_sequence_file(self.files[0], spec)
        self.profile = profile_sequence(representative, spec)
        if self.profile.channels != len(CHANNEL_NAMES):
            raise ValueError(
                f"{spec.name}: expected exactly {len(CHANNEL_NAMES)} channels "
                f"{CHANNEL_NAMES}, found {self.profile.channels}"
            )
        if self.profile.time_length < 2:
            raise ValueError(f"{spec.name}: a Markov sequence needs at least 2 states")
        if not self.profile.terrain_is_static:
            raise ValueError(
                f"{spec.name}: channel {spec.terrain_channel} is not static over time"
            )
        self.time_length = self.profile.time_length
        self.pairs_per_sequence = self.time_length - 1
        self._cached_index = 0
        self._cached_sequence: Optional[LoadedSequence] = representative

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_cached_index"] = None
        state["_cached_sequence"] = None
        return state

    def __len__(self) -> int:
        return len(self.files) * self.pairs_per_sequence

    def _load_sequence(self, sequence_index: int) -> LoadedSequence:
        if not 0 <= sequence_index < len(self.files):
            raise IndexError(sequence_index)
        if (
            self._cached_sequence is not None
            and self._cached_index == sequence_index
        ):
            return self._cached_sequence
        loaded = load_sequence_file(self.files[sequence_index], self.spec)
        shape = tuple(int(v) for v in loaded.data.shape)
        expected = self.profile.tensor_shape
        if shape != expected:
            raise ValueError(
                f"{self.spec.name}: all files in one domain must share layout; "
                f"{self.files[sequence_index]} has {shape}, representative has {expected}"
            )
        self._cached_index = sequence_index
        self._cached_sequence = loaded
        return loaded

    def get_sequence(self, sequence_index: int) -> LoadedSequence:
        return self._load_sequence(sequence_index)

    def _explicit_pair_mask(
        self, sequence: LoadedSequence, t: int
    ) -> torch.Tensor:
        height, width = sequence.data.shape[1:3]
        if sequence.mask is None:
            return torch.ones((height, width), dtype=torch.bool)
        return sequence.mask[t] & sequence.mask[t + 1]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_index = index // self.pairs_per_sequence
        t = index % self.pairs_per_sequence
        sequence = self._load_sequence(sequence_index)
        data = sequence.data
        h_t = data[t, :, :, self.spec.water_channel]
        rain_t = data[t, :, :, self.spec.rain_channel]
        terrain = data[0, :, :, self.spec.terrain_channel]
        h_next = data[t + 1, :, :, self.spec.water_channel]

        valid = self._explicit_pair_mask(sequence, t)
        for values in (h_t, rain_t, terrain, h_next):
            valid &= _valid_values(values, self.spec.nodata_value)
        if not bool(valid.any()):
            raise ValueError(
                f"{self.spec.name}/{self.files[sequence_index].name}@{t}: "
                "no valid pixels in Markov pair"
            )

        h_t_n = self.normalizer.normalize(h_t, WATER_CHANNEL)
        rain_n = self.normalizer.normalize(rain_t, RAIN_CHANNEL)
        terrain_n = self.normalizer.normalize(terrain, TERRAIN_CHANNEL)
        h_next_n = self.normalizer.normalize(h_next, WATER_CHANNEL)
        h_t_n = torch.where(valid, h_t_n, torch.zeros_like(h_t_n))
        rain_n = torch.where(valid, rain_n, torch.zeros_like(rain_n))
        terrain_n = torch.where(valid, terrain_n, torch.zeros_like(terrain_n))
        h_next_n = torch.where(valid, h_next_n, torch.zeros_like(h_next_n))
        x = torch.stack((h_t_n, rain_n), dim=0).float()
        static = terrain_n.unsqueeze(0).float()
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


def homogeneous_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty micro-batch")
    domains = {str(sample["domain"]) for sample in samples}
    shapes = {tuple(sample["x"].shape) for sample in samples}
    static_shapes = {tuple(sample["static"].shape) for sample in samples}
    target_shapes = {tuple(sample["y"].shape) for sample in samples}
    if len(domains) != 1:
        raise AssertionError(f"Mixed-domain micro-batch is forbidden: {domains}")
    if len(shapes) != 1 or len(static_shapes) != 1 or len(target_shapes) != 1:
        raise AssertionError(
            "Every micro-batch must have homogeneous H/W and variable layout: "
            f"x={shapes}, static={static_shapes}, y={target_shapes}"
        )
    batch = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in ("x", "y", "static", "mask")
    }
    for key in (
        "domain",
        "domain_id",
        "sample_id",
        "source_path",
        "time_index",
        "dt_minutes",
        "dx",
        "dy",
        "height",
        "width",
    ):
        batch[key] = [sample[key] for sample in samples]
    return batch


class SequenceLocalBatchSampler(Sampler[List[int]]):
    """Shuffle files/times while keeping every batch inside one sequence file.

    Keeping consecutive batches in the same large ``.pt`` file makes the
    dataset's one-file cache effective.  Across DDP ranks, complete batches are
    sharded after deterministic construction and padding, so every rank has
    exactly the same number of optimizer-update opportunities.
    """

    def __init__(
        self,
        dataset: FloodDomainDataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        if batch_size < 1:
            raise ValueError("micro-batch size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        file_indices = list(range(len(self.dataset.files)))
        rng.shuffle(file_indices)
        batches: List[List[int]] = []
        ntime = self.dataset.pairs_per_sequence
        for file_index in file_indices:
            times = list(range(ntime))
            rng.shuffle(times)
            base = file_index * ntime
            batches.extend(
                [
                    [base + t for t in times[start : start + self.batch_size]]
                    for start in range(0, ntime, self.batch_size)
                ]
            )
        if self.world_size > 1 and batches:
            missing = (-len(batches)) % self.world_size
            batches.extend(batches[:missing])
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._all_batches()
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        count_per_file = math.ceil(
            self.dataset.pairs_per_sequence / self.batch_size
        )
        total = len(self.dataset.files) * count_per_file
        return math.ceil(total / self.world_size)


class CyclingDomainIterator:
    def __init__(
        self,
        loader: DataLoader,
        sampler: SequenceLocalBatchSampler,
        base_epoch: int,
    ):
        self.loader = loader
        self.sampler = sampler
        self.base_epoch = int(base_epoch)
        self.cycle = 0
        self.iterator: Iterator[Dict[str, Any]]
        self._reset()

    def _reset(self) -> None:
        effective_epoch = self.base_epoch * 1_000_000 + self.cycle
        self.sampler.set_epoch(effective_epoch)
        self.iterator = iter(self.loader)

    def next(self) -> Dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.cycle += 1
            self._reset()
            return next(self.iterator)


def parse_name_values(
    value: str,
    valid_names: Iterable[str],
    cast: Any,
    default: Any,
) -> Dict[str, Any]:
    names = tuple(valid_names)
    result = {name: default for name in names}
    if not value.strip():
        return result
    for assignment in value.split(","):
        if "=" not in assignment:
            raise argparse.ArgumentTypeError(
                f"Expected name=value entry, got {assignment!r}"
            )
        name, raw = (part.strip() for part in assignment.split("=", 1))
        if name not in result:
            raise argparse.ArgumentTypeError(
                f"Unknown domain {name!r}; expected one of {names}"
            )
        result[name] = cast(raw)
    return result


class TaskSampler:
    def __init__(
        self,
        domains: Sequence[str],
        mode: str,
        sizes: Mapping[str, int],
        weights: Mapping[str, float],
        seed: int,
    ):
        self.domains = tuple(domains)
        self.mode = mode
        self.rng = random.Random(seed)
        if mode == "uniform":
            probabilities = [1.0 for _ in self.domains]
        elif mode == "proportional":
            probabilities = [float(sizes[name]) for name in self.domains]
        elif mode == "weighted":
            probabilities = [float(weights[name]) for name in self.domains]
        else:
            raise ValueError(f"Unsupported task sampling mode {mode!r}")
        if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
            raise ValueError(f"Invalid task sampling probabilities: {probabilities}")
        total = sum(probabilities)
        self.probabilities = tuple(value / total for value in probabilities)

    def sample(self, count: int) -> List[str]:
        return self.rng.choices(
            self.domains, weights=self.probabilities, k=int(count)
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "domains": self.domains,
            "mode": self.mode,
            "probabilities": self.probabilities,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["domains"]) != self.domains or state["mode"] != self.mode:
            raise ValueError("Task-sampler checkpoint is incompatible with this run")
        self.rng.setstate(state["rng_state"])


class NativeGridEarthDynamics(nn.Module):
    """The existing GPhyT core with dynamic RoPE grid and explicit dh output."""

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
    ):
        super().__init__()
        if not direct_derivative_output:
            raise ValueError(
                "NativeGridEarthDynamics requires direct_derivative_output=True"
            )
        if patch_height < 1 or patch_width < 1:
            raise ValueError("Patch dimensions must be positive")
        if attention_mode not in {"spatial", "full"}:
            raise ValueError(
                "One-step Markov mode supports dynamic spatial/full attention only"
            )
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.input_channels = len(CHANNEL_NAMES)
        patch_size = (1, self.patch_height, self.patch_width)

        # img_size only selects the T=1 spatial attention implementation.  RoPE,
        # tokenizer, attention and detokenizer all obtain H/W from each forward.
        # No learned fixed-grid absolute embedding is constructed.
        model_config = {
            "architecture": "gphyt",
            "img_size": (1, self.patch_height, self.patch_width),
            "transformer": {
                "input_channels": self.input_channels,
                "model_size": model_size,
                "att_mode": attention_mode,
                "dropout": dropout,
                "pos_enc_mode": "rope",
                "patch_size": list(patch_size),
                "stochastic_depth_rate": stochastic_depth_rate,
                "use_derivatives": use_derivatives,
                # Raw derivative field; h + dh is applied by the experiment.
                "integrator": None,
            },
            "tokenizer": {
                "tokenizer_mode": tokenizer_mode,
                "detokenizer_mode": detokenizer_mode,
                "tokenizer_overlap": tokenizer_overlap,
                "detokenizer_overlap": detokenizer_overlap,
            },
        }
        self.backbone = build_gphyt_model(model_config)
        gpt_spec = getattr(model_specs, model_size)()
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

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise AssertionError(
                f"x must be [B,2,H,W] (h,rain), got {tuple(x.shape)}"
            )
        if static is None or static.ndim != 4 or static.shape[1] != 1:
            raise AssertionError(
                "static terrain must be present with shape [B,1,H,W]"
            )
        if x.shape[0] != static.shape[0] or x.shape[-2:] != static.shape[-2:]:
            raise AssertionError(
                f"x/static mismatch: {tuple(x.shape)} vs {tuple(static.shape)}"
            )
        if not torch.isfinite(x).all() or not torch.isfinite(static).all():
            raise FloatingPointError("Model input contains NaN or Inf")
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if metadata is not None:
            meta_heights = metadata.get("height")
            meta_widths = metadata.get("width")
            if meta_heights is not None and any(
                int(value) != height for value in meta_heights
            ):
                raise AssertionError("metadata height does not match x")
            if meta_widths is not None and any(
                int(value) != width for value in meta_widths
            ):
                raise AssertionError("metadata width does not match x")

        fields = torch.cat((x, static), dim=1)
        pad_h, pad_w = self.padding_for(height, width)
        if pad_h or pad_w:
            fields = F.pad(fields, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        # Existing FloodGPhyT2d external layout is [B,H,W,T,C].
        fields = fields.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        derivative_fields = self.backbone(fields.permute(0, 3, 1, 2, 4))
        if derivative_fields.shape[-1] != 1:
            raise AssertionError(
                "EarthDynamics backbone must directly return one derivative "
                f"channel, got {tuple(derivative_fields.shape)}"
            )
        dh = (
            derivative_fields[:, 0, :height, :width, :]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        expected = (x.shape[0], 1, height, width)
        if tuple(dh.shape) != expected:
            raise AssertionError(
                f"Unpadded dh shape {tuple(dh.shape)} != expected {expected}"
            )
        if not torch.isfinite(dh).all():
            raise FloatingPointError("Model output dh contains NaN or Inf")
        return dh


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "patch_height": args.patch_height,
        "patch_width": args.patch_width,
        "model_size": args.model_size,
        "attention_mode": args.attention_mode,
        "tokenizer_mode": args.tokenizer_mode,
        "detokenizer_mode": args.detokenizer_mode,
        "tokenizer_overlap": args.tokenizer_overlap,
        "detokenizer_overlap": args.detokenizer_overlap,
        "use_derivatives": args.use_derivatives,
        "dropout": args.dropout,
        "stochastic_depth_rate": args.stochastic_depth_rate,
        "direct_derivative_output": True,
    }


def build_model(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = NativeGridEarthDynamics(**dict(config))
    if any(
        name.startswith("delta_projection")
        for name, _ in model.named_parameters()
    ):
        raise AssertionError("Direct-derivative model still has projection weights")
    return model.to(device)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in ("x", "y", "static", "mask"):
        moved[key] = batch[key].to(device, non_blocking=True)
    if moved["x"].shape[-2:] != moved["y"].shape[-2:]:
        raise AssertionError("Input and target H/W differ")
    if moved["mask"].shape != moved["y"].shape:
        raise AssertionError("Mask and target shapes differ")
    if not torch.isfinite(moved["x"]).all():
        raise FloatingPointError("Batch input contains NaN/Inf")
    if not torch.isfinite(moved["y"]).all():
        raise FloatingPointError("Batch target contains NaN/Inf")
    return moved


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample masked loss, followed by an equal sample average."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise AssertionError(
            f"loss shape mismatch pred={prediction.shape}, target={target.shape}, "
            f"mask={mask.shape}"
        )
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("At least one sample has zero valid target pixels")
    diff2 = torch.where(valid, (prediction - target).square(), 0.0)
    mse = diff2.flatten(1).sum(dim=1) / counts
    if kind == "mse":
        per_sample = mse
    elif kind == "rmse":
        per_sample = torch.sqrt(mse + eps)
    elif kind == "relative_l2":
        # Equivalent to the original LpLoss relative L2 norm on valid pixels,
        # with epsilon only to make completely dry targets well-defined.
        target2 = torch.where(valid, target.square(), 0.0).flatten(1).sum(dim=1)
        diff_norm = torch.sqrt(diff2.flatten(1).sum(dim=1) + eps)
        target_norm = torch.sqrt(target2 + eps)
        per_sample = diff_norm / target_norm
    else:
        raise ValueError(f"Unknown loss kind {kind!r}")
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {kind} loss")
    return loss


def _metric_value(value: torch.Tensor) -> float:
    result = float(value.detach().float().cpu().item())
    return result if math.isfinite(result) else float("nan")


def masked_step_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        return {
            key: float("nan")
            for key in (
                "rmse",
                "mae",
                "relative_l2",
                "nse",
                "corr",
                *(f"csi_{threshold:g}" for threshold in CSI_THRESHOLDS),
            )
        }
    pred = prediction[valid].double()
    truth = target[valid].double()
    error = pred - truth
    mse = error.square().mean()
    rmse = torch.sqrt(mse)
    mae = error.abs().mean()
    relative_l2 = torch.linalg.vector_norm(error) / (
        torch.linalg.vector_norm(truth) + 1e-12
    )
    truth_centered = truth - truth.mean()
    pred_centered = pred - pred.mean()
    nse_denominator = truth_centered.square().sum()
    nse_value = (
        1.0 - error.square().sum() / nse_denominator
        if nse_denominator > 0
        else torch.tensor(float("nan"), device=truth.device)
    )
    corr_denominator = torch.sqrt(
        truth_centered.square().sum() * pred_centered.square().sum()
    )
    corr_value = (
        (truth_centered * pred_centered).sum() / corr_denominator
        if corr_denominator > 0
        else torch.tensor(float("nan"), device=truth.device)
    )
    metrics = {
        "rmse": _metric_value(rmse),
        "mae": _metric_value(mae),
        "relative_l2": _metric_value(relative_l2),
        "nse": _metric_value(nse_value),
        "corr": _metric_value(corr_value),
    }
    for threshold in CSI_THRESHOLDS:
        truth_wet = truth > threshold
        pred_wet = pred > threshold
        hits = (truth_wet & pred_wet).sum()
        misses = (truth_wet & ~pred_wet).sum()
        false_alarms = (~truth_wet & pred_wet).sum()
        denominator = hits + misses + false_alarms
        value = (
            hits.double() / denominator.double()
            if denominator > 0
            else torch.tensor(float("nan"), device=truth.device)
        )
        metrics[f"csi_{threshold:g}"] = _metric_value(value)
    return metrics


def mean_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def autocast_context(enabled: bool) -> contextlib.AbstractContextManager:
    return torch.cuda.amp.autocast(enabled=enabled)


def _frame_mask(
    sequence: LoadedSequence,
    spec: DomainSpec,
    t: int,
    include_next: bool,
) -> torch.Tensor:
    data = sequence.data
    terrain = data[0, :, :, spec.terrain_channel]
    values = [
        data[t, :, :, spec.water_channel],
        data[t, :, :, spec.rain_channel],
        terrain,
    ]
    if include_next:
        values.append(data[t + 1, :, :, spec.water_channel])
    valid = torch.ones(data.shape[1:3], dtype=torch.bool)
    if sequence.mask is not None:
        valid &= sequence.mask[t]
        if include_next:
            valid &= sequence.mask[t + 1]
    for value in values:
        valid &= _valid_values(value, spec.nodata_value)
    return valid


def rollout_sequence(
    model: nn.Module,
    sequence: LoadedSequence,
    spec: DomainSpec,
    normalizer: ChannelNormalizer,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
) -> Dict[str, Any]:
    """Stream one strict autoregressive rollout, retaining only current h on GPU."""
    data = sequence.data
    if data.ndim != 4:
        raise AssertionError(f"Expected [T,H,W,C], got {tuple(data.shape)}")
    total_steps = int(data.shape[0]) - 1
    if max_steps > 0:
        total_steps = min(total_steps, int(max_steps))
    if total_steps < 1:
        raise ValueError(f"{sequence.source_path}: no rollout target steps")

    terrain_phys = data[0, :, :, spec.terrain_channel]
    initial_valid = _frame_mask(sequence, spec, 0, include_next=False)
    h_state = normalizer.normalize(
        data[0, :, :, spec.water_channel], WATER_CHANNEL
    )
    terrain = normalizer.normalize(terrain_phys, TERRAIN_CHANNEL)
    h_state = torch.where(initial_valid, h_state, torch.zeros_like(h_state))
    terrain = torch.where(initial_valid, terrain, torch.zeros_like(terrain))
    h_state = h_state.unsqueeze(0).unsqueeze(0).float().to(device)
    static = terrain.unsqueeze(0).unsqueeze(0).float().to(device)

    lead_metrics: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for t in range(total_steps):
            valid = _frame_mask(sequence, spec, t, include_next=True)
            rain = normalizer.normalize(
                data[t, :, :, spec.rain_channel], RAIN_CHANNEL
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
            with autocast_context(amp):
                dh = model(x=x, static=static, metadata=metadata)
            # Euler accumulation and the physical water-depth constraint are
            # deliberately FP32.  h is normalized, so physical zero must be
            # converted through the training-only normalizer before clamping.
            next_state = constrain_nonnegative_water_state(
                h_state.float() + dh.float(), normalizer
            )
            if not torch.isfinite(next_state).all():
                raise FloatingPointError(
                    f"{spec.name}/{Path(sequence.source_path).name}: "
                    f"non-finite prediction at lead {t + 1}"
                )
            prediction_phys = normalizer.denormalize(
                next_state, WATER_CHANNEL
            )
            target_phys = data[
                t + 1, :, :, spec.water_channel
            ].unsqueeze(0).unsqueeze(0).to(device)
            valid_device = valid.unsqueeze(0).unsqueeze(0).to(device)
            metrics = masked_step_metrics(
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
            # Detach immediately.  Invalid/nodata cells are never allowed to
            # feed arbitrary values into the next step.
            next_valid = _frame_mask(
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
        *(f"csi_{threshold:g}" for threshold in CSI_THRESHOLDS),
    ]
    overall = {
        key: mean_finite(step[key] for step in lead_metrics)
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


def evaluate_domain(
    model: nn.Module,
    dataset: FloodDomainDataset,
    normalizer: ChannelNormalizer,
    device: torch.device,
    amp: bool,
    max_steps: int = 0,
    max_sequences: int = 0,
) -> Dict[str, Any]:
    count = len(dataset.files)
    if max_sequences > 0:
        count = min(count, max_sequences)
    sequences: List[Dict[str, Any]] = []
    for index in range(count):
        sequence = dataset.get_sequence(index)
        sequences.append(
            rollout_sequence(
                model=model,
                sequence=sequence,
                spec=dataset.spec,
                normalizer=normalizer,
                device=device,
                amp=amp,
                max_steps=max_steps,
            )
        )

    metric_names = list(sequences[0]["overall"])
    overall = {
        key: mean_finite(result["overall"][key] for result in sequences)
        for key in metric_names
    }
    max_leads = max(len(result["lead_metrics"]) for result in sequences)
    lead_metrics: List[Dict[str, Any]] = []
    for lead_index in range(max_leads):
        available = [
            result["lead_metrics"][lead_index]
            for result in sequences
            if lead_index < len(result["lead_metrics"])
        ]
        row: Dict[str, Any] = {
            "lead_step": lead_index + 1,
            "lead_minutes": (lead_index + 1) * dataset.spec.dt_minutes,
            "sequence_count": len(available),
        }
        row.update(
            {
                key: mean_finite(item[key] for item in available)
                for key in metric_names
            }
        )
        lead_metrics.append(row)
    return {
        "domain": dataset.spec.name,
        "role": dataset.spec.role,
        "sequence_count": count,
        "dt_minutes": dataset.spec.dt_minutes,
        "spatial_shape": [dataset.profile.height, dataset.profile.width],
        "normalization": normalizer.state_dict(),
        "overall": overall,
        "lead_metrics": lead_metrics,
        "sequences": sequences,
    }


def macro_average(
    domain_results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, float]:
    if not domain_results:
        raise ValueError("No domain results for macro average")
    metric_names = list(next(iter(domain_results.values()))["overall"])
    return {
        key: mean_finite(result["overall"][key] for result in domain_results.values())
        for key in metric_names
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_evaluation(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    domain = str(result["domain"])
    with (output_dir / f"{domain}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2, ensure_ascii=False)
    lead_rows = result["lead_metrics"]
    if lead_rows:
        with (output_dir / f"{domain}_lead_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(lead_rows[0]))
            writer.writeheader()
            writer.writerows(json_safe(lead_rows))


def compute_training_normalizer(
    train_specs: Mapping[str, DomainSpec],
) -> ChannelNormalizer:
    """Compute exact finite-pixel statistics from training directories only."""
    sums = torch.zeros(3, dtype=torch.float64)
    sum_squares = torch.zeros(3, dtype=torch.float64)
    counts = torch.zeros(3, dtype=torch.float64)
    provenance_paths: Dict[str, List[str]] = {}
    for name in SOURCE_TRAIN_DOMAINS:
        spec = train_specs[name]
        files = list_sequence_files(canonical_path(spec.path))
        provenance_paths[name] = [str(path) for path in files]
        for path in files:
            sequence = load_sequence_file(path, spec)
            selected = sequence.data[
                :,
                :,
                :,
                [spec.water_channel, spec.rain_channel, spec.terrain_channel],
            ]
            for channel in range(3):
                values = selected[:, :, :, channel]
                valid = _valid_values(values, spec.nodata_value)
                if sequence.mask is not None:
                    valid &= sequence.mask
                finite = values[valid].double()
                if finite.numel() == 0:
                    raise ValueError(f"{name}/{path.name}: no valid channel {channel}")
                sums[channel] += finite.sum()
                sum_squares[channel] += finite.square().sum()
                counts[channel] += finite.numel()
            del sequence, selected
            gc.collect()
    means = sums / counts
    variances = torch.clamp(sum_squares / counts - means.square(), min=1e-12)
    stds = torch.sqrt(variances)
    return ChannelNormalizer(
        mode="train_standard",
        mean=means.tolist(),
        std=stds.tolist(),
        provenance={
            "allowed_roles": ["train"],
            "domains": list(SOURCE_TRAIN_DOMAINS),
            "files": provenance_paths,
            "algorithm": "exact_population_mean_std_over_finite_training_pixels",
        },
    )


def make_domain_specs(args: argparse.Namespace) -> Dict[str, DomainSpec]:
    return {
        "australia": DomainSpec(
            "australia", 0, args.australia_train, "train", dx=args.australia_dx,
            dy=args.australia_dy, nodata_value=args.nodata_value
        ),
        "pakistan": DomainSpec(
            "pakistan", 1, args.pakistan_train, "train", dx=args.pakistan_dx,
            dy=args.pakistan_dy, nodata_value=args.nodata_value
        ),
        "uk_30m": DomainSpec(
            "uk_30m", 2, args.uk30_train, "train", dx=args.uk30_dx,
            dy=args.uk30_dy, nodata_value=args.nodata_value
        ),
        "uk": DomainSpec(
            "uk", 3, args.uk_train, "train", dx=args.uk_dx,
            dy=args.uk_dy, nodata_value=args.nodata_value
        ),
        "australia_valid": DomainSpec(
            "australia", 0, args.australia_valid, "valid", dx=args.australia_dx,
            dy=args.australia_dy, nodata_value=args.nodata_value
        ),
        "pakistan_valid": DomainSpec(
            "pakistan", 1, args.pakistan_valid, "valid", dx=args.pakistan_dx,
            dy=args.pakistan_dy, nodata_value=args.nodata_value
        ),
        "australia_test": DomainSpec(
            "australia", 0, args.australia_test, "test", dx=args.australia_dx,
            dy=args.australia_dy, nodata_value=args.nodata_value
        ),
        "pakistan_test": DomainSpec(
            "pakistan", 1, args.pakistan_test, "test", dx=args.pakistan_dx,
            dy=args.pakistan_dy, nodata_value=args.nodata_value
        ),
        "mozambique_576": DomainSpec(
            "mozambique_576", 4, args.mozambique_576, "zero_shot",
            dx=args.mozambique_dx, dy=args.mozambique_dy,
            nodata_value=args.nodata_value
        ),
        "mozambique_1728": DomainSpec(
            "mozambique_1728", 4, args.mozambique_1728, "zero_shot",
            dx=args.mozambique_dx, dy=args.mozambique_dy,
            nodata_value=args.nodata_value
        ),
    }


def describe_dataset(
    dataset: FloodDomainDataset, batch_size: Optional[int]
) -> Dict[str, Any]:
    profile = dataset.profile
    description = {
        "domain": dataset.spec.name,
        "role": dataset.spec.role,
        "path": dataset.spec.path,
        "sequence_files": len(dataset.files),
        "markov_pairs": len(dataset),
        "time_length_per_file": dataset.time_length,
        "channels": profile.channels,
        "channel_names": list(CHANNEL_NAMES),
        "height": profile.height,
        "width": profile.width,
        "dx": dataset.spec.dx,
        "dy": dataset.spec.dy,
        "dt_minutes": dataset.spec.dt_minutes,
        "micro_batch_size": batch_size,
        "file_format": "PyTorch .pt",
        "raw_object": "Tensor" if profile.raw_keys is None else "mapping",
        "raw_keys": profile.raw_keys,
        "dimension_order": "[T,H,W,C]",
        "explicit_mask": profile.has_explicit_mask,
        "timestamps": profile.has_timestamps,
        "terrain_static": profile.terrain_is_static,
        "representative_file": profile.path,
        "representative_dtype": profile.dtype,
        "representative_nan": profile.has_nan,
        "representative_inf": profile.has_inf,
    }
    print(json.dumps(json_safe(description), ensure_ascii=False))
    return description


def build_train_loaders(
    specs: Mapping[str, DomainSpec],
    normalizer: ChannelNormalizer,
    batch_sizes: Mapping[str, int],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Tuple[
    Dict[str, FloodDomainDataset],
    Dict[str, DataLoader],
    Dict[str, SequenceLocalBatchSampler],
]:
    datasets: Dict[str, FloodDomainDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    samplers: Dict[str, SequenceLocalBatchSampler] = {}
    for offset, name in enumerate(SOURCE_TRAIN_DOMAINS):
        dataset = FloodDomainDataset(
            specs[name], normalizer=normalizer, noise_std=args.noise_std
        )
        sampler = SequenceLocalBatchSampler(
            dataset=dataset,
            batch_size=batch_sizes[name],
            seed=args.seed + offset * 10_007,
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=homogeneous_collate,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
        )
        datasets[name] = dataset
        loaders[name] = loader
        samplers[name] = sampler
    return datasets, loaders, samplers


def build_eval_datasets(
    specs: Mapping[str, DomainSpec],
    normalizer: ChannelNormalizer,
    keys: Sequence[str],
) -> Dict[str, FloodDomainDataset]:
    return {
        key: FloodDomainDataset(specs[key], normalizer=normalizer)
        for key in keys
    }


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # Checkpoints are loaded with ``map_location=device`` so every tensor in
    # the payload, including CPU RNG state, may be moved to CUDA. PyTorch's
    # RNG restoration APIs require CPU ByteTensors even when restoring CUDA
    # generator state.
    torch_state = state["torch"].detach().cpu()
    torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            cuda_state.detach().cpu()
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_update: int,
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    task_sampler: TaskSampler,
    normalizer: ChannelNormalizer,
    model_config: Mapping[str, Any],
    args: argparse.Namespace,
    specs: Mapping[str, DomainSpec],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "global_update": int(global_update),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "task_sampler_state": task_sampler.state_dict(),
        "rng_state": capture_rng_state(),
        "normalization": normalizer.state_dict(),
        "model_config": dict(model_config),
        "run_config": vars(args),
        "data_specs": {key: asdict(spec) for key, spec in specs.items()},
        "checkpoint_selection": {
            "metric": "source_domain_macro_rollout_rmse",
            "domains": ["australia", "pakistan"],
            "weights": [0.5, 0.5],
            "lower_is_better": True,
        },
    }


def load_checkpoint_file(path: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint_path = canonical_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {
        "model_state",
        "normalization",
        "model_config",
        "checkpoint_selection",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {sorted(missing)}")
    if checkpoint["model_config"].get("direct_derivative_output") is not True:
        raise RuntimeError(
            "Checkpoint predates the direct single-channel backbone output and "
            "is incompatible with this experiment"
        )
    selection_domains = checkpoint["checkpoint_selection"].get("domains", [])
    if any("mozambique" in str(name).lower() for name in selection_domains):
        raise RuntimeError("Checkpoint was selected using Mozambique and is rejected")
    provenance = checkpoint["normalization"].get("provenance", {})
    if any(
        "mozambique" in str(path).lower()
        for paths in provenance.get("files", {}).values()
        for path in paths
    ):
        raise RuntimeError(
            "Checkpoint normalization provenance includes Mozambique and is rejected"
        )
    return checkpoint


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    total_updates = args.epochs * args.updates_per_epoch
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_updates, 1)
        )
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_updates, gamma=args.scheduler_gamma
        )
    if args.scheduler == "none":
        return None
    raise ValueError(args.scheduler)


def run_source_validation(
    model: nn.Module,
    eval_datasets: Mapping[str, FloodDomainDataset],
    normalizer: ChannelNormalizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, Any]]:
    results: Dict[str, Any] = {}
    for key in ("australia_valid", "pakistan_valid"):
        result = evaluate_domain(
            model=unwrap_model(model),
            dataset=eval_datasets[key],
            normalizer=normalizer,
            device=device,
            amp=EVALUATION_AMP_ENABLED,
            max_steps=args.validation_max_steps,
            max_sequences=args.validation_max_sequences,
        )
        results[result["domain"]] = result
    macro = macro_average(results)
    score = 0.5 * results["australia"]["overall"]["rmse"] + 0.5 * results[
        "pakistan"
    ]["overall"]["rmse"]
    return float(score), {"domains": results, "macro": macro}


def train(
    args: argparse.Namespace,
    specs: Mapping[str, DomainSpec],
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Path:
    output_dir = canonical_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_checkpoint = load_checkpoint_file(args.resume, device)
        normalizer = ChannelNormalizer.from_state_dict(
            resume_checkpoint["normalization"]
        )
        if normalizer.mode != "train_standard":
            raise ValueError(
                "NaN-safe baseline can only resume a train_standard checkpoint"
            )
        if resume_checkpoint.get("run_config", {}).get("loss") != "mse":
            raise ValueError("NaN-safe baseline can only resume an MSE checkpoint")
        model_config = dict(resume_checkpoint["model_config"])
    else:
        normalizer = (
            compute_training_normalizer(
                {name: specs[name] for name in SOURCE_TRAIN_DOMAINS}
            )
            if args.normalization == "train_standard"
            else ChannelNormalizer(
                mode="none",
                provenance={
                    "source": "fixed_identity_physical_scale",
                    "allowed_roles": ["train"],
                    "domains": list(SOURCE_TRAIN_DOMAINS),
                },
            )
        )
        model_config = model_config_from_args(args)

    batch_sizes = parse_name_values(
        args.domain_batch_sizes,
        SOURCE_TRAIN_DOMAINS,
        int,
        args.micro_batch_size,
    )
    if any(value < 1 for value in batch_sizes.values()):
        raise ValueError(f"Invalid domain batch sizes: {batch_sizes}")
    datasets, loaders, samplers = build_train_loaders(
        specs, normalizer, batch_sizes, args, rank, world_size
    )
    domain_loader_steps = {
        name: len(loaders[name]) for name in SOURCE_TRAIN_DOMAINS
    }
    if any(steps < 1 for steps in domain_loader_steps.values()):
        raise ValueError(f"At least one training loader is empty: {domain_loader_steps}")
    args.updates_per_epoch = max(domain_loader_steps.values())
    args.auto_updates_per_epoch = True
    args.domain_loader_steps_per_rank = dict(domain_loader_steps)
    if is_main_process(rank):
        print(
            json.dumps(
                {
                    "auto_updates_per_epoch": args.updates_per_epoch,
                    "rule": "max(per-rank domain DataLoader lengths)",
                    "domain_loader_steps_per_rank": domain_loader_steps,
                    "world_size": world_size,
                    "domain_batch_sizes": dict(batch_sizes),
                },
                ensure_ascii=False,
            )
        )
    eval_datasets = build_eval_datasets(
        specs, normalizer, ("australia_valid", "pakistan_valid")
    )
    if is_main_process(rank):
        for name in SOURCE_TRAIN_DOMAINS:
            describe_dataset(datasets[name], batch_sizes[name])
        for key in ("australia_valid", "pakistan_valid"):
            describe_dataset(eval_datasets[key], batch_size=1)
        print("normalization", json.dumps(json_safe(normalizer.state_dict())))

    layouts = {
        (
            dataset.profile.channels,
            dataset.spec.water_channel,
            dataset.spec.rain_channel,
            dataset.spec.terrain_channel,
        )
        for dataset in datasets.values()
    }
    if len(layouts) != 1:
        raise ValueError(f"Inconsistent source-domain channel layouts: {layouts}")

    model = build_model(model_config, device)
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=args.amp_initial_scale
    )
    task_weights = parse_name_values(
        args.task_weights, SOURCE_TRAIN_DOMAINS, float, 1.0
    )
    task_sampler = TaskSampler(
        domains=SOURCE_TRAIN_DOMAINS,
        mode=args.task_sampling,
        sizes={name: len(dataset) for name, dataset in datasets.items()},
        weights=task_weights,
        seed=args.seed + 424_243,
    )

    start_epoch = 0
    global_update = 0
    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0
    if resume_checkpoint is not None:
        if dict(resume_checkpoint["model_config"]) != model_config:
            raise ValueError("Resume model configuration mismatch")
        unwrap_model(model).load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state", {}))
        task_sampler.load_state_dict(resume_checkpoint["task_sampler_state"])
        restore_rng_state(resume_checkpoint["rng_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_update = int(resume_checkpoint["global_update"])
        best_score = float(resume_checkpoint["best_score"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        bad_epochs = int(resume_checkpoint["bad_epochs"])

    best_path = output_dir / "best_validation.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        iterators = {
            name: CyclingDomainIterator(loaders[name], samplers[name], epoch)
            for name in SOURCE_TRAIN_DOMAINS
        }
        loss_sums = {name: 0.0 for name in SOURCE_TRAIN_DOMAINS}
        loss_counts = {name: 0 for name in SOURCE_TRAIN_DOMAINS}
        sampled_counts = {name: 0 for name in SOURCE_TRAIN_DOMAINS}
        epoch_start = time.perf_counter()

        for update_in_epoch in range(args.updates_per_epoch):
            selected_domains = task_sampler.sample(args.accum_steps)
            optimizer.zero_grad(set_to_none=True)
            for micro_step, domain in enumerate(selected_domains):
                batch = move_batch(iterators[domain].next(), device)
                sampled_counts[domain] += 1
                should_sync = micro_step == args.accum_steps - 1
                sync_context = (
                    contextlib.nullcontext()
                    if should_sync or not isinstance(model, DDP)
                    else model.no_sync()
                )
                with sync_context:
                    with autocast_context(amp_enabled):
                        dh = model(
                            x=batch["x"],
                            static=batch["static"],
                            metadata=batch,
                        )
                    # Keep the residual Euler update, physical constraint and
                    # complete MSE reduction outside autocast.
                    prediction = constrain_nonnegative_water_state(
                        batch["x"][:, :1].float() + dh.float(), normalizer
                    )
                    with autocast_context(False):
                        raw_loss = masked_loss(
                            prediction,
                            batch["y"].float(),
                            batch["mask"],
                            kind="mse",
                        )
                        scaled_loss = raw_loss / args.accum_steps
                    scaler.scale(scaled_loss).backward()
                loss_sums[domain] += float(raw_loss.detach().cpu())
                loss_counts[domain] += 1

            gradient_finite = True
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                gradient_finite = bool(torch.isfinite(gradient_norm))
                if not gradient_finite and not amp_enabled:
                    raise FloatingPointError("Gradient norm is NaN/Inf")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = (
                gradient_finite
                and ((not amp_enabled) or scaler.get_scale() >= old_scale)
            )
            if step_succeeded:
                global_update += 1
                if scheduler is not None:
                    scheduler.step()
            if (
                is_main_process(rank)
                and args.log_every_updates > 0
                and (update_in_epoch + 1) % args.log_every_updates == 0
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "update_in_epoch": update_in_epoch + 1,
                            "global_optimizer_updates": global_update,
                            "selected_domains": selected_domains,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                )

        epoch_seconds = time.perf_counter() - epoch_start
        reduced_losses = {
            name: reduce_sum(loss_sums[name], device)
            / max(reduce_sum(loss_counts[name], device), 1.0)
            for name in SOURCE_TRAIN_DOMAINS
        }
        reduced_sampled = {
            name: int(reduce_sum(sampled_counts[name], device))
            for name in SOURCE_TRAIN_DOMAINS
        }

        validation_score = float("nan")
        validation_result: Optional[Dict[str, Any]] = None
        should_validate = (
            (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1
        )
        if should_validate and is_main_process(rank):
            validation_score, validation_result = run_source_validation(
                model, eval_datasets, normalizer, device, args
            )
            improved = validation_score < best_score
            if improved:
                best_score = validation_score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            summary = {
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "train_loss_by_domain": reduced_losses,
                "sampled_micro_batches_by_domain_all_ranks": reduced_sampled,
                "validation_score_macro_rmse": validation_score,
                "validation_macro": validation_result["macro"],
                "best_score": best_score,
                "best_epoch": best_epoch,
                "global_update": global_update,
            }
            print(json.dumps(json_safe(summary), ensure_ascii=False))
        else:
            improved = False

        state = {
            "validation_score": validation_score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "improved": improved,
        }
        state = broadcast_object(state if is_main_process(rank) else None)
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        improved = bool(state["improved"])

        if is_main_process(rank):
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_update=global_update,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                task_sampler=task_sampler,
                normalizer=normalizer,
                model_config=model_config,
                args=args,
                specs=specs,
            )
            atomic_torch_save(payload, last_path)
            if improved:
                atomic_torch_save(payload, best_path)
                if validation_result is not None:
                    validation_dir = output_dir / "best_validation_metrics"
                    for result in validation_result["domains"].values():
                        write_evaluation(result, validation_dir)
                    with (validation_dir / "source_macro.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            json_safe(validation_result["macro"]),
                            f,
                            indent=2,
                        )
        distributed_barrier()
        stop = (
            args.early_stopping > 0
            and should_validate
            and bad_epochs >= args.early_stopping
        )
        stop = bool(broadcast_object(stop if is_main_process(rank) else None))
        if stop:
            break

    if not best_path.exists() and is_main_process(rank):
        # This can only happen when validation was explicitly skipped through a
        # malformed run setup; keep behavior explicit and recoverable.
        if last_path.exists():
            warnings.warn("No best checkpoint was selected; using last checkpoint")
            checkpoint = torch.load(last_path, map_location="cpu")
            atomic_torch_save(checkpoint, best_path)
        else:
            raise RuntimeError("Training finished without writing a checkpoint")
    distributed_barrier()
    return best_path


def load_eval_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, ChannelNormalizer, Dict[str, Any]]:
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model = build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = ChannelNormalizer.from_state_dict(checkpoint["normalization"])
    if normalizer.mode != "train_standard":
        raise ValueError(
            "NaN-safe evaluation requires a train_standard checkpoint"
        )
    if checkpoint.get("run_config", {}).get("loss") != "mse":
        raise ValueError("NaN-safe evaluation requires an MSE checkpoint")
    return model, normalizer, checkpoint


def run_source_test(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
    specs: Mapping[str, DomainSpec],
    device: torch.device,
) -> Dict[str, Any]:
    model, normalizer, _ = load_eval_model(checkpoint_path, device)
    datasets = build_eval_datasets(
        specs, normalizer, ("australia_test", "pakistan_test")
    )
    results: Dict[str, Any] = {}
    output_dir = canonical_path(args.output_dir) / "source_test"
    for key in ("australia_test", "pakistan_test"):
        describe_dataset(datasets[key], batch_size=1)
        result = evaluate_domain(
            model,
            datasets[key],
            normalizer,
            device,
            amp=EVALUATION_AMP_ENABLED,
            max_steps=args.eval_max_steps,
            max_sequences=args.eval_max_sequences,
        )
        results[result["domain"]] = result
        write_evaluation(result, output_dir)
    macro = macro_average(results)
    with (output_dir / "source_macro.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(macro), f, indent=2)
    print(json.dumps(json_safe({"source_test_macro": macro}), ensure_ascii=False))
    return {"domains": results, "macro": macro}


def check_timestamp_continuity(dataset: FloodDomainDataset) -> None:
    """Report what can and cannot be established from sequence metadata."""
    check_numeric_file_continuity(dataset.files, dataset.spec.name)
    if not dataset.profile.has_timestamps:
        warnings.warn(
            f"{dataset.spec.name}: files contain no timestamps.  Each .pt is "
            f"evaluated as an independent contiguous {dataset.time_length}-state "
            "trajectory in numeric filename order; absolute 5-minute continuity "
            "cannot be independently verified from file contents."
        )


def run_mozambique_test(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
    specs: Mapping[str, DomainSpec],
    device: torch.device,
) -> Dict[str, Any]:
    model, normalizer, checkpoint = load_eval_model(checkpoint_path, device)
    selection_domains = set(checkpoint["checkpoint_selection"]["domains"])
    if selection_domains != {"australia", "pakistan"}:
        raise RuntimeError(
            "Strict zero-shot evaluation requires an Australia/Pakistan "
            f"validation-selected checkpoint, got {selection_domains}"
        )
    selected_keys = {
        "576": ("mozambique_576",),
        "1728": ("mozambique_1728",),
        "both": ("mozambique_576", "mozambique_1728"),
    }[args.mozambique_split]
    datasets = build_eval_datasets(specs, normalizer, selected_keys)
    results: Dict[str, Any] = {}
    output_dir = canonical_path(args.output_dir) / "mozambique_zero_shot"
    for key in selected_keys:
        dataset = datasets[key]
        describe_dataset(dataset, batch_size=1)
        check_timestamp_continuity(dataset)
        result = evaluate_domain(
            model,
            dataset,
            normalizer,
            device,
            amp=EVALUATION_AMP_ENABLED,
            max_steps=args.eval_max_steps,
            max_sequences=args.eval_max_sequences,
        )
        results[key] = result
        write_evaluation(result, output_dir)
        print(
            json.dumps(
                json_safe(
                    {
                        "zero_shot_domain": key,
                        "overall": result["overall"],
                        "sequence_count": result["sequence_count"],
                        "steps": len(result["lead_metrics"]),
                    }
                ),
                ensure_ascii=False,
            )
        )
    return results


def inspect_all_domains(
    specs: Mapping[str, DomainSpec],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    normalizer = ChannelNormalizer()
    batch_sizes = parse_name_values(
        args.domain_batch_sizes,
        SOURCE_TRAIN_DOMAINS,
        int,
        args.micro_batch_size,
    )
    descriptions: Dict[str, Any] = {}
    for key, spec in specs.items():
        dataset = FloodDomainDataset(spec, normalizer)
        batch_size = batch_sizes.get(key, 1)
        descriptions[key] = describe_dataset(dataset, batch_size)
        check_numeric_file_continuity(dataset.files, key)
        del dataset
        gc.collect()
    source_layouts = {
        (
            descriptions[name]["channels"],
            tuple(descriptions[name]["channel_names"]),
        )
        for name in SOURCE_TRAIN_DOMAINS
    }
    if len(source_layouts) != 1:
        raise ValueError(f"Source channel definition mismatch: {source_layouts}")
    return descriptions


def gradient_l1(model: nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().double().abs().sum().cpu().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def run_smoke_tests(
    args: argparse.Namespace,
    specs: Mapping[str, DomainSpec],
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Run the requested real-data, no-resize acceptance smoke suite."""
    if world_size != 1:
        raise RuntimeError("Smoke suite must be launched as a single process")
    normalizer = compute_training_normalizer(
        {name: specs[name] for name in SOURCE_TRAIN_DOMAINS}
    )
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.num_workers = 0
    smoke_args.noise_std = 0.0
    batch_sizes = {name: 1 for name in SOURCE_TRAIN_DOMAINS}
    datasets, loaders, samplers = build_train_loaders(
        specs, normalizer, batch_sizes, smoke_args, rank, world_size
    )
    batches: Dict[str, Dict[str, Any]] = {}
    observed_shapes: Dict[str, Tuple[int, int]] = {}
    for name in SOURCE_TRAIN_DOMAINS:
        batch = next(iter(loaders[name]))
        batches[name] = batch
        observed_shapes[name] = tuple(int(v) for v in batch["x"].shape[-2:])
        describe_dataset(datasets[name], batch_size=1)
    if len(set(observed_shapes.values())) < 2:
        raise AssertionError(
            f"Expected native domains with different H/W, got {observed_shapes}"
        )
    print("SMOKE native_shapes", observed_shapes)

    model_config = model_config_from_args(args)
    model = build_model(model_config, device)
    amp_enabled = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outputs: Dict[str, Tuple[int, ...]] = {}
    model.eval()
    with torch.no_grad():
        for name in SOURCE_TRAIN_DOMAINS:
            batch = move_batch(batches[name], device)
            with autocast_context(EVALUATION_AMP_ENABLED):
                dh = model(x=batch["x"], static=batch["static"], metadata=batch)
            prediction = constrain_nonnegative_water_state(
                batch["x"][:, :1].float() + dh.float(), normalizer
            )
            if prediction.shape != batch["y"].shape:
                raise AssertionError(
                    f"{name}: unpadded prediction {prediction.shape} != "
                    f"target {batch['y'].shape}"
                )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"{name}: non-finite forward output")
            outputs[name] = tuple(int(v) for v in prediction.shape)
            del batch, dh, prediction
    print("SMOKE forward_unpadded_shapes", outputs)

    # One real optimizer update from four different native grids.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=args.amp_initial_scale
    )
    optimizer.zero_grad(set_to_none=True)
    gradient_after_each: List[float] = []
    optimizer_steps = 0
    for name in SOURCE_TRAIN_DOMAINS:
        batch = move_batch(batches[name], device)
        with autocast_context(amp_enabled):
            dh = model(x=batch["x"], static=batch["static"], metadata=batch)
        prediction = constrain_nonnegative_water_state(
            batch["x"][:, :1].float() + dh.float(), normalizer
        )
        with autocast_context(False):
            loss = masked_loss(
                prediction, batch["y"].float(), batch["mask"], kind="mse"
            ) / len(SOURCE_TRAIN_DOMAINS)
        scaler.scale(loss).backward()
        current_gradient = gradient_l1(model)
        if current_gradient <= 0:
            raise AssertionError(f"{name}: gradients were not accumulated")
        gradient_after_each.append(current_gradient)
        del batch, dh, prediction, loss
    scaler.unscale_(optimizer)
    unscaled_gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
    )
    if not torch.isfinite(unscaled_gradient_norm):
        raise FloatingPointError("Smoke accumulated gradient norm is NaN/Inf")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if amp_enabled and scaler.get_scale() < old_scale:
        raise FloatingPointError("GradScaler skipped the smoke optimizer step")
    optimizer_steps += 1
    optimizer.zero_grad(set_to_none=True)
    if optimizer_steps != 1 or len(gradient_after_each) != 4:
        raise AssertionError("Gradient accumulation did not produce exactly one step")
    print(
        "SMOKE accumulation",
        {
            "micro_batches": 4,
            "domains": list(SOURCE_TRAIN_DOMAINS),
            "optimizer_steps": optimizer_steps,
            "gradient_l1_after_each": gradient_after_each,
            "unscaled_gradient_norm": float(unscaled_gradient_norm.cpu()),
        },
    )

    # Short strict rollouts.  No time slicing is written and no spatial resize
    # is applied; max_steps only stops inference early.
    eval_keys = (
        "pakistan_valid",
        "australia_test",
        "mozambique_576",
    )
    eval_datasets = build_eval_datasets(specs, normalizer, eval_keys)
    model.eval()
    for key in eval_keys:
        result = evaluate_domain(
            model,
            eval_datasets[key],
            normalizer,
            device,
            amp=EVALUATION_AMP_ENABLED,
            max_steps=args.smoke_rollout_steps,
            max_sequences=1,
        )
        if result["sequence_count"] != 1:
            raise AssertionError(f"{key}: smoke must evaluate one real sequence")
        if len(result["lead_metrics"]) != args.smoke_rollout_steps:
            raise AssertionError(
                f"{key}: expected {args.smoke_rollout_steps} rollout steps"
            )
        print(
            "SMOKE rollout",
            key,
            json.dumps(json_safe(result["overall"])),
        )

    validate_data_provenance(specs)
    training_paths = {
        canonical_path(specs[name].path) for name in SOURCE_TRAIN_DOMAINS
    }
    mozambique_paths = {
        canonical_path(specs[name].path) for name in MOZAMBIQUE_DOMAINS
    }
    if training_paths & mozambique_paths:
        raise AssertionError("Training and Mozambique path sets intersect")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    print(
        "SMOKE PASS",
        {
            "device": str(device),
            "cuda_peak_memory_bytes": peak_memory,
            "no_resize": True,
            "mozambique_training_intersection": False,
            "finite_loss_and_output": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FloodCastBench multi-country native-grid EarthDynamics training, "
            "source testing and strict Mozambique zero-shot rollout"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("inspect", "smoke", "train", "source-test", "mozambique", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/HDD2/qingsong/qinqsong/EarthDynamics/results/"
            "FloodCastBench_multidomain_native_nan_safe"
        ),
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")

    parser.add_argument(
        "--australia-train",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Australia/288/train",
    )
    parser.add_argument(
        "--pakistan-train",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Pakistan/288/train",
    )
    parser.add_argument(
        "--uk30-train",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/UK_30/288/train",
    )
    parser.add_argument(
        "--uk-train",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/UK/288/train",
    )
    parser.add_argument(
        "--australia-valid",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Australia/288/valid",
    )
    parser.add_argument(
        "--pakistan-valid",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Pakistan/288/valid",
    )
    parser.add_argument(
        "--australia-test",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Australia/288/test",
    )
    parser.add_argument(
        "--pakistan-test",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Pakistan/288/test",
    )
    parser.add_argument(
        "--mozambique-576",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Test/Mozambique/576",
    )
    parser.add_argument(
        "--mozambique-1728",
        default="/mnt/HDD2/qingsong/qinqsong/Datasets/FM/Test/Mozambique/1728",
    )
    parser.add_argument(
        "--mozambique-split", choices=("576", "1728", "both"), default="both"
    )

    parser.add_argument("--australia-dx", type=float, default=None)
    parser.add_argument("--australia-dy", type=float, default=None)
    parser.add_argument("--pakistan-dx", type=float, default=None)
    parser.add_argument("--pakistan-dy", type=float, default=None)
    parser.add_argument("--uk30-dx", type=float, default=None)
    parser.add_argument("--uk30-dy", type=float, default=None)
    parser.add_argument("--uk-dx", type=float, default=None)
    parser.add_argument("--uk-dy", type=float, default=None)
    parser.add_argument("--mozambique-dx", type=float, default=None)
    parser.add_argument("--mozambique-dy", type=float, default=None)
    parser.add_argument("--nodata-value", type=float, default=None)

    parser.add_argument("--patch-height", type=int, default=16)
    parser.add_argument("--patch-width", type=int, default=16)
    parser.add_argument(
        "--model-size",
        choices=("GPT_S", "GPT_M", "GPT_L", "GPT_XL"),
        default="GPT_S",
    )
    parser.add_argument(
        "--attention-mode", choices=("spatial", "full"), default="spatial"
    )
    parser.add_argument(
        "--tokenizer-mode", choices=("linear", "conv_net"), default="linear"
    )
    parser.add_argument(
        "--detokenizer-mode", choices=("linear", "conv_net"), default="linear"
    )
    parser.add_argument("--tokenizer-overlap", type=int, default=0)
    parser.add_argument("--detokenizer-overlap", type=int, default=0)
    parser.add_argument("--use-derivatives", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument(
        "--domain-batch-sizes",
        default="australia=1,pakistan=1,uk_30m=1,uk=1",
    )
    parser.add_argument(
        "--task-sampling",
        choices=("uniform", "proportional", "weighted"),
        default="uniform",
    )
    parser.add_argument(
        "--task-weights",
        default="australia=1,pakistan=1,uk_30m=1,uk=1",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Training AMP only; validation and test rollout are always FP32",
    )
    parser.add_argument("--amp-initial-scale", type=float, default=128.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument(
        "--loss", choices=("relative_l2", "mse", "rmse"), default="mse"
    )
    parser.add_argument(
        "--normalization",
        choices=("none", "train_standard"),
        default="train_standard",
    )
    parser.add_argument(
        "--scheduler", choices=("cosine", "step", "none"), default="cosine"
    )
    parser.add_argument("--scheduler-step-updates", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--validation-max-steps", type=int, default=0)
    parser.add_argument("--validation-max-sequences", type=int, default=0)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--log-every-updates", type=int, default=10)

    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=0,
        help="0 evaluates each sequence to its last state; positive is smoke/debug only",
    )
    parser.add_argument(
        "--eval-max-sequences",
        type=int,
        default=0,
        help="0 evaluates every sequence; positive is smoke/debug only",
    )
    parser.add_argument("--smoke-rollout-steps", type=int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "accum_steps": args.accum_steps,
        "validate_every": args.validate_every,
        "smoke_rollout_steps": args.smoke_rollout_steps,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip cannot be negative")
    if args.amp_initial_scale <= 0:
        raise ValueError("--amp-initial-scale must be positive")
    if args.normalization != "train_standard":
        raise ValueError("NaN-safe baseline requires --normalization train_standard")
    if args.loss != "mse":
        raise ValueError("NaN-safe baseline requires --loss mse")
    if args.mode in {"source-test", "mozambique"} and not args.checkpoint:
        raise ValueError(f"--mode {args.mode} requires --checkpoint")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    distributed, rank, world_size, _, device = setup_distributed()
    try:
        seed_everything(args.seed, deterministic=args.deterministic)
        specs = make_domain_specs(args)
        validate_data_provenance(specs)
        if args.mode == "inspect":
            if is_main_process(rank):
                inspect_all_domains(specs, args)
            return
        if args.mode == "smoke":
            if is_main_process(rank):
                run_smoke_tests(args, specs, device, rank, world_size)
            return

        checkpoint_path: Optional[Path] = None
        if args.mode in {"train", "all"}:
            checkpoint_path = train(
                args, specs, device, distributed, rank, world_size
            )
        elif args.checkpoint:
            checkpoint_path = canonical_path(args.checkpoint)

        distributed_barrier()
        if is_main_process(rank):
            if args.mode in {"source-test", "all"}:
                assert checkpoint_path is not None
                run_source_test(checkpoint_path, args, specs, device)
            if args.mode in {"mozambique", "all"}:
                assert checkpoint_path is not None
                run_mozambique_test(checkpoint_path, args, specs, device)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
