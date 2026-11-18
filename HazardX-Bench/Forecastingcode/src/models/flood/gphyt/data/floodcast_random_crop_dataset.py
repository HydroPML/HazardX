"""Deterministic random-crop view of the existing FloodCastBench Dataset."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
from gphyt.data.floodcast_window_utils import stable_int_seed


@dataclass(frozen=True)
class TrainingWindowConfig:
    window_size: int = 128
    window_seed: int = 42
    batches_per_rank: Optional[int] = None


_CONFIG = TrainingWindowConfig()


def configure_training_windows(
    *,
    window_size: int,
    window_seed: int,
    batches_per_rank: Optional[int],
) -> TrainingWindowConfig:
    global _CONFIG
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if batches_per_rank is not None and batches_per_rank < 1:
        raise ValueError("batches_per_rank must be positive when specified")
    _CONFIG = TrainingWindowConfig(
        window_size=int(window_size),
        window_seed=int(window_seed),
        batches_per_rank=(
            None if batches_per_rank is None else int(batches_per_rank)
        ),
    )
    return _CONFIG


def _decode_index(index: Any) -> Tuple[int, int, int]:
    if isinstance(index, tuple):
        if len(index) != 3:
            raise ValueError(f"window index must have 3 entries, got {index!r}")
        base_index, epoch, global_sample_index = index
        return int(base_index), int(epoch), int(global_sample_index)
    value = int(index)
    return value, 0, value


def _crop_and_pad(
    tensor: torch.Tensor,
    *,
    top: int,
    left: int,
    window_size: int,
    padding_mode: str,
    padding_value: float = 0.0,
) -> torch.Tensor:
    height, width = (int(value) for value in tensor.shape[-2:])
    crop = tensor[
        ..., top : min(top + window_size, height), left : min(left + window_size, width)
    ]
    pad_h = window_size - int(crop.shape[-2])
    pad_w = window_size - int(crop.shape[-1])
    if pad_h or pad_w:
        if padding_mode == "replicate":
            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            crop = F.pad(
                crop,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=float(padding_value),
            )
    return crop


class RandomCropDemFeatureFloodDomainDataset(
    dem_common.DemFeatureFloodDomainDataset
):
    """Use the reference sample exactly, then expose a deterministic train crop."""

    def __init__(
        self,
        spec: baseline.DomainSpec,
        normalizer: baseline.ChannelNormalizer,
        noise_std: float = 0.0,
    ) -> None:
        self.window_noise_std = float(noise_std)
        # Noise is added after the crop with a sample-local deterministic RNG.
        super().__init__(spec, normalizer, noise_std=0.0)

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        base_index, epoch, global_sample_index = _decode_index(index)
        sample = super().__getitem__(base_index)
        if self.spec.role != "train":
            return sample

        window_size = int(_CONFIG.window_size)
        original_height = int(sample["height"])
        original_width = int(sample["width"])
        seed = stable_int_seed(
            _CONFIG.window_seed,
            epoch,
            global_sample_index,
            self.spec.name,
            sample["sample_id"],
            sample["time_index"],
        )
        rng = random.Random(seed)
        top = rng.randrange(max(original_height - window_size, 0) + 1)
        left = rng.randrange(max(original_width - window_size, 0) + 1)

        sample["x"] = _crop_and_pad(
            sample["x"],
            top=top,
            left=left,
            window_size=window_size,
            padding_mode="constant",
        )
        sample["y"] = _crop_and_pad(
            sample["y"],
            top=top,
            left=left,
            window_size=window_size,
            padding_mode="constant",
        )
        sample["mask"] = _crop_and_pad(
            sample["mask"],
            top=top,
            left=left,
            window_size=window_size,
            padding_mode="constant",
        ).bool()
        sample["static"] = _crop_and_pad(
            sample["static"],
            top=top,
            left=left,
            window_size=window_size,
            padding_mode="replicate",
        )
        if not bool(sample["mask"].any()):
            raise ValueError(
                f"{self.spec.name}/{sample['sample_id']}@{sample['time_index']} "
                f"crop ({top},{left}) contains no valid target pixels"
            )
        if self.window_noise_std > 0.0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_int_seed(seed, "noise"))
            noise = torch.randn(
                sample["x"].shape,
                generator=generator,
                dtype=sample["x"].dtype,
            )
            sample["x"] = sample["x"] + (
                noise
                * self.window_noise_std
                * sample["mask"].to(dtype=sample["x"].dtype)
            )

        sample.update(
            {
                "height": window_size,
                "width": window_size,
                "original_height": original_height,
                "original_width": original_width,
                "crop_top": top,
                "crop_left": left,
                "window_epoch": epoch,
                "global_sample_index": global_sample_index,
                "window_seed": int(_CONFIG.window_seed),
            }
        )
        expected = (window_size, window_size)
        for key in ("x", "y", "static", "mask"):
            if tuple(sample[key].shape[-2:]) != expected:
                raise AssertionError(
                    f"{key} crop {tuple(sample[key].shape[-2:])} != {expected}"
                )
        return sample


def windowed_homogeneous_collate(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    batch = baseline.homogeneous_collate.__window_base__(samples)  # type: ignore[attr-defined]
    for key in (
        "original_height",
        "original_width",
        "crop_top",
        "crop_left",
        "window_epoch",
        "global_sample_index",
        "window_seed",
    ):
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]
    return batch


class DeterministicWindowBatchSampler(Sampler[List[Tuple[int, int, int]]]):
    """Emit epoch/global-index tuples so persistent workers remain reproducible."""

    def __init__(
        self,
        dataset: RandomCropDemFeatureFloodDomainDataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        del seed
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch_size/rank/world_size")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(_CONFIG.window_seed) + int(dataset.spec.domain_id) * 10_007
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _reference_batches_per_rank(self) -> int:
        batches_per_file = math.ceil(
            self.dataset.pairs_per_sequence / self.batch_size
        )
        return math.ceil(
            (len(self.dataset.files) * batches_per_file) / self.world_size
        )

    def __len__(self) -> int:
        if _CONFIG.batches_per_rank is not None:
            return int(_CONFIG.batches_per_rank)
        return self._reference_batches_per_rank()

    def _ordered_base_indices(self, needed: int) -> List[int]:
        result: List[int] = []
        cycle = 0
        while len(result) < needed:
            rng = random.Random(
                stable_int_seed(self.seed, self.epoch, "base-order", cycle)
            )
            files = list(range(len(self.dataset.files)))
            rng.shuffle(files)
            for file_index in files:
                times = list(range(self.dataset.pairs_per_sequence))
                rng.shuffle(times)
                base = file_index * self.dataset.pairs_per_sequence
                result.extend(base + time_index for time_index in times)
                if len(result) >= needed:
                    break
            cycle += 1
        return result[:needed]

    def _all_batches(self) -> List[List[Tuple[int, int, int]]]:
        total_batches = len(self) * self.world_size
        needed = total_batches * self.batch_size
        base_indices = self._ordered_base_indices(needed)
        batches: List[List[Tuple[int, int, int]]] = []
        for batch_index in range(total_batches):
            start = batch_index * self.batch_size
            batches.append(
                [
                    (
                        base_indices[start + offset],
                        self.epoch,
                        start + offset,
                    )
                    for offset in range(self.batch_size)
                ]
            )
        return batches

    def __iter__(self) -> Iterator[List[Tuple[int, int, int]]]:
        yield from self._all_batches()[self.rank :: self.world_size]


def install_base_collate(base_collate: Any) -> None:
    """Attach the original collate without a module-global circular alias."""

    setattr(windowed_homogeneous_collate, "__window_base__", base_collate)
    # The callable uses this stable attribute through the baseline hook.
    setattr(baseline.homogeneous_collate, "__window_base__", base_collate)
