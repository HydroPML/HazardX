"""Exact scene-trajectory training views for the 128-pixel benchmark.

Only the training view is changed here.  Validation/test continue to use the
existing full-grid datasets and 128/96 overlap-add inference implementation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

import experiments_EarthDynamics_FloodCastBench_multidomain_native_nan_safe as baseline
import floodcast_dem_sensitivity_common as dem_common
from gphyt.data.floodcast_window_utils import stable_int_seed


TRAIN_DOMAINS = ("australia", "pakistan", "uk_30m")
WINDOW_SIZE = 128
EXPECTED_SCENES = {"australia": 8, "pakistan": 10, "uk_30m": 3}
EXPECTED_TILES = {"australia": 16, "pakistan": 18, "uk_30m": 2}
EXPECTED_PAIRS = 287
EXPECTED_TRAJECTORIES = 21
EXPECTED_SAMPLES = EXPECTED_TRAJECTORIES * EXPECTED_PAIRS


@dataclass(frozen=True)
class SceneTileRef:
    domain: str
    scene_index: int
    time_index: int
    tile_id: int
    logical_index: int
    epoch: int


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    global_refs: Tuple[SceneTileRef, ...]
    rank_refs: Tuple[Tuple[Tuple[SceneTileRef, ...], ...], ...]
    updates: int
    effective_batch_size: int

    @property
    def logical_samples(self) -> int:
        return len(self.global_refs)

    def local_microbatches(self, rank: int) -> Tuple[Tuple[SceneTileRef, ...], ...]:
        return self.rank_refs[int(rank)]


@dataclass(frozen=True)
class TrajectoryScheduleConfig:
    seed: int
    micro_batch_size: int
    accumulation_steps: int
    world_size: int

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps * self.world_size


_CONFIG = TrajectoryScheduleConfig(
    seed=42, micro_batch_size=1, accumulation_steps=1, world_size=1
)


def configure_trajectory_schedule(
    *, seed: int, micro_batch_size: int, accumulation_steps: int, world_size: int
) -> TrajectoryScheduleConfig:
    global _CONFIG
    values = (micro_batch_size, accumulation_steps, world_size)
    if any(int(value) < 1 for value in values):
        raise ValueError(f"invalid trajectory schedule dimensions: {values}")
    _CONFIG = TrajectoryScheduleConfig(
        seed=int(seed),
        micro_batch_size=int(micro_batch_size),
        accumulation_steps=int(accumulation_steps),
        world_size=int(world_size),
    )
    return _CONFIG


def nonoverlap_tile_coordinates(
    height: int, width: int, window_size: int = WINDOW_SIZE
) -> Tuple[Tuple[int, int], ...]:
    """Row-major, non-overlapping, complete tiles; boundary remainders are dropped."""

    if height < window_size or width < window_size:
        return ()
    rows = range(0, int(height) - int(window_size) + 1, int(window_size))
    columns = range(0, int(width) - int(window_size) + 1, int(window_size))
    return tuple((top, left) for top in rows for left in columns)


def scene_tile_order(
    *, seed: int, cycle_id: int, domain: str, scene_id: str, tile_count: int
) -> Tuple[int, ...]:
    """One deterministic permutation for one scene and one spatial cycle."""

    tile_ids = list(range(int(tile_count)))
    # Explicitly contains seed + cycle_id + scene identity, as required by the
    # protocol, while stable_int_seed avoids Python hash randomization.
    cycle_seed = stable_int_seed(int(seed) + int(cycle_id), domain, scene_id)
    random.Random(cycle_seed).shuffle(tile_ids)
    return tuple(tile_ids)


def tile_id_for_epoch(
    *, seed: int, epoch: int, domain: str, scene_id: str, tile_count: int
) -> int:
    if epoch < 0 or tile_count < 1:
        raise ValueError("epoch must be nonnegative and tile_count positive")
    cycle_id, offset = divmod(int(epoch), int(tile_count))
    return scene_tile_order(
        seed=seed,
        cycle_id=cycle_id,
        domain=domain,
        scene_id=scene_id,
        tile_count=tile_count,
    )[offset]


def _decode_dataset_index(index: Any) -> Tuple[int, int, int, int]:
    """Return base_index, epoch, tile_id, logical_index.

    The three-field compatibility form is retained for the existing smoke
    harness: ``(base_index, epoch, logical_index)``.
    """

    if isinstance(index, SceneTileRef):
        raise TypeError("SceneTileRef must be routed through the unified dataset")
    if not isinstance(index, tuple):
        value = int(index)
        return value, 0, -1, value
    if len(index) == 4:
        return tuple(int(value) for value in index)  # type: ignore[return-value]
    if len(index) == 3:
        base_index, epoch, logical_index = (int(value) for value in index)
        return base_index, epoch, -1, logical_index
    raise ValueError(f"trajectory dataset index must have 3 or 4 fields: {index!r}")


class SceneTileTrajectoryDomainDataset(dem_common.DemFeatureFloodDomainDataset):
    """One fixed complete tile for all 287 Markov pairs of a scene/epoch."""

    def __init__(
        self,
        spec: baseline.DomainSpec,
        normalizer: baseline.ChannelNormalizer,
        noise_std: float = 0.0,
    ) -> None:
        self.trajectory_noise_std = float(noise_std)
        super().__init__(spec, normalizer, noise_std=0.0)
        if spec.name not in TRAIN_DOMAINS or spec.role != "train":
            raise ValueError(f"trajectory dataset is train-only, got {spec.name}/{spec.role}")
        if len(self.files) != EXPECTED_SCENES[spec.name]:
            raise RuntimeError(
                f"{spec.name}: expected {EXPECTED_SCENES[spec.name]} scenes, "
                f"found {len(self.files)}"
            )
        if self.pairs_per_sequence != EXPECTED_PAIRS:
            raise RuntimeError(
                f"{spec.name}: expected {EXPECTED_PAIRS} Markov pairs, "
                f"found {self.pairs_per_sequence}"
            )
        self.tile_coordinates = nonoverlap_tile_coordinates(
            self.profile.height, self.profile.width
        )
        if len(self.tile_coordinates) != EXPECTED_TILES[spec.name]:
            raise RuntimeError(
                f"{spec.name}: grid {(self.profile.height, self.profile.width)} "
                f"produced {len(self.tile_coordinates)} complete 128 tiles; "
                f"expected {EXPECTED_TILES[spec.name]}"
            )
        self._trajectory_cache_epoch: int | None = None
        self._trajectory_cache: Dict[int, Dict[str, Any]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_trajectory_cache_epoch"] = None
        state["_trajectory_cache"] = {}
        return state

    def selected_tile_id(self, scene_index: int, epoch: int) -> int:
        scene_id = self.files[int(scene_index)].stem
        return tile_id_for_epoch(
            seed=_CONFIG.seed,
            epoch=int(epoch),
            domain=self.spec.name,
            scene_id=scene_id,
            tile_count=len(self.tile_coordinates),
        )

    def _build_cached_trajectory(
        self, scene_index: int, epoch: int, tile_id: int
    ) -> Dict[str, Any]:
        if self._trajectory_cache_epoch != epoch:
            self._trajectory_cache.clear()
            self._trajectory_cache_epoch = int(epoch)
        cached = self._trajectory_cache.get(int(scene_index))
        if cached is not None:
            if int(cached["tile_id"]) != int(tile_id):
                raise AssertionError("one scene received multiple tiles in one epoch")
            return cached

        top, left = self.tile_coordinates[int(tile_id)]
        stop_y, stop_x = top + WINDOW_SIZE, left + WINDOW_SIZE
        sequence = self._load_sequence(int(scene_index))
        data = sequence.data[:, top:stop_y, left:stop_x, :]

        h_t = data[:-1, :, :, self.spec.water_channel]
        rain_t = data[:-1, :, :, self.spec.rain_channel]
        h_next = data[1:, :, :, self.spec.water_channel]
        static = self.dem_feature[..., top:stop_y, left:stop_x].float().contiguous()
        if sequence.mask is None:
            valid = torch.ones_like(h_t, dtype=torch.bool)
        else:
            valid = (
                sequence.mask[:-1, top:stop_y, left:stop_x]
                & sequence.mask[1:, top:stop_y, left:stop_x]
            )
        for values in (h_t, rain_t, h_next):
            valid &= baseline._valid_values(values, self.spec.nodata_value)
        valid &= torch.isfinite(static).all(dim=0).unsqueeze(0)
        valid_counts = valid.flatten(1).sum(dim=1)
        if bool((valid_counts == 0).any()):
            first_bad = int(torch.nonzero(valid_counts == 0)[0].item())
            raise ValueError(
                f"{self.spec.name}/{self.files[scene_index].stem}@{first_bad}: "
                f"complete tile {tile_id} ({top},{left}) has zero valid pixels"
            )

        h_t_n = self.normalizer.normalize(h_t, baseline.WATER_CHANNEL)
        rain_n = self.normalizer.normalize(rain_t, baseline.RAIN_CHANNEL)
        h_next_n = self.normalizer.normalize(h_next, baseline.WATER_CHANNEL)
        h_t_n = torch.where(valid, h_t_n, torch.zeros_like(h_t_n))
        rain_n = torch.where(valid, rain_n, torch.zeros_like(rain_n))
        h_next_n = torch.where(valid, h_next_n, torch.zeros_like(h_next_n))
        x = torch.stack((h_t_n, rain_n), dim=1).float().contiguous()
        y = h_next_n.unsqueeze(1).float().contiguous()
        mask = valid.unsqueeze(1).contiguous()
        metadata = {
            "domain": self.spec.name,
            "domain_id": self.spec.domain_id,
            "sample_id": self.files[scene_index].stem,
            "source_path": str(self.files[scene_index]),
            "dt_minutes": self.spec.dt_minutes,
            "dx": sequence.dx,
            "dy": sequence.dy,
        }
        cached = {
            "tile_id": int(tile_id),
            "top": int(top),
            "left": int(left),
            "x": x,
            "y": y,
            "mask": mask,
            "static": static,
            "metadata": metadata,
        }
        self._trajectory_cache[int(scene_index)] = cached
        # Do not retain a multi-GB native sequence in addition to normalized
        # 128-pixel trajectory caches.
        self._cached_index = None
        self._cached_sequence = None
        return cached

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        base_index, epoch, requested_tile_id, logical_index = _decode_dataset_index(index)
        if not 0 <= base_index < len(self):
            raise IndexError(base_index)
        scene_index, time_index = divmod(base_index, self.pairs_per_sequence)
        selected = self.selected_tile_id(scene_index, epoch)
        tile_id = selected if requested_tile_id < 0 else int(requested_tile_id)
        if tile_id != selected:
            raise AssertionError(
                f"sampler tile {tile_id} != deterministic scene tile {selected}"
            )
        cached = self._build_cached_trajectory(scene_index, epoch, tile_id)
        mask = cached["mask"][time_index]
        x = cached["x"][time_index].clone()
        y = cached["y"][time_index]
        static = torch.where(mask, cached["static"], 0.0)
        if self.trajectory_noise_std > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                stable_int_seed(
                    _CONFIG.seed,
                    epoch,
                    self.spec.name,
                    cached["metadata"]["sample_id"],
                    time_index,
                    "noise",
                )
            )
            noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
            x.add_(noise * self.trajectory_noise_std * mask.to(x.dtype))
        sample = dict(cached["metadata"])
        sample.update(
            {
                "x": x,
                "y": y,
                "static": static,
                "mask": mask,
                "time_index": int(time_index),
                "height": WINDOW_SIZE,
                "width": WINDOW_SIZE,
                "original_height": int(self.profile.height),
                "original_width": int(self.profile.width),
                "crop_top": int(cached["top"]),
                "crop_left": int(cached["left"]),
                "tile_id": int(tile_id),
                "window_epoch": int(epoch),
                "global_sample_index": int(logical_index),
                "window_seed": int(_CONFIG.seed),
                "scene_index": int(scene_index),
            }
        )
        return sample


class UnifiedSceneTileTrajectoryDataset(Dataset):
    """Route globally shuffled references to the three unchanged domain views."""

    def __init__(
        self,
        specs: Mapping[str, baseline.DomainSpec],
        normalizer: baseline.ChannelNormalizer,
        noise_std: float,
    ) -> None:
        self.domains = {
            name: SceneTileTrajectoryDomainDataset(
                specs[name], normalizer=normalizer, noise_std=noise_std
            )
            for name in TRAIN_DOMAINS
        }
        if sum(len(dataset.files) for dataset in self.domains.values()) != EXPECTED_TRAJECTORIES:
            raise AssertionError("training trajectory count is not 21")
        # Constructors inspect one representative full-grid sequence.  The
        # profile is retained, but the potentially multi-GB raw tensors are
        # released before DataLoader workers are forked.
        for dataset in self.domains.values():
            dataset._cached_index = None
            dataset._cached_sequence = None

    def __len__(self) -> int:
        return EXPECTED_SAMPLES

    def __getitem__(self, ref: SceneTileRef) -> Dict[str, Any]:
        if not isinstance(ref, SceneTileRef):
            raise TypeError(f"unified trajectory dataset requires SceneTileRef, got {ref!r}")
        dataset = self.domains[ref.domain]
        base_index = ref.scene_index * EXPECTED_PAIRS + ref.time_index
        return dataset[(base_index, ref.epoch, ref.tile_id, ref.logical_index)]


def build_epoch_plan(
    dataset: UnifiedSceneTileTrajectoryDataset,
    epoch: int,
    config: TrajectoryScheduleConfig | None = None,
) -> EpochPlan:
    config = _CONFIG if config is None else config
    refs: List[SceneTileRef] = []
    logical_index = 0
    for domain in TRAIN_DOMAINS:
        domain_dataset = dataset.domains[domain]
        for scene_index, path in enumerate(domain_dataset.files):
            tile_id = tile_id_for_epoch(
                seed=config.seed,
                epoch=int(epoch),
                domain=domain,
                scene_id=path.stem,
                tile_count=len(domain_dataset.tile_coordinates),
            )
            for time_index in range(EXPECTED_PAIRS):
                refs.append(
                    SceneTileRef(
                        domain=domain,
                        scene_index=scene_index,
                        time_index=time_index,
                        tile_id=tile_id,
                        logical_index=logical_index,
                        epoch=int(epoch),
                    )
                )
                logical_index += 1
    if len(refs) != EXPECTED_SAMPLES:
        raise AssertionError(f"logical sample count {len(refs)} != {EXPECTED_SAMPLES}")
    random.Random(
        stable_int_seed(config.seed, int(epoch), "global-markov-shuffle")
    ).shuffle(refs)

    effective = config.effective_batch_size
    updates = math.ceil(len(refs) / effective)
    rank_microbatches: List[List[Tuple[SceneTileRef, ...]]] = [
        [] for _ in range(config.world_size)
    ]
    local_capacity = config.micro_batch_size * config.accumulation_steps
    for update in range(updates):
        global_chunk = refs[update * effective : (update + 1) * effective]
        per_rank, remainder = divmod(len(global_chunk), config.world_size)
        cursor = 0
        for rank in range(config.world_size):
            local_count = per_rank + (1 if rank < remainder else 0)
            if local_count > local_capacity:
                raise AssertionError("rank assignment exceeds accumulation capacity")
            local = global_chunk[cursor : cursor + local_count]
            cursor += local_count
            for start in range(0, len(local), config.micro_batch_size):
                rank_microbatches[rank].append(
                    tuple(local[start : start + config.micro_batch_size])
                )
    return EpochPlan(
        epoch=int(epoch),
        global_refs=tuple(refs),
        rank_refs=tuple(tuple(items) for items in rank_microbatches),
        updates=updates,
        effective_batch_size=effective,
    )


class TrajectoryEpochBatchSampler(Sampler[List[SceneTileRef]]):
    def __init__(
        self,
        dataset: UnifiedSceneTileTrajectoryDataset,
        rank: int,
        world_size: int,
    ) -> None:
        if world_size != _CONFIG.world_size or not 0 <= rank < world_size:
            raise ValueError("batch sampler rank/world does not match schedule")
        self.dataset = dataset
        self.rank = int(rank)
        self.epoch = 0
        self.plan = build_epoch_plan(dataset, 0)

    def set_epoch(self, epoch: int) -> EpochPlan:
        self.epoch = int(epoch)
        self.plan = build_epoch_plan(self.dataset, self.epoch)
        return self.plan

    def __iter__(self) -> Iterator[List[SceneTileRef]]:
        for refs in self.plan.local_microbatches(self.rank):
            yield list(refs)

    def __len__(self) -> int:
        return len(self.plan.local_microbatches(self.rank))


def trajectory_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty trajectory micro-batch")
    spatial = {tuple(sample["x"].shape) for sample in samples}
    if spatial != {(2, WINDOW_SIZE, WINDOW_SIZE)}:
        raise AssertionError(f"unexpected trajectory input shapes: {spatial}")
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
        "original_height",
        "original_width",
        "crop_top",
        "crop_left",
        "tile_id",
        "window_epoch",
        "global_sample_index",
        "window_seed",
        "scene_index",
    ):
        batch[key] = [sample[key] for sample in samples]
    return batch


def plan_statistics(plan: EpochPlan) -> Dict[str, Any]:
    scene_tiles: Dict[Tuple[str, int], set[int]] = {}
    counts = {name: 0 for name in TRAIN_DOMAINS}
    for ref in plan.global_refs:
        counts[ref.domain] += 1
        scene_tiles.setdefault((ref.domain, ref.scene_index), set()).add(ref.tile_id)
    if any(len(tile_ids) != 1 for tile_ids in scene_tiles.values()):
        raise AssertionError("a scene changed tile inside one epoch")
    return {
        "epoch": plan.epoch,
        "australia_scenes": EXPECTED_SCENES["australia"],
        "pakistan_scenes": EXPECTED_SCENES["pakistan"],
        "uk_30_scenes": EXPECTED_SCENES["uk_30m"],
        "crop_trajectories": len(scene_tiles),
        "markov_samples": len(plan.global_refs),
        "samples_by_domain": counts,
        "effective_batch_size": plan.effective_batch_size,
        "optimizer_updates": plan.updates,
        "tail_samples": len(plan.global_refs) % plan.effective_batch_size,
        "ddp_padding_samples": 0,
    }


def run_schedule_sanity(dataset: UnifiedSceneTileTrajectoryDataset) -> Dict[str, Any]:
    plan0 = build_epoch_plan(dataset, 0)
    repeated = build_epoch_plan(dataset, 0)
    if plan0.global_refs != repeated.global_refs:
        raise AssertionError("fixed-seed global schedule is not reproducible")
    stats = plan_statistics(plan0)
    if stats["crop_trajectories"] != 21 or stats["markov_samples"] != 6027:
        raise AssertionError(f"unexpected epoch statistics: {stats}")

    for domain, domain_dataset in dataset.domains.items():
        for scene_index, path in enumerate(domain_dataset.files):
            tile_count = len(domain_dataset.tile_coordinates)
            first_cycle = [
                tile_id_for_epoch(
                    seed=_CONFIG.seed,
                    epoch=epoch,
                    domain=domain,
                    scene_id=path.stem,
                    tile_count=tile_count,
                )
                for epoch in range(tile_count)
            ]
            if len(set(first_cycle)) != tile_count:
                raise AssertionError(
                    f"{domain}/{path.stem}: tile repeated before pool exhaustion"
                )
            reproduced = [
                tile_id_for_epoch(
                    seed=_CONFIG.seed,
                    epoch=epoch,
                    domain=domain,
                    scene_id=path.stem,
                    tile_count=tile_count,
                )
                for epoch in range(tile_count)
            ]
            if first_cycle != reproduced:
                raise AssertionError(f"{domain}/{path.stem}: tile pool is not reproducible")
    stats.update(
        same_scene_all_287_same_crop=True,
        no_repeat_before_pool_exhaustion=True,
        fixed_seed_reproducible=True,
        global_markov_shuffle=True,
    )
    return stats

