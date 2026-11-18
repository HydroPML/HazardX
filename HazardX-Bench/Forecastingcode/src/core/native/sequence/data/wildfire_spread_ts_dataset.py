"""WildfireSpreadTS 5-day to next-day dataset.

The numerical preprocessing is delegated to the verified official
``FireSpreadDataset`` implementation.  This wrapper owns only benchmark
protocol: event-safe consecutive windows, leakage-free 128x128 crops, and
metadata/audit output.  It deliberately does not cache already processed
tensors, so 23 -> 40 processing happens exactly once.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


RAW_CHANNELS = 23
PROCESSED_CHANNELS = 40
HISTORY_DAYS = 5
OUTPUT_SIZE = 128
FOLDS: Tuple[Tuple[int, int, int, int], ...] = (
    (2018, 2019, 2020, 2021), (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021), (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020), (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021), (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020), (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019), (2020, 2021, 2019, 2018),
)


@dataclass(frozen=True)
class Window:
    year: int
    event_id: str
    inputs: Tuple[Path, ...]
    target: Path
    input_dates: Tuple[str, ...]
    target_date: str
    hdf5_path: Path | None = None
    hdf5_start: int = -1


def fold_years(fold_id: int) -> Tuple[List[int], List[int], List[int]]:
    if not 0 <= fold_id < len(FOLDS):
        raise ValueError(f"fold_id must be in [0, 11], got {fold_id}")
    fold = FOLDS[fold_id]
    return list(fold[:2]), [fold[2]], [fold[3]]


def _load_official_dataset_class(official_root: str):
    root = Path(official_root).resolve()
    loader = root / "src/dataloader/FireSpreadDataset.py"
    utils = root / "src/dataloader/utils.py"
    if not loader.is_file() or not utils.is_file():
        raise FileNotFoundError(
            f"WildfireSpreadTS official loader not found below {root}; expected {loader}"
        )
    # Import as a package so the official relative import `.utils` is preserved.
    package_name = "_wsts_official_dataloader"
    package = type(sys)(package_name)
    package.__path__ = [str(loader.parent)]
    sys.modules[package_name] = package
    import torch.utils.data.dataset as torch_dataset
    if not hasattr(torch_dataset, "T_co"):
        from typing import TypeVar
        torch_dataset.T_co = TypeVar("T_co", covariant=True)
    for stem, path in (("utils", utils), ("FireSpreadDataset", loader)):
        name = f"{package_name}.{stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import official source {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.FireSpreadDataset"].FireSpreadDataset


def _dates_are_consecutive(paths: Sequence[Path]) -> bool:
    dates = [date.fromisoformat(path.stem) for path in paths]
    return all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:]))


def discover_windows(data_root: str, years: Iterable[int]) -> List[Window]:
    root = Path(data_root).resolve()
    windows: List[Window] = []
    for year in sorted(int(y) for y in years):
        for event_dir in sorted((root / str(year)).glob("fire_*")):
            daily = tuple(sorted(event_dir.glob("*.tif")))
            for start in range(max(0, len(daily) - HISTORY_DAYS)):
                paths = daily[start : start + HISTORY_DAYS + 1]
                if len(paths) != HISTORY_DAYS + 1 or not _dates_are_consecutive(paths):
                    continue
                windows.append(Window(
                    year=year,
                    event_id=event_dir.name,
                    inputs=tuple(paths[:-1]), target=paths[-1],
                    input_dates=tuple(p.stem for p in paths[:-1]),
                    target_date=paths[-1].stem,
                ))
    return windows


def _decode_hdf5_date(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).replace(".tif", "")


def discover_hdf5_windows(hdf5_root: str, years: Iterable[int]) -> List[Window]:
    """Discover the same consecutive windows from official per-fire HDF5 files."""
    root = Path(hdf5_root).resolve()
    windows: List[Window] = []
    for year in sorted(int(y) for y in years):
        for hdf5_path in sorted((root / str(year)).glob("*.hdf5")):
            with h5py.File(hdf5_path, "r") as handle:
                if "data" not in handle:
                    raise RuntimeError(f"{hdf5_path}: missing dataset 'data'")
                dataset = handle["data"]
                if dataset.ndim != 4 or dataset.shape[1] != RAW_CHANNELS:
                    raise RuntimeError(
                        f"{hdf5_path}: expected [days,{RAW_CHANNELS},H,W], got {dataset.shape}"
                    )
                if "img_dates" not in dataset.attrs:
                    raise RuntimeError(f"{hdf5_path}: missing data.attrs['img_dates']")
                dates = tuple(_decode_hdf5_date(v) for v in dataset.attrs["img_dates"])
                event_attr = dataset.attrs.get("fire_name", hdf5_path.stem)
                if isinstance(event_attr, bytes):
                    event_attr = event_attr.decode("utf-8")
                event_id = str(event_attr)
                if len(dates) != dataset.shape[0]:
                    raise RuntimeError(
                        f"{hdf5_path}: {len(dates)} dates for {dataset.shape[0]} days"
                    )
            for start in range(max(0, len(dates) - HISTORY_DAYS)):
                current = dates[start:start + HISTORY_DAYS + 1]
                parsed = [date.fromisoformat(value) for value in current]
                if not all(b - a == timedelta(days=1) for a, b in zip(parsed, parsed[1:])):
                    continue
                windows.append(Window(
                    year=year, event_id=event_id, inputs=(), target=Path(current[-1]),
                    input_dates=tuple(current[:-1]), target_date=current[-1],
                    hdf5_path=hdf5_path, hdf5_start=start,
                ))
    return windows


class WildfireSpreadTS5DayDataset(Dataset):
    """Canonical output: input [5,40,128,128], target [1,128,128]."""

    def __init__(self, *, data_root: str, years: Sequence[int], stats_years: Sequence[int],
                 official_root: str, split: str, seed: int = 0,
                 backend: str = "tiff", hdf5_root: str | None = None) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.data_root = str(Path(data_root).resolve())
        self.years = tuple(int(y) for y in years)
        self.stats_years = tuple(int(y) for y in stats_years)
        self.split = split
        self.seed = int(seed)
        if backend not in {"tiff", "hdf5"}:
            raise ValueError(f"backend must be 'tiff' or 'hdf5', got {backend!r}")
        self.backend = backend
        self.hdf5_root = str(Path(hdf5_root).resolve()) if hdf5_root else None
        if self.backend == "hdf5" and self.hdf5_root is None:
            raise ValueError("hdf5_root is required when backend='hdf5'")
        self.windows = (
            discover_hdf5_windows(self.hdf5_root, self.years)
            if self.backend == "hdf5" else discover_windows(self.data_root, self.years)
        )
        if not self.windows:
            raise RuntimeError(f"no consecutive six-day windows for years {self.years}")
        official_cls = _load_official_dataset_class(official_root)
        # We invoke only the official preprocessing method; inventory/cropping is
        # benchmark-owned to enforce consecutive dates and prevent target-guided crop leakage.
        self.processor = official_cls(
            data_dir=self.hdf5_root if self.backend == "hdf5" else self.data_root,
            included_fire_years=list(self.years),
            n_leading_observations=HISTORY_DAYS, crop_side_length=OUTPUT_SIZE,
            load_from_hdf5=self.backend == "hdf5", is_train=False, remove_duplicate_features=False,
            stats_years=list(self.stats_years), features_to_keep=None, return_doy=False,
        )
        self.raw_channel_names = tuple(
            self.processor.map_channel_index_to_features(only_base=True).values()
        )
        self.processed_channel_names = tuple(
            self.processor.map_channel_index_to_features(only_base=False).values()
        )
        if len(self.raw_channel_names) != RAW_CHANNELS or len(self.processed_channel_names) != PROCESSED_CHANNELS:
            raise RuntimeError("official WildfireSpreadTS channel mapping is not 23 -> 40")

    def __len__(self) -> int:
        return len(self.windows)

    def _read(self, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        if self.backend == "hdf5":
            if window.hdf5_path is None or window.hdf5_start < 0:
                raise RuntimeError(f"invalid HDF5 window metadata for {window.event_id}")
            with h5py.File(window.hdf5_path, "r") as handle:
                stack = handle["data"][window.hdf5_start:window.hdf5_start + HISTORY_DAYS + 1]
            if stack.shape[0] != HISTORY_DAYS + 1:
                raise RuntimeError(f"short HDF5 window in {window.hdf5_path}")
            return stack[:-1], stack[-1, -1]
        arrays = []
        descriptions = None
        for path in (*window.inputs, window.target):
            with rasterio.open(path, "r") as src:
                if src.count != RAW_CHANNELS:
                    raise RuntimeError(f"{path}: expected 23 bands, got {src.count}")
                current = tuple(src.descriptions)
                if descriptions is None:
                    descriptions = current
                elif current != descriptions:
                    # Historical files differ only in the wording of forecast precipitation;
                    # band positions remain authoritative and are verified separately in audit.
                    if current[:17] != descriptions[:17] or current[18:] != descriptions[18:]:
                        raise RuntimeError(f"channel-order drift within {window.event_id}")
                arrays.append(src.read())
        stack = np.stack(arrays)
        return stack[:-1], stack[-1, -1]

    def _augment_raw_target_independent(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Official geometric transforms with a target-independent crop choice.

        This intentionally replaces only the official target-ranked crop, which
        would leak t+1. Angle corrections remain byte-for-byte equivalent in
        ordering (crop, flips, rotation) and occur before sine/standardization.
        """
        height, width = x.shape[-2:]
        if height < OUTPUT_SIZE or width < OUTPUT_SIZE:
            raise RuntimeError(f"native sample {(height,width)} is smaller than 128x128")
        top = int(np.random.randint(0, height - OUTPUT_SIZE + 1));left = int(np.random.randint(0, width - OUTPUT_SIZE + 1))
        x=x[...,top:top+OUTPUT_SIZE,left:left+OUTPUT_SIZE].copy();y=y[...,top:top+OUTPUT_SIZE,left:left+OUTPUT_SIZE].copy()
        degree=(7,13,19)
        if bool(np.random.random()>0.5):
            x=np.flip(x,-1).copy();y=np.flip(y,-1).copy();x[:,degree]=360-x[:,degree]
        if bool(np.random.random()>0.5):
            x=np.flip(x,-2).copy();y=np.flip(y,-2).copy();x[:,degree]=(180-x[:,degree])%360
        rotate=int(np.floor(np.random.random()*4))
        if rotate:
            x=np.rot90(x,rotate,axes=(-2,-1)).copy();y=np.rot90(y,rotate,axes=(-2,-1)).copy();x[:,degree]=(x[:,degree]-90*rotate)%360
        return x,y

    def _crop(self, x: torch.Tensor, y: torch.Tensor, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        if height < OUTPUT_SIZE or width < OUTPUT_SIZE:
            raise RuntimeError(f"native sample {(height, width)} is smaller than 128x128")
        if self.split == "train":
            # Deterministic per (seed,index,worker RNG seed), independent of t+1 target.
            rng = random.Random((self.seed + 1) * 1_000_003 + index)
            top = rng.randrange(height - OUTPUT_SIZE + 1)
            left = rng.randrange(width - OUTPUT_SIZE + 1)
        else:
            top = (height - OUTPUT_SIZE) // 2
            left = (width - OUTPUT_SIZE) // 2
        return (
            x[..., top:top + OUTPUT_SIZE, left:left + OUTPUT_SIZE],
            y[..., top:top + OUTPUT_SIZE, left:left + OUTPUT_SIZE],
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        window = self.windows[index]
        raw_x, raw_y = self._read(window)
        if self.split == "train":
            raw_x, raw_y = self._augment_raw_target_independent(raw_x, raw_y)
        # Official is_train=False center-crops to a multiple of 32 before preprocessing.
        # Our final 128 crop occurs after the same value transforms and is target-independent.
        x, y = self.processor.preprocess_and_augment(raw_x, raw_y)
        if tuple(x.shape[:2]) != (HISTORY_DAYS, PROCESSED_CHANNELS):
            raise RuntimeError(f"official preprocessing returned {tuple(x.shape)}")
        x, y = self._crop(x, y.unsqueeze(0).float(), index)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError(f"non-finite processed sample {window.event_id}/{window.target_date}")
        return {
            "input": x.float(), "target": y.float(), "event_id": window.event_id,
            "year": window.year, "input_dates": window.input_dates,
            "day_of_year": torch.tensor([date.fromisoformat(d).timetuple().tm_yday for d in window.input_dates], dtype=torch.long),
            "target_date": window.target_date,
        }


def assert_event_disjoint(*datasets: WildfireSpreadTS5DayDataset) -> None:
    sets = [{(w.year, w.event_id) for w in ds.windows} for ds in datasets]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"event leakage between split {i} and {j}: {sorted(overlap)[:5]}")
