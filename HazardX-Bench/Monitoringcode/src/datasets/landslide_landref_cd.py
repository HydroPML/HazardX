"""LANDREF bi-temporal Sentinel-2 change-detection dataset adapter."""

from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset


class LANDREFCDDataset(Dataset):
    """Read paired pre/post features and the landslide mask from one split HDF5."""

    def __init__(
        self,
        dataset_path: str,
        h5_path: str,
        pre_features: Optional[Sequence[str]] = None,
        post_features: Optional[Sequence[str]] = None,
        scales: Optional[Sequence[float]] = None,
        pre_scales: Optional[Sequence[float]] = None,
        post_scales: Optional[Sequence[float]] = None,
        clip_min: Optional[Sequence[float]] = None,
        clip_max: Optional[Sequence[float]] = None,
        pre_clip_min: Optional[Sequence[float]] = None,
        pre_clip_max: Optional[Sequence[float]] = None,
        post_clip_min: Optional[Sequence[float]] = None,
        post_clip_max: Optional[Sequence[float]] = None,
        mask_key: str = "None_MASK",
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        input_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        input_cfg = input_cfg or {}
        pre_features = pre_features or input_cfg.get("pre_features")
        post_features = post_features or input_cfg.get("post_features")
        scales = scales or input_cfg.get("scales")
        pre_scales = pre_scales or input_cfg.get("pre_scales") or scales
        post_scales = post_scales or input_cfg.get("post_scales") or scales
        clip_min = clip_min or input_cfg.get("clip_min")
        clip_max = clip_max or input_cfg.get("clip_max")
        pre_clip_min = pre_clip_min or input_cfg.get("pre_clip_min") or clip_min
        pre_clip_max = pre_clip_max or input_cfg.get("pre_clip_max") or clip_max
        post_clip_min = post_clip_min or input_cfg.get("post_clip_min") or clip_min
        post_clip_max = post_clip_max or input_cfg.get("post_clip_max") or clip_max
        mask_key = input_cfg.get("mask_key", mask_key)

        if not pre_features or not post_features:
            raise ValueError("LANDREF-CD requires pre_features and post_features.")
        self.path = Path(dataset_path) / h5_path
        self.pre_features = tuple(pre_features)
        self.post_features = tuple(post_features)
        self.pre_scales = np.asarray(
            pre_scales if pre_scales is not None else np.ones(len(self.pre_features)),
            dtype=np.float32,
        )
        self.post_scales = np.asarray(
            post_scales if post_scales is not None else np.ones(len(self.post_features)),
            dtype=np.float32,
        )
        self.pre_clip_min = (
            np.asarray(pre_clip_min, dtype=np.float32) if pre_clip_min is not None else None
        )
        self.pre_clip_max = (
            np.asarray(pre_clip_max, dtype=np.float32) if pre_clip_max is not None else None
        )
        self.post_clip_min = (
            np.asarray(post_clip_min, dtype=np.float32) if post_clip_min is not None else None
        )
        self.post_clip_max = (
            np.asarray(post_clip_max, dtype=np.float32) if post_clip_max is not None else None
        )
        self.mask_key = mask_key
        self.crop_size = crop_size
        self.split = split
        self.transforms = transforms

        for branch, features, branch_scales, lower, upper in (
            ("pre", self.pre_features, self.pre_scales, self.pre_clip_min, self.pre_clip_max),
            ("post", self.post_features, self.post_scales, self.post_clip_min, self.post_clip_max),
        ):
            if len(branch_scales) != len(features) or np.any(branch_scales == 0):
                raise ValueError(
                    f"LANDREF-CD {branch}_scales must have one non-zero value per feature."
                )
            for name, values in (("clip_min", lower), ("clip_max", upper)):
                if values is not None and len(values) != len(features):
                    raise ValueError(
                        f"LANDREF-CD {branch}_{name} must have one value per feature."
                    )
            if lower is not None and upper is not None and np.any(lower > upper):
                raise ValueError(f"LANDREF-CD {branch}_clip_min cannot exceed clip_max.")
        if not self.path.is_file():
            raise FileNotFoundError(f"LANDREF-CD split file not found: {self.path}")

        with h5py.File(self.path, "r") as handle:
            missing = [
                key
                for key in (*self.pre_features, *self.post_features, self.mask_key)
                if key not in handle
            ]
            if missing:
                raise KeyError(f"LANDREF-CD keys missing from {self.path}: {missing}")
            self.length = int(handle[self.mask_key].shape[0])
            raw_ids = handle.attrs.get("IDs_order")
            self.sample_ids = (
                [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in raw_ids
                ]
                if raw_ids is not None
                else [f"{split}_{index:06d}" for index in range(self.length)]
            )

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _read_features(handle, keys, index):
        return np.stack(
            [np.asarray(handle[key][index], dtype=np.float32).squeeze(0) for key in keys],
            axis=-1,
        )

    def __getitem__(self, index: int):
        with h5py.File(self.path, "r") as handle:
            pre = self._read_features(handle, self.pre_features, index)
            post = self._read_features(handle, self.post_features, index)
            mask = np.asarray(handle[self.mask_key][index]).squeeze()

        pre = np.nan_to_num(pre / self.pre_scales.reshape(1, 1, -1), copy=False)
        post = np.nan_to_num(post / self.post_scales.reshape(1, 1, -1), copy=False)
        if self.pre_clip_min is not None or self.pre_clip_max is not None:
            lower = self.pre_clip_min.reshape(1, 1, -1) if self.pre_clip_min is not None else None
            upper = self.pre_clip_max.reshape(1, 1, -1) if self.pre_clip_max is not None else None
            pre = np.clip(pre, lower, upper)
        if self.post_clip_min is not None or self.post_clip_max is not None:
            lower = self.post_clip_min.reshape(1, 1, -1) if self.post_clip_min is not None else None
            upper = self.post_clip_max.reshape(1, 1, -1) if self.post_clip_max is not None else None
            post = np.clip(post, lower, upper)
        mask = (mask > 0).astype(np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=pre, image_post=post, mask=mask)
            pre = transformed["image"]
            post = transformed["image_post"]
            mask = transformed["mask"]

        pre = np.ascontiguousarray(pre.transpose(2, 0, 1)).astype(np.float32, copy=False)
        post = np.ascontiguousarray(post.transpose(2, 0, 1)).astype(np.float32, copy=False)
        mask = np.ascontiguousarray(mask).astype(np.int64, copy=False)
        return pre, post, mask, self.sample_ids[index]
