"""LANDREF multi-sensor HDF5 semantic-segmentation dataset adapter."""

from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset


class LANDREFDataset(Dataset):
    """Read one official LANDREF split stored in a multi-dataset HDF5 file.

    ``features`` defines both the selected variables and their channel order.
    Every official feature has shape ``(N, 1, 128, 128)``. Optional ``scales``
    divide each selected feature before augmentation and model input.
    """

    def __init__(
        self,
        dataset_path: str,
        h5_path: str,
        features: Optional[Sequence[str]] = None,
        scales: Optional[Sequence[float]] = None,
        mask_key: str = "None_MASK",
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        input_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        input_cfg = input_cfg or {}
        features = features or input_cfg.get("features")
        scales = scales or input_cfg.get("scales")
        mask_key = input_cfg.get("mask_key", mask_key)
        if not features:
            raise ValueError("LANDREF requires a non-empty feature list.")
        self.path = Path(dataset_path) / h5_path
        self.features = tuple(features)
        self.scales = (
            np.asarray(scales, dtype=np.float32)
            if scales is not None
            else np.ones(len(self.features), dtype=np.float32)
        )
        self.mask_key = mask_key
        self.crop_size = crop_size
        self.split = split
        self.transforms = transforms

        if len(self.scales) != len(self.features):
            raise ValueError("LANDREF scales must have one value per selected feature.")
        if np.any(self.scales == 0):
            raise ValueError("LANDREF feature scales must be non-zero.")
        if not self.path.is_file():
            raise FileNotFoundError(f"LANDREF split file not found: {self.path}")

        with h5py.File(self.path, "r") as handle:
            missing = [
                key
                for key in (*self.features, self.mask_key)
                if key != "CONST_ZERO" and key not in handle
            ]
            if missing:
                raise KeyError(f"LANDREF keys missing from {self.path}: {missing}")
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

    def __getitem__(self, index: int):
        with h5py.File(self.path, "r") as handle:
            channels = [
                (
                    np.zeros(handle[self.mask_key].shape[-2:], dtype=np.float32)
                    if key == "CONST_ZERO"
                    else np.asarray(handle[key][index]).squeeze(0)
                )
                for key in self.features
            ]
            mask = np.asarray(handle[self.mask_key][index]).squeeze()

        image = np.stack(channels, axis=-1).astype(np.float32, copy=False)
        image /= self.scales.reshape(1, 1, -1)
        image = np.nan_to_num(image, copy=False)
        mask = (mask > 0).astype(np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(
            np.float32, copy=False
        )
        return image, mask.astype(np.int64, copy=False), self.sample_ids[index]
