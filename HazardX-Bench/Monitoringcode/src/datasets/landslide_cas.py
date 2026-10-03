"""CAS Landslide RGB semantic-segmentation dataset adapter."""

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CASLandslideDataset(Dataset):
    """Read CAS Landslide TIFF images and binary masks.

    The official archive contains multiple regional subdatasets. Each one has
    ``img``, ``label``, and ``mask`` directories. Split files should contain
    image paths relative to ``dataset_path``, for example
    ``Moxitaidi/UAV/img/0001.tif``.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: Optional[str] = None,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        channel_indices: Optional[Sequence[int]] = None,
        channel_divisors: Optional[Sequence[float]] = None,
        input_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        input_cfg = input_cfg or {}
        channel_indices = channel_indices or input_cfg.get("channel_indices")
        channel_divisors = channel_divisors or input_cfg.get("channel_divisors")
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.channel_indices = (
            tuple(int(index) for index in channel_indices)
            if channel_indices is not None
            else None
        )
        if channel_divisors is None and self.channel_indices is not None:
            channel_divisors = [1.0] * len(self.channel_indices)
        self.channel_divisors = (
            np.asarray(channel_divisors, dtype=np.float32)
            if channel_divisors is not None
            else None
        )
        if (
            self.channel_indices is not None
            and len(self.channel_indices) != len(self.channel_divisors)
        ):
            raise ValueError(
                "CAS channel_divisors must have one value per channel index."
            )
        if self.channel_divisors is not None and np.any(self.channel_divisors == 0):
            raise ValueError("CAS channel divisors must be non-zero.")

        if data_list_path:
            self.images = self._read_data_list(data_list_path)
        else:
            self.images = sorted(
                path.relative_to(self.dataset_path)
                for path in self.dataset_path.rglob("*")
                if path.is_file()
                and path.parent.name.lower() == "img"
                and path.suffix.lower() in {".tif", ".tiff"}
            )

        self.images = [path for path in self.images if self._sample_exists(path)]
        if not self.images:
            raise FileNotFoundError(
                f"No paired CAS Landslide TIFF samples found under {self.dataset_path}."
            )

    @staticmethod
    def _read_data_list(path: str) -> List[Path]:
        with open(path, "r") as handle:
            return [Path(line.strip()) for line in handle if line.strip()]

    def _mask_path(self, image: Path) -> Path:
        parts = list(image.parts)
        img_index = next(
            index for index, part in enumerate(parts) if part.lower() == "img"
        )
        parts[img_index] = "mask"
        candidate = Path(*parts)
        if (self.dataset_path / candidate).is_file():
            return candidate
        # Some CAS regions use `.tif` for images and `.TIF` for masks.
        # Resolve TIFF suffixes case-insensitively on Linux filesystems.
        for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
            alternative = candidate.with_suffix(suffix)
            if (self.dataset_path / alternative).is_file():
                return alternative
        return candidate

    def _sample_exists(self, image: Path) -> bool:
        return (self.dataset_path / image).is_file() and (
            self.dataset_path / self._mask_path(image)
        ).is_file()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        relative_image = self.images[index]
        image = np.asarray(
            Image.open(self.dataset_path / relative_image).convert("RGB"),
            dtype=np.float32,
        )
        mask = np.asarray(
            Image.open(self.dataset_path / self._mask_path(relative_image))
        )
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.int64)
        if self.channel_indices is not None:
            source = image
            image = np.stack(
                [
                    source[..., channel] if channel >= 0 else np.zeros(source.shape[:2])
                    for channel in self.channel_indices
                ],
                axis=-1,
            ).astype(np.float32, copy=False)
            image /= self.channel_divisors.reshape(1, 1, -1)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(
            np.float32, copy=False
        )
        return image, mask.astype(np.int64, copy=False), relative_image.as_posix()
