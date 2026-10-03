"""Single-image flood segmentation adapters for GF-FloodNet and FloodPlanet."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from torch.utils.data import Dataset


def _read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def _read_ids(path: Optional[str]):
    if not path:
        return None
    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(
            f"Split file does not exist: {list_path}. Run the dataset split "
            "preparation script first."
        )
    return {
        Path(line.strip()).stem
        for line in list_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


class _PairedTiffFloodDataset(Dataset):
    image_subdir = "images"
    label_subdir = "annotations"

    def __init__(
        self,
        dataset_path: str,
        data_list_path: Optional[str] = None,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        image_subdir: Optional[str] = None,
        label_subdir: Optional[str] = None,
        channels=None,
        label_flood_value: Optional[int] = None,
        label_ignore_value: Optional[int] = None,
        scale: Optional[float] = None,
        **_,
    ):
        self.root = Path(dataset_path)
        self.transforms = transforms
        self.crop_size = crop_size
        self.split = split
        self.channels = channels
        self.label_flood_value = label_flood_value
        self.label_ignore_value = label_ignore_value
        self.scale = scale

        image_dir = self.root / (image_subdir or self.image_subdir)
        label_dir = self.root / (label_subdir or self.label_subdir)
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected image and label directories: {image_dir}, {label_dir}"
            )

        selected = _read_ids(data_list_path)
        label_by_stem = {
            p.stem: p for p in label_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        }
        self.samples = []
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            if selected is not None and image_path.stem not in selected:
                continue
            label_path = label_by_stem.get(image_path.stem)
            if label_path is not None:
                self.samples.append((image_path, label_path))
        if not self.samples:
            raise RuntimeError(
                f"No matching TIFF image/label pairs found below {self.root}"
            )

    def __len__(self):
        return len(self.samples)

    def _map_label(self, raw: np.ndarray) -> np.ndarray:
        raw = np.squeeze(raw)
        if self.label_flood_value is None:
            label = (raw != 0).astype(np.int64)
        else:
            label = (raw == self.label_flood_value).astype(np.int64)
        if self.label_ignore_value is not None:
            label[raw == self.label_ignore_value] = 255
        return label

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = _read_tif(image_path).astype(np.float32)
        if self.channels is not None:
            image = image[list(self.channels)]
        if self.scale:
            image /= float(self.scale)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.moveaxis(image, 0, -1)
        label = self._map_label(_read_tif(label_path))

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]

        if image.ndim == 2:
            image = image[..., None]
        image = np.ascontiguousarray(np.moveaxis(image, -1, 0))
        return (
            image.astype(np.float32, copy=False),
            np.asarray(label, dtype=np.int64),
            image_path.stem,
        )


class GFFloodNetDataset(_PairedTiffFloodDataset):
    """GF-FloodNet: 5-band GF-2/GF-3 images and binary TIFF masks."""

    image_subdir = "images"
    label_subdir = "annotations"

    def __init__(self, *args, label_flood_value=None, **kwargs):
        super().__init__(
            *args, label_flood_value=label_flood_value, **kwargs
        )


class FloodPlanetDataset(_PairedTiffFloodDataset):
    """FloodPlanet single-sensor segmentation.

    The official tree is ``CSDAP_complete/<region>/<sensor>/*.tif`` with
    ``CSDAP_complete/<region>/labels/*.tif``. PlanetScope (PS) uses four
    channels. Label values are 0=no-data, 1=not flooded and 2=flooded.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: Optional[str] = None,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        sensor: str = "PS",
        channels=None,
        scale: Optional[float] = None,
        **kwargs,
    ):
        self.root = Path(dataset_path)
        self.transforms = transforms
        self.crop_size = crop_size
        self.split = split
        self.channels = channels
        self.scale = scale
        selected = _read_ids(data_list_path)

        base = self.root / "CSDAP_complete"
        self.samples = []
        for image_path in sorted(base.glob(f"*/{sensor}/*.tif")):
            if selected is not None and image_path.stem not in selected:
                continue
            label_path = image_path.parent.parent / "labels" / image_path.name
            if label_path.is_file():
                self.samples.append((image_path, label_path))
        if not self.samples:
            raise RuntimeError(
                f"No FloodPlanet {sensor} image/label pairs found below {base}"
            )

    def _map_label(self, raw: np.ndarray) -> np.ndarray:
        raw = np.squeeze(raw)
        label = np.zeros(raw.shape, dtype=np.int64)
        label[raw == 2] = 1
        label[raw == 0] = 255
        return label
