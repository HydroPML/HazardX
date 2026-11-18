"""Validate LANDREF HDF5 splits against a training config and export sample IDs."""

import argparse
import json
from pathlib import Path

import h5py
import yaml


def inspect_split(path, features, mask_key, split):
    if not path.is_file():
        raise FileNotFoundError(f"LANDREF split not found: {path}")
    with h5py.File(path, "r") as handle:
        required = set(features) | {mask_key}
        required.discard("CONST_ZERO")
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path}: missing HDF5 keys: {missing}")
        shape = handle[mask_key].shape
        if len(shape) != 4 or shape[1] != 1 or not all(shape):
            raise ValueError(f"{path}: expected non-empty mask (N, 1, H, W), got {shape}")
        for key in sorted(required):
            if not isinstance(handle[key], h5py.Dataset) or handle[key].shape != shape:
                raise ValueError(f"{path}: {key} must have shape {shape}")
            # Read one sample to detect unsupported compression or damaged data.
            handle[key][0]
        raw_ids = handle.attrs.get("IDs_order")
        if raw_ids is None:
            ids = [f"{split}_{index:06d}" for index in range(shape[0])]
        else:
            if getattr(raw_ids, "ndim", 0) != 1 or len(raw_ids) != shape[0]:
                raise ValueError(f"{path}: IDs_order must contain {shape[0]} IDs")
            ids = [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids]
        if any(not value or '\n' in value or '\r' in value for value in ids):
            raise ValueError(f"{path}: sample IDs must be non-empty single-line strings")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate sample IDs")
        return {"path": str(path), "samples": shape[0], "shape": list(shape),
                "features": list(features), "mask_key": mask_key,
                "ids_source": "IDs_order" if raw_ids is not None else "generated"}, ids


def prepare(dataset_root, config_path, output_dir, test_h5=None):
    config = yaml.safe_load(config_path.read_text())
    dataset = config["dataset"]
    if dataset.get("name") not in {"landref", "landref_cd"}:
        raise ValueError("--config must select the landref or landref_cd dataset")
    inputs = dataset.get("input", {})
    features = inputs.get("features") or (
        inputs.get("pre_features", []) + inputs.get("post_features", [])
    )
    if not features:
        raise ValueError("Config must define dataset.input features or pre/post_features")
    mask_key = inputs.get("mask_key", "None_MASK")
    results = {}
    # Validate every split before writing outputs. Official splits are never reshuffled.
    for split in ("train", "val", "test"):
        filename = test_h5 if split == "test" and test_h5 else dataset[split]["h5_path"]
        path = (dataset_root / filename).resolve()
        results[split] = inspect_split(path, features, mask_key, split)
    destinations = [output_dir / (split + '.txt') for split in results]
    destinations.append(output_dir / 'summary.json')
    sources = {Path(info['path']) for info, _ in results.values()} | {config_path.resolve()}
    if any(path.resolve() in sources for path in destinations):
        raise ValueError("Output paths must not overwrite input files")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, (info, ids) in results.items():
        (output_dir / (split + '.txt')).write_text('\n'.join(ids) + '\n')
        print(f"{split}: {info['samples']} samples, {len(features)} selected channels")
    report = {"config": str(config_path.resolve()),
              "splits": {split: info for split, (info, _) in results.items()}}
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path('/root/autodl-tmp/datasetcsc/landslide/LANDREF/reference_data'),
                        help='Directory containing the official LANDREF split HDF5 files')
    parser.add_argument('--config', type=Path,
                        default=repo_root / 'configs/landslide/landref/seg_unet.yaml')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Destination for diagnostic ID lists and summary.json')
    parser.add_argument('--test-h5', help='Override test HDF5 filename (for spatial evaluation)')
    args = parser.parse_args()
    try:
        prepare(args.dataset_root, args.config, args.output_dir, args.test_h5)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'LANDREF preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
