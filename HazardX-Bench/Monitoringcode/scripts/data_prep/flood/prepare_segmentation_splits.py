#!/usr/bin/env python3
"""Create reproducible FloodPlanet splits from matched sensor/label TIFF pairs."""

import argparse
import math
import random
from pathlib import Path


def prepare(dataset_path, sensor='PS', seed=42, train_ratio=0.7,
            val_ratio=0.15, image_glob=None):
    if (not all(math.isfinite(x) for x in (train_ratio, val_ratio))
            or train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1):
        raise ValueError('Require train_ratio > 0, val_ratio >= 0 and train + val < 1')
    if not sensor or '/' in sensor or '\\' in sensor or sensor in {'.', '..'}:
        raise ValueError('sensor must be a single directory name')
    pattern = image_glob or f'CSDAP_complete/*/{sensor}/*.tif'
    images = sorted(p for p in dataset_path.glob(pattern) if p.is_file())
    if not images:
        raise ValueError(f'No images match {pattern!r} under {dataset_path}')
    ids = []
    for image in images:
        # Match FloodPlanetDataset: region/<sensor>/name.tif -> region/labels/name.tif.
        label = image.parent.parent / 'labels' / image.name
        if not label.is_file():
            raise ValueError(f'Missing paired label: {label}')
        ids.append(image.stem)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image IDs across regions; loader selects by stem, '
                         'so splitting these IDs could leak samples between splits')
    random.Random(seed).shuffle(ids)
    train_end = int(len(ids) * train_ratio)
    val_end = train_end + int(len(ids) * val_ratio)
    splits = {'train': ids[:train_end], 'validation': ids[train_end:val_end],
              'test': ids[val_end:]}
    for split, values in splits.items():
        if not values and (split != 'validation' or val_ratio > 0):
            raise ValueError(f'{len(ids)} samples cannot produce a non-empty {split} split '
                             'with the requested ratios')
    for split, values in splits.items():
        path = dataset_path / f'{split}.txt'
        path.write_text(''.join(value + '\n' for value in values))
        print(f'{path}: {len(values)} samples')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_path', type=Path, nargs='?',
                        default=Path('/root/autodl-tmp/datasetcsc/flood/FloodPlanet'))
    parser.add_argument('--sensor', default='PS')
    parser.add_argument('--image-glob', help='Optional image glob; same FloodPlanet pairing layout required')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()
    try:
        prepare(args.dataset_path, args.sensor, args.seed, args.train_ratio,
                args.val_ratio, args.image_glob)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'FloodPlanet preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
