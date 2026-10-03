#!/usr/bin/env python3
"""Generate deterministic CAS Landslide train/val/test image lists."""

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio + test-ratio must be smaller than 1.")

    images = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.parent.name.lower() == "img"
        and path.suffix.lower() in {".tif", ".tiff"}
    )
    paired = []
    for image in images:
        parts = list(image.parts)
        index = next(i for i, part in enumerate(parts) if part.lower() == "img")
        parts[index] = "mask"
        mask = Path(*parts)
        candidates = [mask.with_suffix(suffix) for suffix in (".tif", ".tiff", ".TIF", ".TIFF")]
        if any((root / candidate).is_file() for candidate in candidates):
            paired.append(image)
    if not paired:
        raise FileNotFoundError(f"No paired img/mask TIFF files found under {root}.")

    random.Random(args.seed).shuffle(paired)
    test_count = round(len(paired) * args.test_ratio)
    val_count = round(len(paired) * args.val_ratio)
    splits = {
        "test": paired[:test_count],
        "val": paired[test_count : test_count + val_count],
        "train": paired[test_count + val_count :],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        path = args.output_dir / f"{name}.txt"
        path.write_text(
            "".join(f"{sample.as_posix()}\n" for sample in samples),
            encoding="utf-8",
        )
        print(f"{name}: {len(samples)} -> {path}")


if __name__ == "__main__":
    main()
