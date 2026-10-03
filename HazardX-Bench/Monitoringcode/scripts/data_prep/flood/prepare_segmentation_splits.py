#!/usr/bin/env python3
"""Create deterministic train/validation/test lists for paired flood TIFFs."""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--image-glob", default="images/**/*.tif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ids = sorted({p.stem for p in args.dataset_path.glob(args.image_glob)})
    if not ids:
        raise SystemExit(
            f"No images match {args.image_glob!r} under {args.dataset_path}"
        )
    random.Random(args.seed).shuffle(ids)
    train_end = int(len(ids) * args.train_ratio)
    val_end = train_end + int(len(ids) * args.val_ratio)
    splits = {
        "train": ids[:train_end],
        "validation": ids[train_end:val_end],
        "test": ids[val_end:],
    }
    for name, values in splits.items():
        path = args.dataset_path / f"{name}.txt"
        path.write_text("\n".join(values) + "\n")
        print(f"{path}: {len(values)} samples")


if __name__ == "__main__":
    main()
