"""Fetch one official initialization at its audited revision, or verify locally."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    rows = json.loads((ROOT / "docs/pretrained_weights.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=[row["model_slug"] for row in rows]
    )
    parser.add_argument("--pretrained-root", type=Path, default=ROOT / "pretrained")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    row = next(row for row in rows if row["model_slug"] == args.model)
    destination = args.pretrained_root / row["local_filename"]
    if not destination.exists():
        if args.verify_only:
            raise FileNotFoundError(destination)
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=row["repository"],
                filename=row["file"],
                revision=row["revision"],
            )
        )
        if checksum(cached) != row["lfs"]["sha256"]:
            raise ValueError(f"Official download hash mismatch: {cached}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Preserve an existing weight rather than silently replacing it.
        with cached.open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=8 << 20)
    if (
        destination.stat().st_size != row["lfs"]["size"]
        or checksum(destination) != row["lfs"]["sha256"]
    ):
        raise ValueError(f"Initialization size/SHA256 mismatch: {destination}")
    print("Verified", destination, row["lfs"]["sha256"])
    if args.model == "prithvi":
        print(
            "Also run: python scripts/setup_prithvi.py --pretrained-root",
            args.pretrained_root,
        )


if __name__ == "__main__":
    main()
