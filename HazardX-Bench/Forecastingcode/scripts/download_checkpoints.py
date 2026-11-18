"""Download selected benchmark checkpoints and verify their exact SHA256.

Large checkpoints are lossless byte parts because the upload connector has a
100 MiB per-file limit. Reassembly restores the original checkpoint unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(path, item):
    path = Path(path)
    if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
        raise ValueError(f"Size or SHA256 mismatch: {path}")


def download(item, destination, offline):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination, item)
        return
    if offline:
        raise FileNotFoundError(destination)
    try:
        import gdown
    except ImportError as error:
        raise SystemExit(
            "Install the download helper: pip install 'gdown>=5.2,<6'"
        ) from error
    temporary = destination.with_name(destination.name + ".download")
    if gdown.download(id=item["id"], output=str(temporary), quiet=False) is None:
        raise RuntimeError(f"Drive download failed for {item['name']}")
    verify(temporary, item)
    os.replace(temporary, destination)


def restore(row, root, cache, offline=False):
    target = root / row["checkpoint"]
    if target.exists():
        verify(target, row)
        print(f"Verified existing {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if row["format"] == "single_file":
        download({**row["file"], "sha256": row["sha256"]}, target, offline)
    elif row["format"] == "byte_parts":
        local_parts = []
        for part in row["parts"]:
            path = cache / row["dataset"] / row["model_slug"] / part["name"]
            download(part, path, offline)
            local_parts.append(path)
        temporary = target.with_name(target.name + ".assembling")
        checksum = hashlib.sha256()
        with temporary.open("xb") as output:
            for path in local_parts:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        output.write(chunk)
                        checksum.update(chunk)
        if (
            temporary.stat().st_size != row["bytes"]
            or checksum.hexdigest() != row["sha256"]
        ):
            raise ValueError(f"Reassembled checkpoint failed verification: {temporary}")
        # The selected destination is never silently overwritten.
        if target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
    else:
        raise ValueError(
            f"Checkpoint is not available: {row['dataset']}/{row['model_slug']}"
        )
    verify(target, row)
    print(f"Restored and verified {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "checkpoints/manifest.json"
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--model", help="Model slug, for example unet or pde_transformer"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download every selected checkpoint"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "checkpoint_parts")
    parser.add_argument(
        "--offline", action="store_true", help="Reassemble already downloaded parts"
    )
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    if isinstance(rows, dict):
        rows = [rows]
    selected = [
        row
        for row in rows
        if (not args.dataset or row["dataset"] == args.dataset)
        and (not args.model or row["model_slug"] == args.model)
    ]
    if not selected:
        parser.error("No matching dataset/model in the manifest")
    if not (args.all or args.dataset or args.model):
        for row in selected:
            print(
                row["dataset"],
                row["model_slug"],
                f"seed={row['seed']}",
                row["format"],
                row["bytes"],
            )
        return
    for row in selected:
        restore(row, args.root, args.parts_dir, args.offline)


if __name__ == "__main__":
    main()
