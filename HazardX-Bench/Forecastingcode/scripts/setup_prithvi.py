"""Place the audited official Prithvi source/config beside its weight file."""

import argparse
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "prithvi_mae.py": "9cee9a8cb17c3859d6365a7e58f723e0f4c08e3e29aa4b8e6523506db70321da",
    "config.json": "748bc314d9b0653b00e3a2eaf14b69b4c27a295d73eb0b85112fe21bd2c36aa8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-root", type=Path, default=ROOT / "pretrained")
    args = parser.parse_args()
    args.pretrained_root.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        source = ROOT / "src/models/third_party/PrithviEO2" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Packaged official source has changed: {source}")
        target = args.pretrained_root / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise FileExistsError(f"Refusing to overwrite different file: {target}")
        else:
            shutil.copy2(source, target)
        print("Verified", target)
    print("Download the official Prithvi_EO_V2_300M.pt into the same directory.")


if __name__ == "__main__":
    main()
