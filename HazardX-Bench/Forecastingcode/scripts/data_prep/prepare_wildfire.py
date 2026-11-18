"""Run the original WildfireSpreadTS TIFF-to-HDF5 converter in isolation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    official = ROOT / "src/models/third_party/WildfireSpreadTS"
    converter = official / "src/preprocess/CreateHDF5Dataset.py"
    # Its own `src` package must not resolve to the release's common `src`.
    env = dict(os.environ, PYTHONPATH=str(official))
    # PyTorch 2.5 removed this typing-only export used by the upstream source.
    # Restore the annotation alias without touching conversion calculations.
    launcher = (
        "import runpy,sys; from typing import TypeVar; "
        "import torch.utils.data.dataset as d; "
        "d.T_co=getattr(d,'T_co',TypeVar('T_co',covariant=True)); "
        "script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            launcher,
            str(converter),
            "--data_dir",
            str(args.data_dir.resolve()),
            "--target_dir",
            str(args.target_dir.resolve()),
        ],
        cwd=official,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
