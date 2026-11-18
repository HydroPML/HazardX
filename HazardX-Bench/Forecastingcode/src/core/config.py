"""Read one experiment YAML and resolve user-controlled filesystem locations."""

from __future__ import annotations
import copy
import os
from pathlib import Path
from string import Template
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, **overrides):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    required = {
        "dataset",
        "model",
        "seed",
        "backend",
        "paths",
        "native_args",
        "checkpoint",
    }
    if not required.issubset(cfg):
        raise ValueError(f"Missing config keys: {required-set(cfg)}")
    paths = dict(cfg["paths"])
    for key, value in overrides.items():
        if value is not None and key in paths:
            paths[key] = str(value)
    paths = {
        k: (
            str((ROOT / Path(os.path.expandvars(v)).expanduser()).resolve())
            if not Path(os.path.expandvars(v)).expanduser().is_absolute()
            else str(Path(os.path.expandvars(v)).expanduser().resolve())
        )
        for k, v in paths.items()
    }

    def expand(value):
        if isinstance(value, str):
            return Template(value).safe_substitute(paths)
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    cfg = expand(copy.deepcopy(cfg))
    cfg["paths"] = paths
    cfg["config_file"] = str(path)
    cfg["checkpoint"] = str(
        Path(overrides.get("checkpoint") or ROOT / cfg["checkpoint"])
        .expanduser()
        .resolve()
    )
    cfg["native_args"]["output_dir"] = paths["output_dir"]
    return cfg
