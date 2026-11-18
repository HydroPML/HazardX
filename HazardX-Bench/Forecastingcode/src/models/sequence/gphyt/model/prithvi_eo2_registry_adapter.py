"""Register the audited official Prithvi-EO-2.0 backbone in TerraTorch.

The TerraTorch commit pinned by FloodCastBench predates its built-in
``prithvi_eo_v2_300`` registration. TerraTorch exposes a public registry
extension API, so this module registers the exact official model source
shipped beside the official checkpoint without modifying site-packages.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Tuple

import torch

from gphyt.model.official_import import (
    OfficialDependencyError,
    import_official_module,
    sha256_file,
)


REGISTRY_NAME = "prithvi_eo_v2_300"
OFFICIAL_SOURCE_FILENAME = "prithvi_mae.py"
OFFICIAL_CONFIG_FILENAME = "config.json"
OFFICIAL_SOURCE_SHA256 = (
    "9cee9a8cb17c3859d6365a7e58f723e0f4c08e3e29aa4b8e6523506db70321da"
)
OFFICIAL_CONFIG_SHA256 = (
    "748bc314d9b0653b00e3a2eaf14b69b4c27a295d73eb0b85112fe21bd2c36aa8"
)

_EXPECTED_CONFIG = {
    "architecture": REGISTRY_NAME,
    "img_size": 224,
    "num_frames": 4,
    "patch_size": [1, 16, 16],
    "in_chans": 6,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4,
    "coords_encoding": [],
}


def _official_files(checkpoint_path: str | Path) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    source = checkpoint.parent / OFFICIAL_SOURCE_FILENAME
    config = checkpoint.parent / OFFICIAL_CONFIG_FILENAME
    for path, expected in (
        (source, OFFICIAL_SOURCE_SHA256),
        (config, OFFICIAL_CONFIG_SHA256),
    ):
        if not path.is_file():
            raise OfficialDependencyError(
                f"official Prithvi companion file is missing: {path}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise OfficialDependencyError(
                f"official Prithvi companion SHA256 mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
    return source, config


def _validate_official_config(config_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    pretrained = payload.get("pretrained_cfg")
    if not isinstance(pretrained, dict):
        raise OfficialDependencyError("Prithvi config lacks pretrained_cfg")
    actual = {"architecture": payload.get("architecture")}
    actual.update(
        {
            key: pretrained.get(key)
            for key in _EXPECTED_CONFIG
            if key != "architecture"
        }
    )
    if actual != _EXPECTED_CONFIG:
        raise OfficialDependencyError(
            f"official Prithvi config mismatch: expected {_EXPECTED_CONFIG}, "
            f"got {actual}"
        )


def load_official_prithvi_module(checkpoint_path: str | Path) -> ModuleType:
    """Load the SHA-audited official implementation beside the checkpoint."""

    source, config = _official_files(checkpoint_path)
    _validate_official_config(config)
    module_name = f"_flood_official_prithvi_eo2_{OFFICIAL_SOURCE_SHA256[:16]}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise OfficialDependencyError(f"cannot import official Prithvi source {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(module, "PrithviViT") or not hasattr(
        module, "_interpolate_pos_encoding"
    ):
        raise OfficialDependencyError(
            "official Prithvi source lacks PrithviViT/position interpolation"
        )
    return module


def register_official_prithvi_backbone(
    checkpoint_path: str | Path,
) -> tuple[Any, ModuleType]:
    """Register and return the official backbone through TerraTorch."""

    registry_module = import_official_module(
        "terratorch.registry",
        root=None,
        install_hint=(
            "install the pinned TerraTorch commit; this adapter supplies only "
            "its missing official Prithvi-EO2 constructor"
        ),
    )
    registry = registry_module.BACKBONE_REGISTRY
    official = load_official_prithvi_module(checkpoint_path)
    if REGISTRY_NAME in registry:
        return registry, official

    def prithvi_eo_v2_300(
        *,
        pretrained: bool = False,
        img_size: int | Tuple[int, int] = 224,
        num_frames: int = 1,
        in_chans: int = 3,
        patch_size: tuple[int, int, int] = (1, 16, 16),
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        **kwargs: Any,
    ):
        if pretrained:
            raise ValueError(
                "FloodCastBench performs its own audited local checkpoint load"
            )
        if kwargs:
            raise TypeError(
                f"unsupported local Prithvi constructor arguments: {sorted(kwargs)}"
            )
        required = {
            "patch_size": (1, 16, 16),
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        }
        actual = {
            "patch_size": tuple(patch_size),
            "embed_dim": int(embed_dim),
            "depth": int(depth),
            "num_heads": int(num_heads),
            "mlp_ratio": float(mlp_ratio),
        }
        if actual != required:
            raise ValueError(
                f"refusing non-standard Prithvi-EO2-300M config: {actual}"
            )
        return official.PrithviViT(
            img_size=img_size,
            num_frames=int(num_frames),
            patch_size=tuple(patch_size),
            in_chans=int(in_chans),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            coords_encoding=list(coords_encoding or []),
            coords_scale_learn=bool(coords_scale_learn),
            drop_path=float(drop_path),
        )

    # Registry.register uses the function name as the public model key.
    registry_module.TERRATORCH_BACKBONE_REGISTRY.register(prithvi_eo_v2_300)
    if REGISTRY_NAME not in registry:
        raise OfficialDependencyError(
            "local official Prithvi registration did not reach BACKBONE_REGISTRY"
        )
    return registry, official


def convert_official_position_embedding(
    official_module: ModuleType,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    runtime_height: int,
    runtime_width: int,
) -> torch.Tensor:
    """Convert fixed four-frame/224 position features through official code."""

    if tuple(source.shape) != (1, 785, 1024):
        raise RuntimeError(
            f"unexpected official Prithvi pos_embed shape {tuple(source.shape)}"
        )
    expected_target = (
        1,
        1 + (int(runtime_height) // 16) * (int(runtime_width) // 16),
        1024,
    )
    if tuple(target.shape) != expected_target:
        raise RuntimeError(
            f"unexpected runtime Prithvi pos_embed shape {tuple(target.shape)}, "
            f"expected {expected_target}"
        )
    converted = official_module._interpolate_pos_encoding(
        pos_embed=source,
        grid_size=(4, 14, 14),
        patch_size=(1, 16, 16),
        shape=(1, int(runtime_height), int(runtime_width)),
        embed_dim=1024,
    )
    if tuple(converted.shape) != expected_target:
        raise AssertionError(
            f"official position conversion returned {tuple(converted.shape)}"
        )
    if not torch.isfinite(converted).all() or not bool((converted != 0).any()):
        raise FloatingPointError("converted Prithvi position embedding is invalid")
    return converted
