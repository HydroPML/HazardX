"""Fail-fast imports and provenance checks for externally installed backbones."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


class OfficialDependencyError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_file(
    root: str | Path,
    relative_path: str,
    expected_sha256: str,
    *,
    repository: str,
    commit: str,
) -> Path:
    path = Path(root).expanduser().resolve() / relative_path
    if not path.is_file():
        raise OfficialDependencyError(
            f"{repository}@{commit} sentinel is missing: {path}"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise OfficialDependencyError(
            f"official source mismatch for {path}: expected {expected_sha256}, "
            f"got {actual}; refusing an unverified architecture"
        )
    return path


def import_official_module(
    module_name: str,
    *,
    root: str | Path | None,
    import_subdirs: Iterable[str] = ("",),
    install_hint: str,
) -> ModuleType:
    if root:
        resolved = Path(root).expanduser().resolve()
        for subdir in reversed(tuple(import_subdirs)):
            candidate = str(resolved / subdir)
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise OfficialDependencyError(
            f"cannot import official module {module_name!r}. {install_hint}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
