"""Safe loading policy for official legacy PyTorch tensor archives."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from gphyt.model import restricted_torch_pickle


def safe_torch_load(path: str | Path, *, map_location: str = "cpu"):
    """Load tensors without ever invoking Python's unrestricted unpickler.

    The normal path is PyTorch's weights-only loader.  DPOT's official archive
    additionally stores ``argparse.Namespace``; PyTorch 2.0 rejects that safe
    metadata type, so the fallback uses the four-global static allowlist in
    :mod:`restricted_torch_pickle`.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(
            path,
            map_location=map_location,
            pickle_module=restricted_torch_pickle,
        )

