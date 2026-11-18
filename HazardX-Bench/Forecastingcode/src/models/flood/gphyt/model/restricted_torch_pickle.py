"""Minimal pickle allowlist for legacy tensor-only ``torch.save`` archives.

PyTorch 2.0's ``weights_only`` loader cannot deserialize harmless
``argparse.Namespace`` metadata used by the official DPOT checkpoint.  This
module is supplied as ``pickle_module`` to ``torch.load`` and refuses every
global except the four symbols found by a non-executing ``pickletools`` audit
of that archive.  It is deliberately not a general-purpose pickle loader.
"""

from __future__ import annotations

import argparse
import collections
import io
import pickle

import torch


_ALLOWED = {
    ("argparse", "Namespace"): argparse.Namespace,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return _ALLOWED[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"checkpoint global is not allowlisted: {module}.{name}"
            ) from exc


Unpickler = RestrictedUnpickler
UnpicklingError = pickle.UnpicklingError
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps

