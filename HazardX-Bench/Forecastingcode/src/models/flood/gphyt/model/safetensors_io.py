"""Minimal, non-executing safetensors checkpoint reader."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import torch

from gphyt.model.checkpoint_loading import normalize_checkpoint_keys
from gphyt.model.official_import import sha256_file


def load_safetensors_state(
    path: Union[str, Path],
    *,
    key_prefixes: Iterable[str] = (),
) -> Tuple[Dict[str, torch.Tensor], str]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    raw = checkpoint_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"invalid safetensors file: {checkpoint_path}")
    header_length = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + header_length
    metadata = json.loads(raw[8:header_end].decode("utf-8"))
    dtype_map = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    state: Dict[str, torch.Tensor] = {}
    data = memoryview(raw)[header_end:]
    for key, entry in metadata.items():
        if key == "__metadata__":
            continue
        dtype_name = entry["dtype"]
        if dtype_name not in dtype_map:
            raise TypeError(f"unsupported safetensors dtype {dtype_name!r} for {key}")
        start, end = (int(v) for v in entry["data_offsets"])
        shape = tuple(int(v) for v in entry["shape"])
        tensor = torch.frombuffer(data[start:end], dtype=dtype_map[dtype_name])
        if tensor.numel() != math.prod(shape):
            raise ValueError(f"safetensors byte count disagrees with shape for {key}")
        state[key] = tensor.reshape(shape).clone()
    return (
        normalize_checkpoint_keys(state, prefixes=key_prefixes),
        sha256_file(checkpoint_path),
    )
