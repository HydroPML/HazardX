"""Audited pretrained checkpoint loading for FloodCastBench PT-FT models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

from gphyt.model.official_import import sha256_file
from gphyt.model.safe_torch_loader import safe_torch_load


TensorConverter = Callable[[str, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]


@dataclass
class PretrainedLoadingReport:
    exact_loaded: list[str] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    newly_initialized: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[dict] = field(default_factory=list)
    loaded_parameter_count: int = 0
    backbone_parameter_count: int = 0
    backbone_loading_ratio: float = 0.0
    checkpoint_sha256: str = ""

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return output


def _has_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    current = checkpoint
    for _ in range(4):
        if not isinstance(current, Mapping):
            break
        tensor_items = {
            str(key): value
            for key, value in current.items()
            if isinstance(value, torch.Tensor)
        }
        if tensor_items and len(tensor_items) == len(current):
            return tensor_items
        for key in ("model", "state_dict", "model_state_dict", "module"):
            if key in current and isinstance(current[key], Mapping):
                current = current[key]
                break
        else:
            if tensor_items:
                return tensor_items
            break
    raise TypeError("checkpoint does not contain a tensor state_dict")


def normalize_checkpoint_keys(
    state: Mapping[str, torch.Tensor], prefixes: Iterable[str] = ()
) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    removable = tuple(prefixes) + ("module._orig_mod.", "module.", "_orig_mod.")
    for original_key, value in state.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in removable:
                if prefix and key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if key in normalized:
            raise KeyError(f"checkpoint key normalization collision: {key}")
        normalized[key] = value
    return normalized


def load_torch_checkpoint_state(
    path: str | Path,
    *,
    key_prefixes: Iterable[str] = (),
) -> tuple[Dict[str, torch.Tensor], str]:
    """Load a trusted PyTorch checkpoint and return normalized tensor keys.

    PyTorch pickle checkpoints can execute code while loading.  This function
    must only be used with a checkpoint obtained from the documented official
    repository and verified against its published SHA256 before invocation.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    digest = sha256_file(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    return normalize_checkpoint_keys(
        extract_state_dict(checkpoint), prefixes=key_prefixes
    ), digest


def load_pretrained_with_report(
    model: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    checkpoint_sha256: str,
    report_path: str | Path,
    converters: Mapping[str, TensorConverter] | None = None,
    newly_initialized_prefixes: Iterable[str] = (),
    backbone_excluded_prefixes: Iterable[str] = (),
    minimum_backbone_ratio: float = 0.90,
) -> PretrainedLoadingReport:
    """Load all compatible tensors and account for every remaining key."""
    converters = dict(converters or {})
    model_state = model.state_dict()
    loadable: MutableMapping[str, torch.Tensor] = {}
    report = PretrainedLoadingReport(checkpoint_sha256=checkpoint_sha256)

    for key, source in checkpoint_state.items():
        if key not in model_state:
            report.unexpected.append(key)
            continue
        target = model_state[key]
        if tuple(source.shape) == tuple(target.shape):
            loadable[key] = source.to(dtype=target.dtype)
            report.exact_loaded.append(key)
            continue
        converter = converters.get(key)
        converted = converter(key, source, target) if converter else None
        if converted is not None:
            if tuple(converted.shape) != tuple(target.shape):
                raise AssertionError(
                    f"converter for {key} returned {tuple(converted.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            loadable[key] = converted.to(dtype=target.dtype)
            report.converted.append(key)
        else:
            report.shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(source.shape),
                    "model_shape": list(target.shape),
                }
            )

    loaded_keys = set(loadable)
    for key in model_state:
        if key in loaded_keys:
            continue
        if _has_prefix(key, newly_initialized_prefixes):
            report.newly_initialized.append(key)
        else:
            report.missing.append(key)

    incompatible = model.load_state_dict(loadable, strict=False)
    if set(incompatible.unexpected_keys):
        raise AssertionError(f"unexpected keys after filtered load: {incompatible.unexpected_keys}")
    if set(incompatible.missing_keys) != (set(model_state) - loaded_keys):
        raise AssertionError("PyTorch missing-key result disagrees with loading report")

    named_parameters = dict(model.named_parameters())
    backbone_keys = {
        key
        for key in named_parameters
        if not _has_prefix(key, backbone_excluded_prefixes)
    }
    report.backbone_parameter_count = sum(
        named_parameters[key].numel() for key in backbone_keys
    )
    report.loaded_parameter_count = sum(
        named_parameters[key].numel()
        for key in backbone_keys
        if key in loaded_keys
    )
    if report.backbone_parameter_count:
        report.backbone_loading_ratio = (
            report.loaded_parameter_count / report.backbone_parameter_count
        )
    report.exact_loaded.sort()
    report.converted.sort()
    report.newly_initialized.sort()
    report.missing.sort()
    report.unexpected.sort()
    report.shape_mismatch.sort(key=lambda item: item["key"])
    report.write(report_path)

    if report.backbone_loading_ratio < float(minimum_backbone_ratio):
        raise RuntimeError(
            f"backbone loading ratio {report.backbone_loading_ratio:.4f} is "
            f"below required {minimum_backbone_ratio:.4f}; report={report_path}"
        )
    return report


def assert_full_finetuning(model: nn.Module) -> None:
    frozen = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and not parameter.requires_grad
    ]
    if frozen:
        raise AssertionError(f"PT-FT model has frozen floating parameters: {frozen}")


def assert_depth_gradients(
    model: nn.Module,
    parameter_names: tuple[str, str, str],
) -> None:
    parameters = dict(model.named_parameters())
    failures = []
    for depth, name in zip(("shallow", "middle", "deep"), parameter_names):
        if name not in parameters:
            failures.append(f"{depth}:{name}:missing")
            continue
        gradient = parameters[name].grad
        if gradient is None:
            failures.append(f"{depth}:{name}:no_gradient")
        elif not torch.isfinite(gradient).all():
            failures.append(f"{depth}:{name}:non_finite")
        elif not bool((gradient != 0).any()):
            failures.append(f"{depth}:{name}:all_zero")
    if failures:
        raise AssertionError("backbone gradient audit failed: " + ", ".join(failures))
