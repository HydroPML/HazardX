from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


_ANYSAT_ROOT = Path("/root/workcsc/AnySat")
_ANYSAT_SRC = _ANYSAT_ROOT / "src"
if _ANYSAT_SRC.exists() and str(_ANYSAT_SRC) not in sys.path:
    sys.path.insert(0, str(_ANYSAT_SRC))

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_local_checkpoint_path(checkpoint_path: Optional[str], model_size: str) -> Optional[Path]:
    def _candidate_from(value: str) -> Optional[Path]:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            repo_candidate = (_REPO_ROOT / candidate).expanduser()
            if repo_candidate.is_file():
                return repo_candidate
        return None

    if checkpoint_path:
        candidate = _candidate_from(checkpoint_path)
        if candidate is None:
            raise FileNotFoundError(f"Specified AnySat checkpoint not found: {checkpoint_path}")
        return candidate

    env_checkpoint = os.environ.get("ANYSAT_CHECKPOINT_PATH")
    if env_checkpoint:
        candidate = _candidate_from(env_checkpoint)
        if candidate is None:
            raise FileNotFoundError(f"Environment AnySat checkpoint not found: {env_checkpoint}")
        return candidate

    default_candidates = {
        "base": (
            _REPO_ROOT / "pretrained_weight" / "AnySat.pth",
            _ANYSAT_ROOT / ".media" / "AnySat.pth",
            _ANYSAT_ROOT / "models" / "AnySat.pth",
        ),
        "small": (),
        "tiny": (),
    }

    for candidate in default_candidates.get(model_size, ()):
        if candidate.is_file():
            return candidate

    return None


def _load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Load an AnySat checkpoint and return its encoder state dictionary."""
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint: Any = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:  # PyTorch < 2.0 has no ``weights_only`` argument.
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except Exception as exc:
        # Some older research checkpoints contain metadata that cannot be
        # deserialised with ``weights_only=True``. Only use this for a trusted
        # local file because regular pickle loading can execute code.
        print(f"[AnySat] weights_only load failed ({exc}); retrying trusted checkpoint.")
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    # Training scripts commonly wrap the actual tensor mapping in one of these
    # fields. Unwrap repeatedly because some frameworks nest them.
    while isinstance(checkpoint, dict):
        nested = next(
            (checkpoint[key] for key in ("state_dict", "model", "encoder") if key in checkpoint),
            None,
        )
        if not isinstance(nested, dict):
            break
        checkpoint = nested

    if not isinstance(checkpoint, dict):
        raise ValueError(f"AnySat checkpoint must contain a state dictionary: {checkpoint_path}")
    return checkpoint


def _compatible_state_dict(
    checkpoint_state: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Keep only checkpoint tensors whose key and shape match this encoder."""
    compatible: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "encoder.", "backbone.")
    for key, value in checkpoint_state.items():
        if not isinstance(value, torch.Tensor):
            continue

        candidates = [key]
        normalized = key
        # Remove nested DDP/model wrappers, e.g. module.encoder.blocks.0.*.
        while True:
            prefix = next((item for item in prefixes if normalized.startswith(item)), None)
            if prefix is None:
                break
            normalized = normalized[len(prefix):]
            candidates.append(normalized)

        target_key = next((candidate for candidate in candidates if candidate in model_state), None)
        if target_key is not None and model_state[target_key].shape == value.shape:
            compatible[target_key] = value
    return compatible


class AnySatBackbone(nn.Module):
    """Minimal AnySat wrapper for single-image dense prediction.

    This wrapper configures AnySat in release mode with a single modality
    (`aerial`) so it can accept generic (B, C, H, W) tensors.
    """

    def __init__(
        self,
        in_channels: int,
        model_size: str = "base",
        embed_dim: Optional[int] = None,
        depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        patch_size: int = 10,
        patch_scale: int = 1,
        spatial_resolution: Optional[float] = None,
        flash_attn: bool = False,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if model_size not in {"tiny", "small", "base"}:
            raise ValueError(f"Unsupported AnySat model_size: {model_size}")

        dim = 768 if model_size == "base" else (512 if model_size == "small" else 256)
        dpth = 6 if model_size == "base" else (4 if model_size == "small" else 2)
        heads = 12 if model_size == "base" else (8 if model_size == "small" else 4)

        dim = int(embed_dim) if embed_dim is not None else dim
        dpth = int(depth) if depth is not None else dpth
        heads = int(num_heads) if num_heads is not None else heads
        # ``PatchMLPMulti`` produces one local token for each ``patch_size``
        # pixels at patch_scale=1.  AnySat's release positional encoder uses
        # ``scale * 10 / input_res`` to derive that local-token grid.  Hence
        # input_res must be patch_size (10 by default), not 1.0; using 1.0
        # makes it create a 10x10 position grid for a single token.
        spatial_resolution = (
            float(spatial_resolution) if spatial_resolution is not None else float(patch_size)
        )

        patch_embeddings_mod = importlib.import_module("models.networks.encoder.utils.patch_embeddings")
        transformer_mod = importlib.import_module("models.networks.encoder.Transformer")
        any_multi_mod = importlib.import_module("models.networks.encoder.Any_multi")

        PatchMLPMulti = getattr(patch_embeddings_mod, "PatchMLPMulti")
        TransformerMulti = getattr(transformer_mod, "TransformerMulti")
        AnyModule = getattr(any_multi_mod, "AnyModule")

        projectors = {
            "aerial": PatchMLPMulti(
                in_chans=int(in_channels),
                patch_size=int(patch_size),
                embed_dim=dim,
                bias=False,
                resolution=1.0,
                mlp=[dim, dim * 2, dim],
            )
        }

        spatial_encoder = TransformerMulti(
            embed_dim=dim,
            depth=dpth,
            num_heads=heads,
            mlp_ratio=4.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            modalities={"all": ["aerial"]},
            scales={},
            input_res={"aerial": spatial_resolution},
        )

        self.encoder = AnyModule(
            projectors=projectors,
            spatial_encoder=spatial_encoder,
            modalities={"all": ["aerial"]},
            num_patches={},
            embed_dim=dim,
            depth=dpth,
            num_heads=heads,
            mlp_ratio=4.0,
            class_token=True,
            pre_norm=False,
            drop_rate=0.0,
            patch_drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            scales={},
            flash_attn=bool(flash_attn),
            release=True,
        )

        self.embed_dim = dim
        self.model_size = model_size
        self.input_patch_size = int(patch_scale) * 10
        # Expose which parameters actually came from the checkpoint.  This is
        # important for non-standard channel counts: a shape-mismatched input
        # projector is deliberately skipped and must not later be mistaken for
        # a pretrained parameter when the backbone is frozen.
        self.pretrained_parameter_names: set[str] = set()

        resolved_checkpoint_path = _resolve_local_checkpoint_path(checkpoint_path, model_size)
        if resolved_checkpoint_path is not None:
            checkpoint_state = _load_checkpoint_state(resolved_checkpoint_path)
            compatible_state = _compatible_state_dict(checkpoint_state, self.encoder.state_dict())
            if not compatible_state:
                raise RuntimeError(
                    f"No compatible AnySat encoder weights found in: {resolved_checkpoint_path}"
                )
            load_result = self.encoder.load_state_dict(compatible_state, strict=False)
            self.pretrained_parameter_names = set(compatible_state)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            skipped = len(checkpoint_state) - len(compatible_state)
            print(
                f"[AnySat] Loaded {len(compatible_state)}/{len(checkpoint_state)} tensors "
                f"from {resolved_checkpoint_path} (skipped {skipped})."
            )
            if missing:
                print(f"[AnySat] Missing keys: {missing}")
            if unexpected:
                print(f"[AnySat] Unexpected keys: {unexpected}")
        elif pretrained:
            print(
                "[AnySat] pretrained_backbone=True but no local checkpoint was found. "
                "Set checkpoint_path to a local .pth file to avoid Hugging Face downloads; "
                "using random initialization instead."
            )

    def freeze_pretrained_encoder(self) -> list[str]:
        """Freeze the encoder while keeping a random input projector trainable.

        AnySat's released ``aerial`` projector has four input channels.  For a
        dataset such as L4S (14 channels), its convolution cannot be restored
        from the checkpoint because the shapes differ.  Freezing that random
        layer makes linear probing operate on an arbitrary, fixed projection.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        projector = self.encoder.projector_aerial
        projector_prefix = "projector_aerial."
        projector_names = {
            name for name, _ in self.encoder.named_parameters()
            if name.startswith(projector_prefix)
        }
        missing_projector_names = sorted(
            projector_names - self.pretrained_parameter_names
        )
        if missing_projector_names:
            # Train the complete projector rather than only the missing first
            # convolution so its pretrained downstream layers can adapt to the
            # new 14-channel input distribution.
            for parameter in projector.parameters():
                parameter.requires_grad = True

        return missing_projector_names

    def forward(self, x: torch.Tensor, output: str = "patch") -> torch.Tensor:
        inputs = {"aerial": x}
        return self.encoder.forward_release(inputs, scale=self.input_patch_size // 10, output=output)
