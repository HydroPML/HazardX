"""Physical-coordinate positional encodings for one-step Flood GPhyT models.

This module is intentionally separate from the existing GPhyT implementation.
It keeps the tokenizer, transformer block, MLP, detokenizer, and attention
parameterization unchanged while replacing only positional encoding behavior.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from gphyt.model.tokenizer.tokenizer import Detokenizer, Tokenizer
from gphyt.model.transformer.derivatives import FiniteDifference
from gphyt.model.transformer.num_integration import Euler, Heun, RK4


POSITION_ENCODINGS = ("physical_rope", "fourier_ape")
SPACE_REF_METERS = 1000.0
ROPE_BASE = 10000.0
FOURIER_NUM_FREQUENCIES = 16


def _as_batch_spacing(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert a scalar or length-B spacing to a validated FP32 tensor."""
    if value is None:
        raise ValueError(
            f"{name} is required for physical positional encoding; "
            "set the domain resolution in metres"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(item is None for item in value):
            raise ValueError(f"{name} contains a missing resolution: {value}")
    spacing = torch.as_tensor(value, dtype=torch.float32, device=device)
    if spacing.ndim == 0:
        spacing = spacing.expand(batch_size)
    elif spacing.ndim == 1 and spacing.numel() == batch_size:
        spacing = spacing.reshape(batch_size)
    else:
        raise ValueError(
            f"{name} must be scalar or [B], got shape {tuple(spacing.shape)} "
            f"for B={batch_size}"
        )
    if not torch.isfinite(spacing).all() or bool((spacing <= 0).any()):
        raise ValueError(f"{name} must contain finite positive metres, got {value}")
    return spacing


def physical_coordinates(
    length: int,
    spacing: Any,
    *,
    batch_size: int,
    device: torch.device,
    space_ref: float = SPACE_REF_METERS,
) -> torch.Tensor:
    """Return ``arange(length) * spacing / space_ref`` with shape ``[B,L]``."""
    if length < 1:
        raise ValueError(f"Coordinate length must be positive, got {length}")
    if not math.isfinite(space_ref) or space_ref <= 0:
        raise ValueError(f"space_ref must be finite and positive, got {space_ref}")
    spacing_tensor = _as_batch_spacing(
        spacing,
        batch_size=batch_size,
        device=device,
        name="spacing",
    )
    indices = torch.arange(length, dtype=torch.float32, device=device)
    return indices.unsqueeze(0) * spacing_tensor.unsqueeze(1) / float(space_ref)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Physical2DRotaryEmbedding(nn.Module):
    """Apply physical 2D RoPE to already split attention-head Q/K tensors.

    Q/K must have shape ``[B,T,num_heads,H,W,head_dim]``. Half of each head is
    assigned to x and half to y; no channels are assigned to time.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        base: float = ROPE_BASE,
        space_ref: float = SPACE_REF_METERS,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                "Physical 2D RoPE requires head_dim divisible by 4 so the x/y "
                f"halves each contain rotary pairs, got {head_dim}"
            )
        if base <= 0 or not math.isfinite(base):
            raise ValueError(f"RoPE base must be finite and positive, got {base}")
        if space_ref <= 0 or not math.isfinite(space_ref):
            raise ValueError(
                f"space_ref must be finite and positive, got {space_ref}"
            )
        self.head_dim = int(head_dim)
        self.axis_dim = self.head_dim // 2
        self.base = float(base)
        self.space_ref = float(space_ref)
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.axis_dim, 2, dtype=torch.float32)
                / self.axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def _axis_cos_sin(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, ...]:
        angles = coordinates.unsqueeze(-1) * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.ndim != 6:
            raise ValueError(
                "Physical RoPE expects matching Q/K [B,T,heads,H,W,head_dim], "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        batch_size, _, _, height, width, head_dim = q.shape
        if head_dim != self.head_dim:
            raise ValueError(
                f"Q/K head_dim {head_dim} does not match initialized "
                f"head_dim {self.head_dim}"
            )
        x = physical_coordinates(
            width,
            dx,
            batch_size=batch_size,
            device=q.device,
            space_ref=self.space_ref,
        )
        y = physical_coordinates(
            height,
            dy,
            batch_size=batch_size,
            device=q.device,
            space_ref=self.space_ref,
        )
        x_cos, x_sin = self._axis_cos_sin(x)
        y_cos, y_sin = self._axis_cos_sin(y)
        x_cos = x_cos[:, None, None, None, :, :].to(dtype=q.dtype)
        x_sin = x_sin[:, None, None, None, :, :].to(dtype=q.dtype)
        y_cos = y_cos[:, None, None, :, None, :].to(dtype=q.dtype)
        y_sin = y_sin[:, None, None, :, None, :].to(dtype=q.dtype)

        q_x, q_y = q.split(self.axis_dim, dim=-1)
        k_x, k_y = k.split(self.axis_dim, dim=-1)
        q_x = q_x * x_cos + _rotate_half(q_x) * x_sin
        k_x = k_x * x_cos + _rotate_half(k_x) * x_sin
        q_y = q_y * y_cos + _rotate_half(q_y) * y_sin
        k_y = k_y * y_cos + _rotate_half(k_y) * y_sin
        return torch.cat((q_x, q_y), dim=-1), torch.cat((k_x, k_y), dim=-1)


class PhysicalFourierCoordinateAPE(nn.Module):
    """Dynamic physical Fourier-coordinate absolute token embedding."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_frequencies: int = FOURIER_NUM_FREQUENCIES,
        space_ref: float = SPACE_REF_METERS,
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if space_ref <= 0 or not math.isfinite(space_ref):
            raise ValueError(
                f"space_ref must be finite and positive, got {space_ref}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_frequencies = int(num_frequencies)
        self.space_ref = float(space_ref)
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(4 * num_frequencies, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
    ) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError(
                f"Fourier APE expects [B,T,H,W,C], got {tuple(tokens.shape)}"
            )
        batch_size, _, height, width, channels = tokens.shape
        if channels != self.hidden_dim:
            raise ValueError(
                f"Token width {channels} does not match hidden_dim {self.hidden_dim}"
            )
        x = physical_coordinates(
            width,
            dx,
            batch_size=batch_size,
            device=tokens.device,
            space_ref=self.space_ref,
        )
        y = physical_coordinates(
            height,
            dy,
            batch_size=batch_size,
            device=tokens.device,
            space_ref=self.space_ref,
        )
        x_angles = x[:, None, :, None] * self.frequencies[None, None, None, :]
        y_angles = y[:, :, None, None] * self.frequencies[None, None, None, :]
        x_angles = x_angles.expand(-1, height, -1, -1)
        y_angles = y_angles.expand(-1, -1, width, -1)
        features = torch.cat(
            (
                x_angles.sin(),
                x_angles.cos(),
                y_angles.sin(),
                y_angles.cos(),
            ),
            dim=-1,
        ).to(dtype=tokens.dtype)
        embedding = self.projection(features).unsqueeze(1)
        return tokens + embedding


class PhysicalSpatialAttention(nn.Module):
    """Original two-stage QKV attention with optional post-head physical RoPE.

    The existing model first applies ``to_qkv`` and then passes those three
    tensors through ``nn.MultiheadAttention``, which performs its own Q/K/V
    projections. This implementation preserves both parameter sets. It only
    expands the second operation so RoPE can be inserted after the actual
    projected Q/K tensors have been split into heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope: Optional[Physical2DRotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by heads {num_heads}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim // num_heads)
        self.rope = rope
        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            batch_first=True,
        )

    def _actual_projected_qkv(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = self.attention.in_proj_weight
        bias = self.attention.in_proj_bias
        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        if bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
        return (
            F.linear(q_input, q_weight, q_bias),
            F.linear(k_input, k_weight, k_bias),
            F.linear(v_input, v_weight, v_bias),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Attention expects [B,T,H,W,C], got {x.shape}")
        batch_size, time, height, width, hidden_dim = x.shape
        qkv = self.to_qkv(x)
        q_input, k_input, v_input = qkv.chunk(3, dim=-1)
        token_count = height * width
        bt = batch_size * time
        q_input = q_input.reshape(bt, token_count, hidden_dim)
        k_input = k_input.reshape(bt, token_count, hidden_dim)
        v_input = v_input.reshape(bt, token_count, hidden_dim)
        q, k, v = self._actual_projected_qkv(q_input, k_input, v_input)

        q = q.reshape(bt, token_count, self.num_heads, self.head_dim)
        k = k.reshape(bt, token_count, self.num_heads, self.head_dim)
        v = v.reshape(bt, token_count, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.rope is not None:
            q_grid = q.reshape(
                batch_size,
                time,
                self.num_heads,
                height,
                width,
                self.head_dim,
            )
            k_grid = k.reshape_as(q_grid)
            q_grid, k_grid = self.rope(q_grid, k_grid, dx=dx, dy=dy)
            q = q_grid.reshape(bt, self.num_heads, token_count, self.head_dim)
            k = k_grid.reshape(bt, self.num_heads, token_count, self.head_dim)

        weights = torch.matmul(q * (self.head_dim**-0.5), k.transpose(-2, -1))
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(
            weights,
            p=float(self.attention.dropout),
            training=self.training,
        )
        output = torch.matmul(weights, v)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(bt, token_count, hidden_dim)
        )
        output = self.attention.out_proj(output)
        return output.reshape(batch_size, time, height, width, hidden_dim)


class PhysicalPositionAttentionBlock(nn.Module):
    """The baseline pre-norm attention/MLP block with physical metadata."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float,
        stochastic_depth_rate: float,
        rope: Optional[Physical2DRotaryEmbedding],
    ) -> None:
        super().__init__()
        self.attention = PhysicalSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope=rope,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.sd = StochasticDepth(stochastic_depth_rate, mode="row")

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
    ) -> torch.Tensor:
        normalized = self.norm1(inputs)
        attended = self.sd(self.attention(normalized, dx=dx, dy=dy)) + inputs
        normalized = self.norm2(attended)
        return self.sd(self.mlp(normalized)) + attended


class PhysicalPositionPhysicsTransformer(nn.Module):
    """GPhyT backbone with exactly one physical positional encoding mode."""

    def __init__(
        self,
        *,
        num_fields: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: tuple[int, int, int],
        img_size: tuple[int, int, int],
        position_encoding: str,
        use_derivatives: bool = False,
        att_mode: str = "spatial",
        integrator: Optional[str] = None,
        tokenizer_mode: str = "linear",
        detokenizer_mode: str = "linear",
        tokenizer_overlap: int = 0,
        detokenizer_overlap: int = 0,
        tokenizer_net_channels: Optional[list[int]] = None,
        detokenizer_net_channels: Optional[list[int]] = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        space_ref: float = SPACE_REF_METERS,
    ) -> None:
        super().__init__()
        if position_encoding not in POSITION_ENCODINGS:
            raise ValueError(
                f"position_encoding must be exactly one of {POSITION_ENCODINGS}, "
                f"got {position_encoding!r}"
            )
        if att_mode not in {"spatial", "full"}:
            raise ValueError(
                "One-step physical baselines support spatial/full attention only"
            )
        if img_size[0] // patch_size[0] != 1:
            raise ValueError("Physical position baselines are Markov T=1 models")
        self.num_fields = int(num_fields)
        self.hidden_dim = int(hidden_dim)
        self.position_encoding = position_encoding
        self.patch_size = tuple(int(value) for value in patch_size)
        self.use_derivatives = bool(use_derivatives)
        num_input_channels = self.num_fields
        if self.use_derivatives:
            self.derivatives = FiniteDifference(
                num_channels=self.num_fields,
                filter_1d="2nd",
            )
            num_input_channels *= 4

        if integrator == "Euler":
            self.integrator = Euler()
        elif integrator == "RK4":
            self.integrator = RK4()
        elif integrator == "Heun":
            self.integrator = Heun()
        elif integrator is None:
            self.integrator = None
        else:
            raise ValueError(f"Unknown integrator {integrator!r}")

        self.tokenizer = Tokenizer(
            patch_size=patch_size,
            in_channels=num_input_channels,
            dim_embed=hidden_dim,
            mode=tokenizer_mode,
            conv_net_channels=tokenizer_net_channels,
            overlap=tokenizer_overlap,
        )

        rope: Optional[Physical2DRotaryEmbedding]
        if position_encoding == "physical_rope":
            rope = Physical2DRotaryEmbedding(
                head_dim=hidden_dim // num_heads,
                space_ref=space_ref,
            )
            self.coordinate_embedding = None
        else:
            rope = None
            # Preserve the random stream used to initialize all baseline layers.
            # The Fourier MLP is positional-encoding-only extra capacity.
            rng_state = torch.get_rng_state()
            self.coordinate_embedding = PhysicalFourierCoordinateAPE(
                hidden_dim=hidden_dim,
                space_ref=space_ref,
            )
            torch.set_rng_state(rng_state)

        self.attention_blocks = nn.ModuleList(
            [
                PhysicalPositionAttentionBlock(
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    stochastic_depth_rate=stochastic_depth_rate,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.detokenizer = Detokenizer(
            patch_size=patch_size,
            dim_embed=hidden_dim,
            out_channels=self.num_fields,
            mode=detokenizer_mode,
            conv_net_channels=detokenizer_net_channels,
            overlap=detokenizer_overlap,
            img_size=img_size,
        )

    def differentiate(
        self,
        fields: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
    ) -> torch.Tensor:
        if self.use_derivatives:
            dt, dh, dw = self.derivatives(fields)
            fields = torch.cat((fields, dt, dh, dw), dim=-1)
        tokens = self.tokenizer(fields)
        # Token spacing is patch spacing in physical space. Therefore the
        # requested arange(W)*dx formula uses raw dx*patch_width at token level.
        token_dx = _as_batch_spacing(
            dx,
            batch_size=tokens.shape[0],
            device=tokens.device,
            name="dx",
        ) * self.patch_size[2]
        token_dy = _as_batch_spacing(
            dy,
            batch_size=tokens.shape[0],
            device=tokens.device,
            name="dy",
        ) * self.patch_size[1]
        if self.coordinate_embedding is not None:
            tokens = self.coordinate_embedding(tokens, dx=token_dx, dy=token_dy)
        for block in self.attention_blocks:
            tokens = block(tokens, dx=token_dx, dy=token_dy)
        return self.detokenizer(tokens)

    def forward(
        self,
        fields: torch.Tensor,
        *,
        dx: Any,
        dy: Any,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        del step_size
        if fields.ndim != 5 or fields.shape[1] != 1:
            raise ValueError(
                "Physical positional baselines require fields [B,1,H,W,C], "
                f"got {tuple(fields.shape)}"
            )
        if self.integrator is not None:
            raise RuntimeError(
                "The FloodCastBench direct-derivative baseline must disable the "
                "internal integrator"
            )
        output = self.differentiate(fields, dx=dx, dy=dy)
        return output[:, -1, ...].unsqueeze(1)


def build_physical_position_gphyt(
    model_config: Mapping[str, Any],
    *,
    hidden_dim: int,
    mlp_dim: int,
    num_heads: int,
    num_layers: int,
    conv_channels: Sequence[int],
) -> PhysicalPositionPhysicsTransformer:
    """Build the physical-position backbone from the baseline config schema."""
    transformer = model_config["transformer"]
    tokenizer = model_config["tokenizer"]
    return PhysicalPositionPhysicsTransformer(
        num_fields=transformer["input_channels"],
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        att_mode=transformer.get("att_mode", "spatial"),
        integrator=transformer.get("integrator"),
        position_encoding=transformer["position_encoding"],
        img_size=tuple(model_config["img_size"]),
        patch_size=tuple(transformer["patch_size"]),
        use_derivatives=transformer["use_derivatives"],
        tokenizer_mode=tokenizer["tokenizer_mode"],
        detokenizer_mode=tokenizer["detokenizer_mode"],
        tokenizer_overlap=tokenizer["tokenizer_overlap"],
        detokenizer_overlap=tokenizer["detokenizer_overlap"],
        tokenizer_net_channels=list(conv_channels),
        detokenizer_net_channels=list(conv_channels),
        dropout=transformer["dropout"],
        stochastic_depth_rate=transformer["stochastic_depth_rate"],
        space_ref=float(transformer.get("space_ref", SPACE_REF_METERS)),
    )
