"""Lightweight local-global physical slice attention for 2D flood features.

The design follows the LinearNO view of physics attention:

    k = softmax(to_k(x), dim=spatial_tokens)
    q = softmax(to_q(x), dim=physical_states)
    state = k^T v
    out = q state

Compared with ``attention_fno_conditioned_slice.py``, this version reduces
parameters by:

* using low-dimensional physical states instead of full hidden channels;
* replacing dense spectral convolution with depthwise low-mode FNO;
* replacing full slice-token MHA with a LinearNO-style state mixer;
* using depthwise local convolution for local flood-front detail.
"""

from typing import Optional

import torch
from torch import nn
from einops import rearrange

from gphyt.model.transformer.attention_FNO import (
    MLP,
    StochasticDepth,
    as_complex,
    complex_parameter,
)


def _infer_state_dim(hidden_dim: int, num_heads: int, state_reduction: int) -> int:
    state_reduction = max(1, int(state_reduction))
    raw_dim = max(num_heads, hidden_dim // state_reduction)
    state_dim = (raw_dim // num_heads) * num_heads
    return max(num_heads, state_dim)


class DepthwiseSpectralConv2d(nn.Module):
    """Depthwise low-mode spectral convolution.

    Dense FNO uses ``C_in * C_out * modes_h * modes_w`` complex weights. This
    depthwise version uses only ``C * modes_h * modes_w`` complex weights and
    relies on cheap pointwise projections around it for channel mixing.
    """

    def __init__(self, channels: int, modes_height: int, modes_width: int):
        super().__init__()
        if modes_height <= 0 or modes_width <= 0:
            raise ValueError("FNO modes must be positive integers")
        self.channels = channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        scale = 1.0 / max(1, channels)
        self.weights = complex_parameter(
            channels,
            modes_height,
            modes_width,
            scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        modes_h = min(self.modes_height, h)
        modes_w = min(self.modes_width, w // 2 + 1)

        input_dtype = x.dtype
        x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1))
        out_ft = torch.zeros(
            b,
            c,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes_h, :modes_w] = (
            x_ft[:, :, :modes_h, :modes_w]
            * as_complex(self.weights)[None, :, :modes_h, :modes_w]
        )
        out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        return out.to(dtype=input_dtype)


class LiteGlobalFNO2d(nn.Module):
    """Low-parameter FNO global branch on the real spatial grid."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        modes_height: int,
        modes_width: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lift = nn.Linear(hidden_dim, state_dim)
        self.spectral = DepthwiseSpectralConv2d(
            state_dim, modes_height, modes_width
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        x = self.lift(x)
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = self.spectral(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t, h=h, w=w)
        return self.proj(x)


class LinearPhysicalStateMixer(nn.Module):
    """LinearNO-style mixer over physical state tokens.

    Input shape: ``(BT, heads, M, state_head_dim)``.
    """

    def __init__(
        self,
        num_heads: int,
        state_head_dim: int,
        key_ratio: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        key_ratio = max(1, int(key_ratio))
        self.temperature_q = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.temperature_k = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.to_q = nn.Linear(state_head_dim, key_ratio, bias=False)
        self.to_k = nn.Linear(state_head_dim, key_ratio, bias=False)
        self.to_v = nn.Linear(state_head_dim, state_head_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(state_head_dim, state_head_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = torch.softmax(
            q / torch.clamp(self.temperature_q, min=0.01, max=1.0), dim=-1
        )
        k = torch.softmax(
            k / torch.clamp(self.temperature_k, min=0.01, max=1.0), dim=-2
        )

        kv = torch.einsum("bhmr,bhmd->bhrd", k, v)
        out = torch.einsum("bhmr,bhrd->bhmd", q, kv)
        return self.to_out(out)


class LiteFNOConditionedSliceAttention2d(nn.Module):
    """Parameter-efficient local/global conditioned physical attention.

    Input and output shape are both ``(B, T, H, W, C)``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int = 64,
        state_dim: Optional[int] = None,
        state_reduction: int = 4,
        key_ratio: int = 8,
        dropout: float = 0.0,
        fno_modes_height: int = 8,
        fno_modes_width: int = 8,
        fno_dropout: float = 0.0,
        local_kernel: int = 3,
        init_local_gamma: float = 0.5,
        init_global_gamma: float = 0.1,
        use_ffn: bool = True,
        ffn_ratio: float = 2.0,
        residual: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if state_dim is None or state_dim <= 0:
            state_dim = _infer_state_dim(hidden_dim, num_heads, state_reduction)
        if state_dim % num_heads != 0:
            raise ValueError("state_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.slice_num = int(slice_num)
        self.state_dim = int(state_dim)
        self.state_head_dim = self.state_dim // num_heads
        self.residual = residual
        self.use_ffn = use_ffn

        padding = local_kernel // 2
        self.local_depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            local_kernel,
            padding=padding,
            groups=hidden_dim,
        )
        self.local_project = nn.Conv2d(hidden_dim, self.state_dim, 1)
        self.value_project = nn.Conv2d(hidden_dim, self.state_dim, 1)
        self.slice_project = nn.Conv2d(hidden_dim, self.state_dim, 1)

        self.global_fno = LiteGlobalFNO2d(
            hidden_dim=hidden_dim,
            state_dim=self.state_dim,
            modes_height=fno_modes_height,
            modes_width=fno_modes_width,
            dropout=fno_dropout,
        )

        self.to_slice_q = nn.Linear(self.state_head_dim, self.slice_num, bias=False)
        self.to_slice_k = nn.Linear(self.state_head_dim, self.slice_num, bias=False)
        self.temperature_q = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.temperature_k = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        self.gamma_local = nn.Parameter(torch.tensor(float(init_local_gamma)))
        self.gamma_global = nn.Parameter(torch.tensor(float(init_global_gamma)))
        self.state_norm = nn.LayerNorm(self.state_head_dim)
        self.state_mixer = LinearPhysicalStateMixer(
            num_heads=num_heads,
            state_head_dim=self.state_head_dim,
            key_ratio=key_ratio,
            dropout=dropout,
        )
        self.out_project = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        if use_ffn:
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = MLP(hidden_dim, int(hidden_dim * ffn_ratio), dropout)
        else:
            self.ffn_norm = None
            self.ffn = None

    def _to_heads_from_grid(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "bt (heads d) h w -> bt heads (h w) d",
            heads=self.num_heads,
            d=self.state_head_dim,
        )

    def _to_heads_from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b t h w (heads d) -> (b t) heads (h w) d",
            heads=self.num_heads,
            d=self.state_head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "LiteFNOConditionedSliceAttention2d expects input shape "
                "(B, T, H, W, C)"
            )
        b, t, h, w, c = x.shape
        if c != self.hidden_dim:
            raise ValueError("Input hidden dimension does not match hidden_dim")

        x_grid = rearrange(x, "b t h w c -> (b t) c h w")
        local_grid = self.local_depthwise(x_grid)
        slice_base = x_grid + local_grid

        slice_features = self._to_heads_from_grid(self.slice_project(slice_base))
        value_features = self._to_heads_from_grid(self.value_project(x_grid))
        local_features = self._to_heads_from_grid(self.local_project(local_grid))
        global_features = self._to_heads_from_tokens(self.global_fno(x))

        slice_q_logits = self.to_slice_q(slice_features)
        slice_k_logits = self.to_slice_k(slice_features)
        slice_q = torch.softmax(
            slice_q_logits / torch.clamp(self.temperature_q, min=0.01, max=1.0),
            dim=-1,
        )
        slice_k = torch.softmax(
            slice_k_logits / torch.clamp(self.temperature_k, min=0.01, max=1.0),
            dim=2,
        )

        state = torch.einsum("bhnm,bhnd->bhmd", slice_k, value_features)
        local_state = torch.einsum("bhnm,bhnd->bhmd", slice_k, local_features)
        global_state = torch.einsum("bhnm,bhnd->bhmd", slice_k, global_features)

        conditioned_state = (
            state
            + self.gamma_local * local_state
            + self.gamma_global * global_state
        )
        mixed_state = state + self.state_mixer(self.state_norm(conditioned_state))

        spatial_update = torch.einsum("bhmd,bhnm->bhnd", mixed_state, slice_q)
        spatial_update = rearrange(
            spatial_update,
            "(b t) heads (h w) d -> b t h w (heads d)",
            b=b,
            t=t,
            h=h,
            w=w,
            heads=self.num_heads,
            d=self.state_head_dim,
        )
        spatial_update = self.out_project(spatial_update)

        out = x + spatial_update if self.residual else spatial_update
        if self.use_ffn:
            out = out + self.ffn(self.ffn_norm(out))
        return out


class LiteFNOConditionedSliceAttentionBlock2d(nn.Module):
    """Transformer-style wrapper used by Flood GPhyT models."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        slice_num: int = 64,
        state_dim: Optional[int] = None,
        state_reduction: int = 4,
        key_ratio: int = 8,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        fno_modes_height: int = 8,
        fno_modes_width: int = 8,
        fno_dropout: float = 0.0,
        local_kernel: int = 3,
        init_local_gamma: float = 0.5,
        init_global_gamma: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = LiteFNOConditionedSliceAttention2d(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            slice_num=slice_num,
            state_dim=state_dim,
            state_reduction=state_reduction,
            key_ratio=key_ratio,
            dropout=dropout,
            fno_modes_height=fno_modes_height,
            fno_modes_width=fno_modes_width,
            fno_dropout=fno_dropout,
            local_kernel=local_kernel,
            init_local_gamma=init_local_gamma,
            init_global_gamma=init_global_gamma,
            use_ffn=False,
            residual=False,
        )
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)
        self.sd = StochasticDepth(stochastic_depth_rate, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sd(self.attention(self.norm1(x)))
        x = x + self.sd(self.mlp(self.norm2(x)))
        return x
