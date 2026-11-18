from __future__ import annotations

import os
import sys
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from torch import nn
from einops import rearrange

from gphyt.model.transformer.attention_FNO import MLP, StochasticDepth
from gphyt.model.transformer.LiteFNOConditionedSliceAttention2d import (
    LiteGlobalFNO2d,
    _infer_state_dim,
)


class ResidualConditionedTransolverSliceAttention2d(nn.Module):
    """Transolver-style slice attention with residual conditioned slice correction."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int = 64,
        state_dim: Optional[int] = None,
        state_reduction: int = 4,
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
            hidden_dim, hidden_dim, local_kernel, padding=padding, groups=hidden_dim
        )
        self.slice_project = nn.Conv2d(hidden_dim, self.state_dim, 1)
        self.value_project = nn.Conv2d(hidden_dim, self.state_dim, 1)
        self.local_project = nn.Conv2d(hidden_dim, self.state_dim, 1)
        self.global_fno = LiteGlobalFNO2d(
            hidden_dim=hidden_dim,
            state_dim=self.state_dim,
            modes_height=fno_modes_height,
            modes_width=fno_modes_width,
            dropout=fno_dropout,
        )

        self.to_slice = nn.Linear(self.state_head_dim, self.slice_num, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.gamma_local = nn.Parameter(torch.tensor(float(init_local_gamma)))
        self.gamma_global = nn.Parameter(torch.tensor(float(init_global_gamma)))

        self.slice_attn_norm = nn.LayerNorm(self.state_head_dim)
        self.slice_token_q = nn.Linear(self.state_head_dim, self.state_head_dim, bias=False)
        self.slice_token_k = nn.Linear(self.state_head_dim, self.state_head_dim, bias=False)
        self.slice_token_v = nn.Linear(self.state_head_dim, self.state_head_dim, bias=False)
        self.slice_scale = self.state_head_dim ** -0.5
        self.slice_attn_dropout = nn.Dropout(dropout)
        self.slice_token_out = nn.Sequential(
            nn.Linear(self.state_head_dim, self.state_head_dim),
            nn.Dropout(dropout),
        )
        self.out_project = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.Dropout(dropout))

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

    def _slice_tokens(
        self,
        features: torch.Tensor,
        slice_weights: torch.Tensor,
        slice_norm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bhnd,bhnm->bhmd", features, slice_weights) / slice_norm[..., None]

    def _slice_self_attention(self, slice_token: torch.Tensor) -> torch.Tensor:
        z = self.slice_attn_norm(slice_token)
        q = self.slice_token_q(z)
        k = self.slice_token_k(z)
        v = self.slice_token_v(z)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.slice_scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.slice_attn_dropout(attn)
        out = torch.matmul(attn, v)
        return self.slice_token_out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "ResidualConditionedTransolverSliceAttention2d expects input shape (B, T, H, W, C)"
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

        slice_logits = self.to_slice(slice_features)
        slice_weights = torch.softmax(
            slice_logits / torch.clamp(self.temperature, min=0.01, max=1.0),
            dim=-1,
        )
        slice_norm = slice_weights.sum(dim=2).clamp_min(1e-5)

        conditioned_values = (
            value_features
            + self.gamma_local * local_features
            + self.gamma_global * global_features
        )
        base_slice_token = self._slice_tokens(value_features, slice_weights, slice_norm)
        cond_slice_token = self._slice_tokens(conditioned_values, slice_weights, slice_norm)
        delta_slice_token = self._slice_self_attention(cond_slice_token)
        out_slice_token = base_slice_token + delta_slice_token

        spatial_update = torch.einsum("bhmd,bhnm->bhnd", out_slice_token, slice_weights)
        spatial_update = rearrange(
            spatial_update,
            "(b t) heads (h w) d -> b t h w (heads d)",
            b=b, t=t, h=h, w=w, heads=self.num_heads, d=self.state_head_dim,
        )
        spatial_update = self.out_project(spatial_update)

        out = x + spatial_update if self.residual else spatial_update
        if self.use_ffn:
            out = out + self.ffn(self.ffn_norm(out))
        return out


class ResidualConditionedTransolverSliceAttentionBlock2d(nn.Module):
    """Transformer-style wrapper used by Flood GPhyT models."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        slice_num: int = 64,
        state_dim: Optional[int] = None,
        state_reduction: int = 4,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        fno_modes_height: int = 8,
        fno_modes_width: int = 8,
        fno_dropout: float = 0.0,
        local_kernel: int = 3,
        init_local_gamma: float = 0.5,
        init_global_gamma: float = 0.1,
        **unused_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = ResidualConditionedTransolverSliceAttention2d(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            slice_num=slice_num,
            state_dim=state_dim,
            state_reduction=state_reduction,
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


if __name__ == "__main__":
    B, T, H, W, C = 2, 3, 32, 32, 128
    x = torch.randn(B, T, H, W, C)
    model = ResidualConditionedTransolverSliceAttention2d(hidden_dim=C, num_heads=4, slice_num=64)
    y = model(x)
    assert y.shape == x.shape
    block = ResidualConditionedTransolverSliceAttentionBlock2d(hidden_dim=C, mlp_dim=256, num_heads=4, slice_num=64)
    y = block(x)
    assert y.shape == x.shape
    print("passed", y.shape)
