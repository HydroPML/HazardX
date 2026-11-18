"""
Spatial attention with a 2D Fourier branch.

This module keeps the same public tensor contract as attention.py:
input/output tensors are shaped as (B, T, H, W, C).  The FNO branch is the
2D counterpart of the frequency branch used in StFT_3D.py: it lifts token
features, keeps low spatial Fourier modes, maps them back with an inverse FFT,
and adds the result to the spatial attention output.
"""

import inspect
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from gphyt.model.transformer.pos_encodings import RotaryPositionalEmbedding


def complex_parameter(*shape, scale=1.0):
    weight = scale * torch.randn(*shape, dtype=torch.cfloat)
    return nn.Parameter(torch.view_as_real(weight))


def as_complex(weight: torch.Tensor) -> torch.Tensor:
    if weight.is_complex():
        return weight
    return torch.view_as_complex(weight.contiguous())


try:
    from torchvision.ops import StochasticDepth
except ImportError:
    class StochasticDepth(nn.Module):
        def __init__(self, p=0.0, mode="row"):
            super().__init__()
            self.p = p
            self.mode = mode

        def forward(self, x):
            return x


class BatchFirstMultiheadAttention(nn.Module):
    """
    nn.MultiheadAttention wrapper that accepts (B, S, C) on old and new torch.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        has_batch_first = "batch_first" in inspect.signature(
            nn.MultiheadAttention.__init__
        ).parameters
        self.batch_first = has_batch_first
        kwargs = {}
        if has_batch_first:
            kwargs["batch_first"] = True
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            **kwargs,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
        if self.batch_first:
            return self.attention(q, k, v, **kwargs)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        out, weights = self.attention(q, k, v, **kwargs)
        return out.transpose(0, 1), weights


class AbstractAttention(nn.Module):
    """
    Abstract attention class.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pe = pe

        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.attention = BatchFirstMultiheadAttention(hidden_dim, num_heads, dropout)

    def _add_pos_embeddings(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        if self.pe is not None:
            q, k = self.pe(q, k)
        return q, k

    def _get_qkv(self, x: torch.Tensor) -> tuple:
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self._add_pos_embeddings(q, k)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpatioTemporalAttention(AbstractAttention):
    """
    Full attention over time, height, and width.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        q, k, v = self._get_qkv(x)
        q = rearrange(q, "b t h w c -> b (t h w) c")
        k = rearrange(k, "b t h w c -> b (t h w) c")
        v = rearrange(v, "b t h w c -> b (t h w) c")

        x, _ = self.attention(q, k, v)
        return rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)


class CausalSpatioTemporalAttention(AbstractAttention):
    """
    Full attention with a causal mask over time.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        time: int,
        height: int,
        width: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
        return_att: bool = False,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)
        self.return_att = return_att
        num_patches = time * height * width
        patches_per_timestep = height * width
        indices = torch.arange(num_patches)
        token_timesteps = indices // patches_per_timestep
        query_timesteps = token_timesteps.unsqueeze(1)
        key_timesteps = token_timesteps.unsqueeze(0)
        self.register_buffer("mask", key_timesteps > query_timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        q, k, v = self._get_qkv(x)
        q = rearrange(q, "b t h w c -> b (t h w) c")
        k = rearrange(k, "b t h w c -> b (t h w) c")
        v = rearrange(v, "b t h w c -> b (t h w) c")

        x, att_weights = self.attention(
            q, k, v, attn_mask=self.mask, need_weights=self.return_att
        )
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)
        if self.return_att:
            return x, att_weights
        return x


class SpatialAttention(AbstractAttention):
    """
    Spatial attention over height and width.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        q, k, v = self._get_qkv(x)
        q = rearrange(q, "b t h w c -> (b t) (h w) c")
        k = rearrange(k, "b t h w c -> (b t) (h w) c")
        v = rearrange(v, "b t h w c -> (b t) (h w) c")

        x, _ = self.attention(q, k, v)
        return rearrange(x, "(b t) (h w) c -> b t h w c", b=B, t=T, h=H, w=W)


class TemporalAttention(AbstractAttention):
    """
    Temporal attention over time.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        q, k, v = self._get_qkv(x)
        q = rearrange(q, "b t h w c -> (b h w) t c")
        k = rearrange(k, "b t h w c -> (b h w) t c")
        v = rearrange(v, "b t h w c -> (b h w) t c")

        x, _ = self.attention(q, k, v)
        return rearrange(x, "(b h w) t c -> b t h w c", b=B, h=H, w=W, t=T)


class MLP(nn.Module):
    """
    MLP with linear layers.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SpectralConv2d(nn.Module):
    """
    Low-mode 2D spectral convolution on a token grid.

    Input shape: (B, C, H, W)
    Output shape: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, modes_height: int, modes_width: int):
        super().__init__()
        if modes_height <= 0 or modes_width <= 0:
            raise ValueError("FNO modes must be positive integers")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights = complex_parameter(
            in_channels,
            out_channels,
            modes_height,
            modes_width,
            scale=scale,
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bihw,iohw->bohw", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        modes_h = min(self.modes_height, H)
        modes_w = min(self.modes_width, W // 2 + 1)

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes_h, :modes_w] = self.compl_mul2d(
            x_ft[:, :, :modes_h, :modes_w],
            as_complex(self.weights)[:, :, :modes_h, :modes_w],
        )
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))


class SpatialFNO2d(nn.Module):
    """
    2D FNO branch over the spatial patch-token grid for each time slice.

    The branch is intentionally parallel to spatial attention.  Time is kept as
    a batch-like axis here so Markov rollout supplies temporal coupling outside
    the block while this module performs spatial spectral mixing.
    """

    def __init__(
        self,
        hidden_dim: int,
        modes_height: int = 4,
        modes_width: int = 4,
        lift_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        lift_dim = lift_dim or hidden_dim
        self.lift = nn.Linear(hidden_dim, lift_dim)
        self.spectral = SpectralConv2d(lift_dim, lift_dim, modes_height, modes_width)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(lift_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        x = self.lift(x)
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = self.spectral(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T, h=H, w=W)
        return self.proj(x)


class SpatialAttentionFNO(AbstractAttention):
    """
    Spatial MHA plus a 2D FNO branch.

    Input shape: (B, T, H, W, C)
    Output shape: (B, T, H, W, C)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
        fno_modes_height: int = 12,
        fno_modes_width: int = 12,
        fno_lift_dim: Optional[int] = None,
        fno_dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, num_heads, dropout, pe)
        self.fno = SpatialFNO2d(
            hidden_dim=hidden_dim,
            modes_height=fno_modes_height,
            modes_width=fno_modes_width,
            lift_dim=fno_lift_dim,
            dropout=fno_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        q, k, v = self._get_qkv(x)

        q = rearrange(q, "b t h w c -> (b t) (h w) c")
        k = rearrange(k, "b t h w c -> (b t) (h w) c")
        v = rearrange(v, "b t h w c -> (b t) (h w) c")

        att, _ = self.attention(q, k, v)
        att = rearrange(att, "(b t) (h w) c -> b t h w c", b=B, t=T, h=H, w=W)
        return att + self.fno(x)


class AttentionBlock(nn.Module):
    """
    Attention block with optional spatial-FNO attention and MLP.

    Input shape: (B, T, H, W, C)
    Output shape: (B, T, H, W, C)
    """

    def __init__(
        self,
        att_type: str,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        time: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        pe: Optional[RotaryPositionalEmbedding] = None,
        fno_modes_height: int = 4,
        fno_modes_width: int = 4,
        fno_lift_dim: Optional[int] = None,
        fno_dropout: float = 0.0,
    ):
        super().__init__()
        if att_type == "full":
            if time == 1:
                self.attention = SpatialAttention(hidden_dim, num_heads, dropout, pe)
            else:
                self.attention = SpatioTemporalAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type == "spatial":
            self.attention = SpatialAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type in {"spatial_fno", "spatial+fno"}:
            self.attention = SpatialAttentionFNO(
                hidden_dim,
                num_heads,
                dropout,
                pe,
                fno_modes_height=fno_modes_height,
                fno_modes_width=fno_modes_width,
                fno_lift_dim=fno_lift_dim,
                fno_dropout=fno_dropout,
            )
        elif att_type == "temporal":
            self.attention = TemporalAttention(hidden_dim, num_heads, dropout, pe)
        elif att_type == "full_causal":
            if time is None or height is None or width is None:
                raise ValueError(
                    "time, height, and width must be provided for causal attention"
                )
            self.attention = CausalSpatioTemporalAttention(
                hidden_dim, num_heads, time, height, width, dropout, pe
            )
        else:
            raise ValueError(f"Invalid attention type: {att_type}")

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)
        self.sd = StochasticDepth(stochastic_depth_rate, mode="row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1(input)
        att = self.sd(self.attention(x)) + input

        x = self.norm2(att)
        x = self.sd(self.mlp(x)) + att
        return x
