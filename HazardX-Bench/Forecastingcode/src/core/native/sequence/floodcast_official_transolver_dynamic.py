"""Official Transolver Structured-Mesh-2D architecture on dynamic native grids.

The modules below follow the official THUML Transolver implementation used by
``PDE-Solving-StandardBenchmark/exp_ns.py`` and
``model/Transolver_Structured_Mesh_2D.py``.  The only implementation-level
adaptations are that ``height`` and ``width`` are read from each forward call
and the official unified-position tensor is cached per native grid/device.
The physical-slice attention, block order, residual paths, MLPs and parameter
initialization are otherwise unchanged.

Official source (MIT): https://github.com/thuml/Transolver
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class OfficialMLP(nn.Module):
    """MLP copied structurally from the official Transolver implementation."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ) -> None:
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(act)
        activation = ACTIVATION[act]
        self.n_layers = int(n_layers)
        self.res = bool(res)
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class OfficialPhysicsAttentionStructuredMesh2D(nn.Module):
    """Official structured-mesh physical-slice attention with runtime H/W."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        kernel: int = 3,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self, x: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        batch, nodes, channels = x.shape
        if nodes != int(height) * int(width):
            raise AssertionError(
                f"Transolver nodes={nodes} != H*W={height}*{width}"
            )
        image = (
            x.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        fx_mid = (
            self.in_project_fx(image)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch, nodes, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(image)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch, nodes, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        slice_weights = self.softmax(
            self.in_project_slice(x_mid)
            / torch.clamp(self.temperature, min=0.1, max=5.0)
        )
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[..., None].repeat(1, 1, 1, self.dim_head)
        )

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(
            q_slice_token, k_slice_token.transpose(-1, -2)
        ) * self.scale
        attention = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attention, v_slice_token)
        out_x = torch.einsum(
            "bhgc,bhng->bhnc", out_slice_token, slice_weights
        )
        out_x = out_x.permute(0, 2, 1, 3).reshape(
            batch, nodes, self.heads * self.dim_head
        )
        return self.to_out(out_x)


class OfficialTransolverBlock(nn.Module):
    """Official pre-norm Transolver encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ) -> None:
        super().__init__()
        self.last_layer = bool(last_layer)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = OfficialPhysicsAttentionStructuredMesh2D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = OfficialMLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, fx: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        fx = self.Attn(self.ln_1(fx), height=height, width=width) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class OfficialTransolverStructuredMesh2D(nn.Module):
    """Official NS Transolver with dynamic unified-position construction."""

    def __init__(
        self,
        space_dim: int = 2,
        n_layers: int = 8,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        fun_dim: int = 3,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = True,
    ) -> None:
        super().__init__()
        if n_hidden % n_head:
            raise ValueError("n_hidden must be divisible by n_head")
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)
        preprocess_dim = fun_dim + (self.ref * self.ref if unified_pos else space_dim)
        self.preprocess = OfficialMLP(
            preprocess_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            [
                OfficialTransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=layer == n_layers - 1,
                )
                for layer in range(n_layers)
            ]
        )
        self.placeholder = nn.Parameter(
            (1.0 / n_hidden) * torch.rand(n_hidden, dtype=torch.float32)
        )
        self._position_cache: Dict[
            Tuple[int, int, str, int, torch.dtype], torch.Tensor
        ] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def _unified_position(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (
            int(height),
            int(width),
            device.type,
            -1 if device.index is None else int(device.index),
            dtype,
        )
        cached = self._position_cache.get(key)
        if cached is not None:
            return cached
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.ref, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, self.ref, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        reference = torch.stack((ref_x, ref_y), dim=-1)
        position = torch.sqrt(
            torch.sum(
                (grid[:, :, None, None, :] - reference[None, None, :, :, :])
                ** 2,
                dim=-1,
            )
        ).reshape(1, height * width, self.ref * self.ref)
        self._position_cache[key] = position
        return position

    def forward(
        self,
        fields: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if fields.ndim != 3 or fields.shape[1] != height * width:
            raise AssertionError(
                f"fields must be [B,H*W,C], got {tuple(fields.shape)}"
            )
        if self.unified_pos:
            pos = self._unified_position(
                height=height,
                width=width,
                device=fields.device,
                dtype=fields.dtype,
            ).expand(fields.shape[0], -1, -1)
        else:
            raise ValueError("FloodCastBench official Transolver requires unified_pos")
        fx = self.preprocess(torch.cat((pos, fields), dim=-1))
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, height=height, width=width)
        return fx

