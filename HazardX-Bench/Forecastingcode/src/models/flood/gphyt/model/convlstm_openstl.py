"""OpenSTL-style ConvLSTM-Scratch adapted to one-step FloodCastBench Markov I/O.

Source:
  https://github.com/chengtan9907/OpenSTL
  commit eecf8a3078f0a178dbc7b28723da20f94ce36985
  openstl/modules/convlstm_modules.py and openstl/models/convlstm_model.py

The benchmark-facing ``forward`` always creates zero hidden/cell states.  The
explicit ``forward_step`` method exists only for unit-testing state semantics;
the experiment runner never carries its returned state across rollout steps.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gphyt.model.spatial_adapter import (
    assert_flood_output,
    crop_to_original,
    minimal_pad_bottom_right,
    validate_flood_input,
)


State = List[Tuple[torch.Tensor, torch.Tensor]]


class ConvLSTMCell(nn.Module):
    """Standard, no-peephole ConvLSTM cell used by the cited OpenSTL model."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ConvLSTM kernel size must be odd")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(
            self.in_channels,
            4 * self.hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.conv_h = nn.Conv2d(
            self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv_x(x) + self.conv_h(h)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMScratch(nn.Module):
    experiment_name = "ConvLSTM-Scratch"
    architecture_variant = "ConvLSTM-OpenSTL-Markov"
    temporal_input_steps = 1
    persistent_hidden_state = False
    patch_size = 2

    def __init__(
        self,
        hidden_channels: Sequence[int] = (128, 128, 128, 128),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        hidden_channels = tuple(int(v) for v in hidden_channels)
        if hidden_channels != (128, 128, 128, 128):
            raise ValueError(
                "ConvLSTM-Scratch must use four OpenSTL layers of 128 channels"
            )
        self.hidden_channels = hidden_channels
        cells = []
        for layer, channels in enumerate(hidden_channels):
            in_channels = 3 * self.patch_size**2 if layer == 0 else hidden_channels[layer - 1]
            cells.append(ConvLSTMCell(in_channels, channels, kernel_size))
        self.cells = nn.ModuleList(cells)
        self.recovery = nn.Conv2d(hidden_channels[-1], self.patch_size**2, 1)
        self.pixel_shuffle = nn.PixelShuffle(self.patch_size)

    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        """[B,1,3,H,W] -> [B,1,12,H/2,W/2]."""
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[2] != 3:
            raise AssertionError(f"expected [B,1,3,H,W], got {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        if height % 2 or width % 2:
            raise AssertionError("patchify requires even H/W after minimal padding")
        patches = F.pixel_unshuffle(x.reshape(batch * steps, channels, height, width), 2)
        return patches.reshape(batch, steps, 12, height // 2, width // 2)

    def zero_state(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> State:
        return [
            (
                torch.zeros(batch, channels, height, width, device=device, dtype=dtype),
                torch.zeros(batch, channels, height, width, device=device, dtype=dtype),
            )
            for channels in self.hidden_channels
        ]

    def _validate_state(
        self, state: State, batch: int, height: int, width: int
    ) -> None:
        if len(state) != len(self.cells):
            raise AssertionError("ConvLSTM state does not contain four layers")
        for layer, ((h, c), channels) in enumerate(zip(state, self.hidden_channels)):
            expected = (batch, channels, height, width)
            if tuple(h.shape) != expected or tuple(c.shape) != expected:
                raise AssertionError(
                    f"layer {layer} state shape mismatch: h={tuple(h.shape)}, "
                    f"c={tuple(c.shape)}, expected={expected}"
                )

    def forward_step(
        self, x: torch.Tensor, state: Optional[State] = None
    ) -> Tuple[torch.Tensor, State]:
        batch, _, _ = validate_flood_input(x)
        padded, pad = minimal_pad_bottom_right(x, 2)
        sequence = self.patchify(padded.unsqueeze(1))
        patch_input = sequence[:, 0]
        patch_height, patch_width = (int(v) for v in patch_input.shape[-2:])
        if state is None:
            state = self.zero_state(
                batch,
                patch_height,
                patch_width,
                device=patch_input.device,
                dtype=patch_input.dtype,
            )
        else:
            self._validate_state(state, batch, patch_height, patch_width)

        new_state: State = []
        hidden = patch_input
        for cell, layer_state in zip(self.cells, state):
            hidden, cell_state = cell(hidden, layer_state)
            new_state.append((hidden, cell_state))
        patch_output = self.recovery(hidden)
        if patch_output.shape[1] != 4:
            raise AssertionError("ConvLSTM recovery must produce four patch channels")
        prediction = self.pixel_shuffle(patch_output)
        prediction = crop_to_original(prediction, pad)
        return assert_flood_output(prediction, x), new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _discarded_state = self.forward_step(x, state=None)
        return prediction
