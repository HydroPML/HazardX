# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .attention import CausalSelfAttention, LinearKMaskedBias, SelfAttention
from .block import CausalSelfAttentionBlock, SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
from .layer_scale import LayerScale
from .patch_embed import PatchEmbed
from .rms_norm import RMSNorm
from .rope_position_encoding import RopePositionEmbedding


def __getattr__(name):
    # FP8 support requires recent PyTorch compiler APIs. Keep it optional so
    # standard DINOv3 models can run on older PyTorch versions.
    if name == "convert_linears_to_fp8":
        from .fp8_linear import convert_linears_to_fp8

        return convert_linears_to_fp8
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
