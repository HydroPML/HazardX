# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# Keep model imports lazy: SAM2Mamba has optional Triton dependencies that
# should not be required when only SAM2FPN is used.
__all__ = ["SAM2FPN", "SAM2Mamba"]


def __getattr__(name):
    if name == "SAM2FPN":
        from .SAM2FPN import SAM2FPN

        return SAM2FPN
    if name == "SAM2Mamba":
        from .SAM2Mamba import SAM2Mamba

        return SAM2Mamba
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
