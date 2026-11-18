#!/usr/bin/env bash
# Use a separate output root so the shared runner cannot skip training or
# resume the checkpoints trained with the old /10000 input scaling.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_NAMES="seg_spectralgpt.yaml"
export SEEDS="${SEEDS:-43 44}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/floodplanet_spectralgpt_scale255}"
export PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/disaster/bin/python}"
exec bash "${PROJECT_ROOT}/shfile/train_floodplanet_two_seeds.sh" "$@"
