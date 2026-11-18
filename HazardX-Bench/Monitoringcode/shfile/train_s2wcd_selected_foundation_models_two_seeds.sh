#!/usr/bin/env bash
# Train SkySense++, Clay v1.5 Large, and Galileo Nano sequentially on one GPU.
# Each seed is tested immediately after successful training; tables are refreshed
# after every successful test by the shared S2WCD runner.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export CONFIG_NAMES="cd_skysensepp.yaml cd_galileo_nano.yaml"
export SEEDS="${SEEDS:-43 44}"
export GPU_ID="${GPU_ID:-0}"

exec bash "${PROJECT_ROOT}/shfile/train_s2wcd_sen2_norm_two_seeds.sh" "$@"
 