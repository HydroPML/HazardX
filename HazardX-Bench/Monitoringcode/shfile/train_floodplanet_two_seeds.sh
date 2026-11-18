#!/usr/bin/env bash
# Train and test all FloodPlanet configs using the shared single-GPU runner.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export CONFIG_ROOT="${PROJECT_ROOT}/configs/flood/floodplanet"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/floodplanet}"
export TABLE_DIR="${TABLE_DIR:-${OUTPUT_ROOT}/tables}"
export SUMMARY_SCRIPT="${PROJECT_ROOT}/scripts/summarize_floodplanet_results.py"
export SUMMARY_FILE="${TABLE_DIR}/floodplanet_results.xlsx"

exec bash "${PROJECT_ROOT}/shfile/train_floga_sen2_norm_two_seeds.sh" "$@"
