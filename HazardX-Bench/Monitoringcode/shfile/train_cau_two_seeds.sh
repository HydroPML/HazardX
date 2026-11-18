#!/usr/bin/env bash
# Train and test all CAU-Flood configs using the shared single-GPU runner.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export CONFIG_ROOT="${PROJECT_ROOT}/configs/flood/cau"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/cau}"
export DATASET_ROOT="${DATASET_ROOT:-/root/workcsc/datasets/flood/CAU_Flood}"
export TRAIN_LIST="${TRAIN_LIST:-train.txt}"
export VAL_LIST="${VAL_LIST:-validation.txt}"
export TEST_LIST="${TEST_LIST:-test.txt}"

exec bash "${PROJECT_ROOT}/shfile/train_floga_sen2_norm_two_seeds.sh" "$@"
