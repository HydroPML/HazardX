#!/usr/bin/env bash
# Example: DRY_RUN=1 bash shfile/train_floga_unet_stream_ablation.sh
# Default: seed 42, single-stream and independent dual-stream.
# All three seeds: SEEDS="42 43 44" bash shfile/train_floga_unet_stream_ablation.sh
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/floga/stream_ablation}"
read -r -a seeds <<< "${SEEDS:-42}"
read -r -a variants <<< "${VARIANTS:-single_stream dual_stream}"

run() {
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" != 1 ]]; then "$@"; fi
}

for variant in "${variants[@]}"; do
    case "${variant}" in
        single_stream|dual_stream|dual_stream_shared) ;;
        *) echo "Unknown variant: ${variant}" >&2; exit 1 ;;
    esac
    config="configs/wildfire/floga_sen2_norm/stream_ablation/cd_unet_${variant}.yaml"
    for seed in "${seeds[@]}"; do
        exp_path="${OUTPUT_ROOT}/cd_unet_${variant}_seed${seed}"
        run env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
            --config "${config}" --override seed "${seed}" output_dir "${exp_path}"
        run env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" test.py --exp_path "${exp_path}"
    done
done
