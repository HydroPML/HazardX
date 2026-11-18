#!/usr/bin/env bash
# A1--A3 only. Existing A4 experiments are intentionally not scheduled.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
START_FROM="${START_FROM:-}"
started=0
[[ -z "${START_FROM}" ]] && started=1
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/l4s/terrain_ablation}"
read -r -a seeds <<< "${SEEDS:-42 43 44}"
read -r -a models <<< "${MODELS:-unet galileo_nano}"
read -r -a variants <<< "${VARIANTS:-A1 A2 A3}"
run() {
    printf '%q ' "$@"; printf '\n'
    if [[ "${DRY_RUN}" != 1 ]]; then "$@"; fi
}
for model in "${models[@]}"; do
    case "${model}" in unet|galileo_nano) ;; *) echo "Unknown model: ${model}" >&2; exit 1 ;; esac
    for variant in "${variants[@]}"; do
        case "${variant}" in A1|A2|A3) ;; *) echo 'Only A1--A3 are supported; reuse the existing A4 experiment.' >&2; exit 1 ;; esac
        config="configs/landslide/l4s/terrain_ablation/seg_${model}_${variant}.yaml"
        [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
        for seed in "${seeds[@]}"; do
            exp_path="${OUTPUT_ROOT}/seg_${model}_${variant}_seed${seed}"
            exp_name="seg_${model}_${variant}_seed${seed}"
            if [[ "${started}" == 0 ]]; then
                [[ "${exp_name}" == "${START_FROM}" ]] || continue
                started=1
            fi
            resume_args=()
            if [[ -f "${exp_path}/latest.pth" ]]; then
                resume_args=(--resume "${exp_path}/latest.pth")
            fi
            run env CUDA_VISIBLE_DEVICES="${GPU_ID}" NO_ALBUMENTATIONS_UPDATE=1 "${PYTHON_BIN}" train.py \
                --config "${config}" "${resume_args[@]}" --override seed "${seed}" output_dir "${exp_path}"
            run env CUDA_VISIBLE_DEVICES="${GPU_ID}" NO_ALBUMENTATIONS_UPDATE=1 "${PYTHON_BIN}" test.py --exp_path "${exp_path}"
        done
    done
done
[[ "${started}" == 1 ]] || { echo "START_FROM experiment not found: ${START_FROM}" >&2; exit 1; }
