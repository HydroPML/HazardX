#!/usr/bin/env bash
# Preview: DRY_RUN=1 bash shfile/train_landref_modality_ablation.sh
# Default: Galileo nano B1--B4, seeds 42/43/44; resume latest.pth when present.
# To include U-Net: MODELS="unet galileo_nano" bash shfile/train_landref_modality_ablation.sh
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/landref/modality_ablation}"
read -r -a seeds <<< "${SEEDS:-42 43 44}"
read -r -a models <<< "${MODELS:-galileo_nano}"
read -r -a variants <<< "${VARIANTS:-B1 B2 B3 B4}"
run() {
    printf '%q ' "$@"; printf '\n'
    if [[ "${DRY_RUN}" != 1 ]]; then "$@"; fi
}
for model in "${models[@]}"; do
    case "${model}" in unet|galileo_nano) ;; *) echo "Unknown model: ${model}" >&2; exit 1 ;; esac
    for variant in "${variants[@]}"; do
        case "${variant}" in B1|B2|B3|B4) ;; *) echo "Unknown variant: ${variant}" >&2; exit 1 ;; esac
        config="configs/landslide/landref_cd/modality_ablation/cd_${model}_${variant}.yaml"
        [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
        for seed in "${seeds[@]}"; do
            exp_path="${OUTPUT_ROOT}/cd_${model}_${variant}_seed${seed}"
            if [[ -f "${exp_path}/latest.pth" ]]; then
                echo "Resume: ${model} ${variant} seed=${seed}"
                run env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
                    --config "${config}" --override seed "${seed}" output_dir "${exp_path}" \
                    --resume "${exp_path}/latest.pth"
            elif [[ -f "${exp_path}/best.pth" ]]; then
                echo "Only best.pth exists; preserve weights and evaluate: ${exp_path}"
            else
                echo "Start: ${model} ${variant} seed=${seed} (no checkpoint)"
                run env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
                    --config "${config}" --override seed "${seed}" output_dir "${exp_path}"
            fi
            run env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" test.py --exp_path "${exp_path}"
        done
    done
done
