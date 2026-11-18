#!/usr/bin/env bash
# Run every LANDREF change-detection config with two random seeds.
#
# Examples:
#   bash shfile/train_landref_cd_two_seeds.sh
#   SEEDS="43 44" GPU_ID=0 bash shfile/train_landref_cd_two_seeds.sh
#   DRY_RUN=1 bash shfile/train_landref_cd_two_seeds.sh
#
# Optional environment variables:
#   SEEDS, GPU_ID, PYTHON_BIN, OUTPUT_ROOT, LOG_ROOT, TABLE_DIR, DRY_RUN, STOP_ON_ERROR
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
CONFIG_ROOT="${PROJECT_ROOT}/configs/landslide/landref_cd"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/workcsc/HazardX_Monitoring/output/landref}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/batch_logs}"
TABLE_DIR="${TABLE_DIR:-${PROJECT_ROOT}/output/tables}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
read -r -a seeds <<< "${SEEDS:-43 44}"

if [[ ${#seeds[@]} -ne 2 || ! ${seeds[0]} =~ ^[0-9]+$ || ! ${seeds[1]} =~ ^[0-9]+$ ]]; then
    echo 'SEEDS must contain exactly two non-negative integers, for example: SEEDS="43 44".' >&2
    exit 2
fi
if [[ ${seeds[0]} == "${seeds[1]}" ]]; then
    echo 'The two seeds must be different.' >&2
    exit 2
fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
}

shopt -s nullglob
configs=("${CONFIG_ROOT}"/*.yaml "${CONFIG_ROOT}"/*.yml)
if (( ${#configs[@]} == 0 )); then
    echo "No YAML configs found in ${CONFIG_ROOT}" >&2
    exit 2
fi

cd "${PROJECT_ROOT}" || exit 1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

run_id="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_$$"
batch_log_dir="${LOG_ROOT}/${run_id}"
if [[ ${DRY_RUN} != 1 ]]; then
    mkdir -p "${batch_log_dir}" || exit 1
    summary="${batch_log_dir}/summary.tsv"
    printf 'config\tseed\texit_code\toutput_dir\tlog\tstage\n' > "${summary}" || exit 1
fi

success=0
failed=0
printf 'Found %d configs; running %d experiments on GPU %s.\n' \
    "${#configs[@]}" "$(( ${#configs[@]} * 2 ))" "${GPU_ID}"

for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        config_name="$(basename -- "${config}")"
        model_name="${config_name%.*}"
        model_name="${model_name#cd_}"
        output_base="${OUTPUT_ROOT}/${model_name}"
        output_dir="${output_base}"
        if [[ ${seed} != 42 ]]; then
            output_dir="${output_base}_seed${seed}"
        fi

        mode="train"
        cmd=("${PYTHON_BIN}" train.py --config "${config}" --override seed "${seed}" output_dir "${output_base}")
        if [[ -f ${output_dir}/best.pth ]]; then
            mode="skip-train"
        elif [[ -f ${output_dir}/latest.pth ]]; then
            mode="resume"
            cmd+=(--resume "${output_dir}/latest.pth")
        fi

        log="${batch_log_dir}/${model_name}_seed${seed}.log"
        test_log="${batch_log_dir}/${model_name}_seed${seed}_test.log"
        test_cmd=("${PYTHON_BIN}" test.py --exp_path "${output_dir}" --checkpoint best.pth)
        table_cmd=("${PYTHON_BIN}" scripts/summarize_landref_cd_results.py
            --output-root "${OUTPUT_ROOT}" --table-dir "${TABLE_DIR}" --seeds "${seeds[@]}")
        printf '[%s] model=%s seed=%s output=%s\n' "${mode}" "${model_name}" "${seed}" "${output_dir}"
        if [[ ${mode} != skip-train ]]; then
            printf 'Train: CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${GPU_ID}" "${seed}"
            printf '%q ' "${cmd[@]}"; printf '\n'
        else
            echo "Train: skipped because ${output_dir}/best.pth exists."
        fi
        printf 'Test:  CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${GPU_ID}" "${seed}"
        printf '%q ' "${test_cmd[@]}"; printf '\n'
        printf 'Table: '; printf '%q ' "${table_cmd[@]}"; printf '\n'
        if [[ ${DRY_RUN} == 1 ]]; then
            continue
        fi

        code=0
        if [[ ${mode} != skip-train ]]; then
            CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONHASHSEED="${seed}" "${cmd[@]}" 2>&1 | tee "${log}"
            pipeline_status=("${PIPESTATUS[@]}")
            code=${pipeline_status[0]}
            if (( code == 0 && pipeline_status[1] != 0 )); then
                code=${pipeline_status[1]}
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${config_name}" "${seed}" "${code}" "${output_dir}" "${log}" "${mode}" >> "${summary}"
        fi

        if (( code == 0 )); then
            CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONHASHSEED="${seed}" "${test_cmd[@]}" 2>&1 | tee "${test_log}"
            pipeline_status=("${PIPESTATUS[@]}")
            code=${pipeline_status[0]}
            if (( code == 0 && pipeline_status[1] != 0 )); then
                code=${pipeline_status[1]}
            fi
            printf '%s\t%s\t%s\t%s\t%s\ttest\n' \
                "${config_name}" "${seed}" "${code}" "${output_dir}" "${test_log}" >> "${summary}"
        fi
        if (( code == 0 )); then
            "${table_cmd[@]}"
            code=$?
            printf '%s\t%s\t%s\t%s\t%s\ttable\n' \
                "${config_name}" "${seed}" "${code}" "${output_dir}" "${TABLE_DIR}/landref_cd_results.xlsx" >> "${summary}"
        fi

        if (( code != 0 )); then
            failed=$((failed + 1))
            echo "[failed] ${model_name}, seed=${seed}, exit=${code}; log: ${log}" >&2
            if (( code == 130 || code == 143 )); then
                exit "${code}"
            fi
            if [[ ${STOP_ON_ERROR} == 1 ]]; then
                exit "${code}"
            fi
        else
            success=$((success + 1))
        fi
    done
done

if [[ ${DRY_RUN} == 1 ]]; then
    echo 'Dry run complete; no training was started.'
    exit 0
fi

printf 'Finished: %d succeeded, %d failed. Summary: %s\n' "${success}" "${failed}" "${summary}"
(( failed == 0 ))
