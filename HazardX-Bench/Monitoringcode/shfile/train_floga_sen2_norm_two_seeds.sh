#!/usr/bin/env bash
# Train and test every FLOGA Sentinel-2 model on one GPU with seeds 43 and 44.
# Examples:
#   bash shfile/train_floga_sen2_norm_two_seeds.sh
#   GPU_ID=0 SEEDS="43 44" bash shfile/train_floga_sen2_norm_two_seeds.sh
#   START_CONFIG=cd_sam2_fpn_t.yaml bash shfile/train_floga_sen2_norm_two_seeds.sh
#   DRY_RUN=1 bash shfile/train_floga_sen2_norm_two_seeds.sh
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
DEFAULT_FLOGA_CONFIG_ROOT="${PROJECT_ROOT}/configs/wildfire/floga_sen2_norm"
CONFIG_ROOT="${CONFIG_ROOT:-${DEFAULT_FLOGA_CONFIG_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/floga}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/batch_logs}"
TABLE_DIR="${TABLE_DIR:-${OUTPUT_ROOT}/tables}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${PROJECT_ROOT}/scripts/summarize_floga_results.py}"
SUMMARY_FILE="${SUMMARY_FILE:-${TABLE_DIR}/floga_results.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
read -r -a seeds <<< "${SEEDS:-43 44}"

# The shared runner is also called by dataset-specific wrapper scripts. Apply
# FLOGA's dataset defaults only when this script is invoked for FLOGA itself.
if [[ ${CONFIG_ROOT} == "${DEFAULT_FLOGA_CONFIG_ROOT}" ]]; then
    DATASET_ROOT="${DATASET_ROOT:-/root/workcsc/datasets/wildfire/floga}"
    TRAIN_LIST="${TRAIN_LIST:-patch256/train.txt}"
    VAL_LIST="${VAL_LIST:-patch256/val.txt}"
    TEST_LIST="${TEST_LIST:-patch256/test.txt}"
fi

if [[ ${#seeds[@]} -lt 1 ]]; then
    echo 'SEEDS must contain at least one non-negative integer, for example: SEEDS="44".' >&2
    exit 2
fi
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    if [[ ! ${seed} =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}; expected a non-negative integer." >&2
        exit 2
    fi
    if [[ -n ${seen_seeds[${seed}]:-} ]]; then
        echo "Duplicate seed: ${seed}." >&2
        exit 2
    fi
    seen_seeds[${seed}]=1
done
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
}

shopt -s nullglob
if [[ -n ${CONFIG_NAMES:-} ]]; then
    read -r -a requested_configs <<< "${CONFIG_NAMES}"
    configs=()
    for config_name in "${requested_configs[@]}"; do
        config_path="${CONFIG_ROOT}/${config_name}"
        if [[ ! -f ${config_path} ]]; then
            echo "Requested config does not exist: ${config_path}" >&2
            exit 2
        fi
        configs+=("${config_path}")
    done
else
    configs=("${CONFIG_ROOT}"/*.yaml "${CONFIG_ROOT}"/*.yml)
fi
if (( ${#configs[@]} == 0 )); then
    echo "No YAML configs found in ${CONFIG_ROOT}" >&2
    exit 2
fi

# Optionally resume the first seed at the requested config. Later seeds still
# run the complete config list, which is important when seed 43 was interrupted
# but seed 44 has not started. A stem such as cd_sam2_fpn_t is accepted as well.
first_seed_configs=("${configs[@]}")
if [[ -n ${START_CONFIG:-} ]]; then
    start_name="$(basename -- "${START_CONFIG}")"
    if [[ ${start_name} != *.yaml && ${start_name} != *.yml ]]; then
        start_name="${start_name}.yaml"
    fi
    first_seed_configs=()
    found_start=0
    for config in "${configs[@]}"; do
        if [[ $(basename -- "${config}") == "${start_name}" ]]; then
            found_start=1
        fi
        if (( found_start )); then
            first_seed_configs+=("${config}")
        fi
    done
    if (( ! found_start )); then
        echo "START_CONFIG was not found in ${CONFIG_ROOT}: ${start_name}" >&2
        exit 2
    fi
    echo "First seed starts at ${start_name}; later seeds run all configs."
fi

# Optional dataset override used by wrappers whose data may live elsewhere.
if [[ -n ${DATASET_ROOT:-} ]]; then
    TRAIN_LIST="${TRAIN_LIST:-train.txt}"
    VAL_LIST="${VAL_LIST:-validation.txt}"
    TEST_LIST="${TEST_LIST:-test.txt}"
    if [[ ${DRY_RUN} != 1 ]]; then
        for split_file in "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}"; do
            if [[ ! -f ${DATASET_ROOT}/${split_file} ]]; then
                echo "Required dataset split is missing: ${DATASET_ROOT}/${split_file}" >&2
                exit 2
            fi
        done
    fi
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
experiment_count=$(( ${#first_seed_configs[@]} + (${#seeds[@]} - 1) * ${#configs[@]} ))
printf 'Found %d configs; running %d experiments sequentially on GPU %s.\n' \
    "${#configs[@]}" "${experiment_count}" "${GPU_ID}"

for seed in "${seeds[@]}"; do
    seed_configs=("${configs[@]}")
    if [[ ${seed} == "${seeds[0]}" ]]; then
        seed_configs=("${first_seed_configs[@]}")
    fi
    for config in "${seed_configs[@]}"; do
        config_name="$(basename -- "${config}")"
        model_name="${config_name%.*}"
        output_base="${OUTPUT_ROOT}/${model_name}"
        output_dir="${output_base}"
        if [[ ${seed} != 42 ]]; then
            output_dir="${output_base}_seed${seed}"
        fi

        mode="train"
        train_cmd=("${PYTHON_BIN}" train.py --config "${config}" --override seed "${seed}" output_dir "${output_base}")
        if [[ -n ${DATASET_ROOT:-} ]]; then
            train_cmd+=(
                dataset.train.dataset_path "${DATASET_ROOT}"
                dataset.train.data_list_path "${DATASET_ROOT}/${TRAIN_LIST}"
                dataset.val.dataset_path "${DATASET_ROOT}"
                dataset.val.data_list_path "${DATASET_ROOT}/${VAL_LIST}"
                dataset.test.dataset_path "${DATASET_ROOT}"
                dataset.test.data_list_path "${DATASET_ROOT}/${TEST_LIST}"
            )
        fi
        if [[ -f ${output_dir}/best.pth ]]; then
            mode="skip-train"
        elif [[ -f ${output_dir}/latest.pth ]]; then
            mode="resume"
            train_cmd+=(--resume "${output_dir}/latest.pth")
        fi
        test_cmd=("${PYTHON_BIN}" test.py --exp_path "${output_dir}" --checkpoint best.pth)
        summary_cmd=()
        if [[ -n ${SUMMARY_SCRIPT:-} ]]; then
            summary_cmd=("${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --output-root "${OUTPUT_ROOT}"
                --table-dir "${TABLE_DIR}" --seeds "${seeds[@]}")
        fi
        train_log="${batch_log_dir}/${model_name}_seed${seed}_train.log"
        test_log="${batch_log_dir}/${model_name}_seed${seed}_test.log"

        printf '\n[%s] model=%s seed=%s output=%s\n' "${mode}" "${model_name}" "${seed}" "${output_dir}"
        if [[ ${mode} != skip-train ]]; then
            printf 'Train: CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${GPU_ID}" "${seed}"
            printf '%q ' "${train_cmd[@]}"; printf '\n'
        else
            echo "Train: skipped because ${output_dir}/best.pth exists."
        fi
        printf 'Test:  CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${GPU_ID}" "${seed}"
        printf '%q ' "${test_cmd[@]}"; printf '\n'
        if (( ${#summary_cmd[@]} > 0 )); then
            printf 'Table: '; printf '%q ' "${summary_cmd[@]}"; printf '\n'
        fi
        if [[ ${DRY_RUN} == 1 ]]; then
            continue
        fi

        code=0
        if [[ ${mode} != skip-train ]]; then
            CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONHASHSEED="${seed}" "${train_cmd[@]}" 2>&1 | tee "${train_log}"
            statuses=("${PIPESTATUS[@]}")
            code=${statuses[0]}
            if (( code == 0 && statuses[1] != 0 )); then code=${statuses[1]}; fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${config_name}" "${seed}" "${code}" "${output_dir}" "${train_log}" "${mode}" >> "${summary}"
        fi

        if (( code == 0 )); then
            CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONHASHSEED="${seed}" "${test_cmd[@]}" 2>&1 | tee "${test_log}"
            statuses=("${PIPESTATUS[@]}")
            code=${statuses[0]}
            if (( code == 0 && statuses[1] != 0 )); then code=${statuses[1]}; fi
            printf '%s\t%s\t%s\t%s\t%s\ttest\n' \
                "${config_name}" "${seed}" "${code}" "${output_dir}" "${test_log}" >> "${summary}"
        fi

        if (( code == 0 )); then
            if (( ${#summary_cmd[@]} > 0 )); then
                "${summary_cmd[@]}"
                code=$?
                printf '%s\t%s\t%s\t%s\t%s\ttable\n' \
                    "${config_name}" "${seed}" "${code}" "${output_dir}" \
                    "${SUMMARY_FILE:-${TABLE_DIR}}" >> "${summary}"
            fi
        fi

        if (( code == 0 )); then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            echo "[failed] ${model_name}, seed=${seed}, exit=${code}; logs: ${batch_log_dir}" >&2
            if (( code == 130 || code == 143 )); then exit "${code}"; fi
            if [[ ${STOP_ON_ERROR} == 1 ]]; then exit "${code}"; fi
        fi
    done
done

if [[ ${DRY_RUN} == 1 ]]; then
    echo 'Dry run complete; no training or testing was started.'
    exit 0
fi

printf 'Finished: %d succeeded, %d failed. Summary: %s\n' "${success}" "${failed}" "${summary}"
(( failed == 0 ))
