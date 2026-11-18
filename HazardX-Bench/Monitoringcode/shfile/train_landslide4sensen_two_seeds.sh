#!/usr/bin/env bash
# Train L4S only with two parallel workers: GPU 0 / seed 43 and GPU 1 / seed 44 by default.
# Usage:
#   bash shfile/train_landslide4sensen_two_seeds.sh
#   DRY_RUN=1 bash shfile/train_landslide4sensen_two_seeds.sh
#   SEEDS="43 2026" GPU_IDS="0 1" PYTHON_BIN=python bash shfile/train_landslide4sensen_two_seeds.sh
# Existing best.pth: test only. Otherwise resume latest.pth or train from scratch, then test.
# Optional: GPU_IDS, SEEDS, OUTPUT_ROOT, LOG_ROOT, STOP_ON_ERROR=1.
# Trainer automatically adds _seed<N> to output_dir when seed != 42.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/output/landslide}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/batch_logs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_ROOT}/l4s_summary}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
read -r -a seeds <<< "${SEEDS:-43 44}"
if [[ ${#seeds[@]} -ne 2 || ! ${seeds[0]} =~ ^[0-9]+$ || ! ${seeds[1]} =~ ^[0-9]+$ ]]; then
    echo 'SEEDS must contain exactly two non-negative integers, e.g. "43 44".' >&2
    exit 2
fi
# Normalize decimal seeds so leading zeros cannot cause output-path collisions.
for i in 0 1; do
    if [[ ${#seeds[i]} -gt 10 ]]; then echo 'Seed must be in [0, 4294967295].' >&2; exit 2; fi
    seeds[i]=$((10#${seeds[i]}))
    if (( seeds[i] > 4294967295 )); then echo 'Seed must be in [0, 4294967295].' >&2; exit 2; fi
done
if [[ ${seeds[0]} == "${seeds[1]}" ]]; then echo 'The two seeds must be different.' >&2; exit 2; fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
cd "${PROJECT_ROOT}" || exit 1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
shopt -s nullglob
configs=()
for dataset in l4s; do
    files=("${PROJECT_ROOT}/configs/landslide/${dataset}/"*.yaml "${PROJECT_ROOT}/configs/landslide/${dataset}/"*.yml)
    if (( ${#files[@]} == 0 )); then echo "No configs found for ${dataset}" >&2; exit 2; fi
    configs+=("${files[@]}")
done
read -r -a gpus <<< "${GPU_IDS:-0 1}"
if [[ ${#gpus[@]} -ne 2 || ${gpus[0]} == "${gpus[1]}" ]]; then
    echo 'GPU_IDS must contain two distinct GPU IDs, e.g. "0 1".' >&2
    exit 2
fi

run_worker() {
    local index=$1 seed=${seeds[$1]} gpu=${gpus[$1]}
    local config dataset model output_base actual_output log test_log code stage
    local success=0 failed=0
    local -a cmd test_cmd statuses
    local summary="${BATCH_LOG_DIR:-}/worker_${index}.tsv"
    if [[ ${DRY_RUN} != 1 ]]; then
        printf 'dataset\tconfig\tseed\tgpu\texit_code\toutput_dir\tlog\tstage\n' > "${summary}" || { echo "Cannot write summary (check disk space): ${summary}" >&2; return 74; }
    fi
    for config in "${configs[@]}"; do
        dataset="$(basename -- "$(dirname -- "${config}")")"
        model="$(basename -- "${config}")"; model="${model%.*}"
        output_base="${OUTPUT_ROOT}/${dataset}/${model}"
        actual_output="${output_base}"
        if [[ ${seed} != 42 ]]; then actual_output="${output_base}_seed${seed}"; fi
        stage=train
        cmd=("${PYTHON_BIN}" train.py --config "${config}" --override seed "${seed}" output_dir "${output_base}")
        if [[ -f ${actual_output}/best.pth ]]; then
            stage=test
        elif [[ -f ${actual_output}/latest.pth ]]; then
            stage=resume
            cmd+=(--resume "${actual_output}/latest.pth")
        fi
        test_cmd=("${PYTHON_BIN}" test.py --exp_path "${actual_output}" --checkpoint best.pth --force)
        printf '\n[GPU %s / %s / %s / seed=%s] action=%s output=%s\n' "${gpu}" "${dataset}" "${model}" "${seed}" "${stage}" "${actual_output}"
        if [[ ${stage} != test ]]; then
            printf 'Train command: CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${gpu}" "${seed}"
            printf '%q ' "${cmd[@]}"; printf '\n'
        else
            echo 'best.pth exists: skip training and test it directly.'
        fi
        printf 'Test command: CUDA_VISIBLE_DEVICES=%q PYTHONHASHSEED=%q ' "${gpu}" "${seed}"
        printf '%q ' "${test_cmd[@]}"; printf '\n'
        if [[ ${DRY_RUN} == 1 ]]; then continue; fi
        code=0
        if [[ ${stage} != test ]]; then
            log="${BATCH_LOG_DIR}/${dataset}_${model}_seed${seed}.log"
            CUDA_VISIBLE_DEVICES="${gpu}" PYTHONHASHSEED="${seed}" "${cmd[@]}" 2>&1 | tee "${log}"
            statuses=("${PIPESTATUS[@]}"); code=${statuses[0]}
            if (( code == 0 && statuses[1] != 0 )); then code=${statuses[1]}; fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${model}" "${seed}" "${gpu}" "${code}" "${actual_output}" "${log}" "${stage}" >> "${summary}" || { echo "Cannot update summary (check disk space): ${summary}" >&2; return 74; }
        fi
        if (( code == 0 )); then
            stage=test
            test_log="${BATCH_LOG_DIR}/${dataset}_${model}_seed${seed}_test.log"
            printf '\nTesting best checkpoint: %s\n' "${actual_output}"
            if [[ ! -f ${actual_output}/best.pth || ! -f ${actual_output}/config.yaml ]]; then
                echo "Cannot test: best.pth or config.yaml is missing in ${actual_output}" | tee "${test_log}"
                code=1
            else
                CUDA_VISIBLE_DEVICES="${gpu}" PYTHONHASHSEED="${seed}" "${test_cmd[@]}" 2>&1 | tee "${test_log}"
                statuses=("${PIPESTATUS[@]}"); code=${statuses[0]}
                if (( code == 0 && statuses[1] != 0 )); then code=${statuses[1]}; fi
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\ttest\n' "${dataset}" "${model}" "${seed}" "${gpu}" "${code}" "${actual_output}" "${test_log}" >> "${summary}" || { echo "Cannot update summary (check disk space): ${summary}" >&2; return 74; }
        fi
        if (( code == 0 )); then
            stage=summary
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/summarize_l4s_results.py \
                --config-root "${PROJECT_ROOT}/configs/landslide/l4s" \
                --exp-root "${OUTPUT_ROOT}/l4s" --output-dir "${SUMMARY_ROOT}" \
                --seeds 42 "${seeds[@]}"
            code=$?
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tsummary\n' "${dataset}" "${model}" "${seed}" "${gpu}" "${code}" "${actual_output}" "${SUMMARY_ROOT}/l4s_results.xlsx" >> "${summary}" || { echo "Cannot update summary (check disk space): ${summary}" >&2; return 74; }
        fi
        if (( code == 0 )); then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            echo "${stage} failed on GPU ${gpu} (exit ${code}); see ${BATCH_LOG_DIR}" >&2
            if (( code == 130 || code == 143 )); then return "${code}"; fi
            if [[ ${STOP_ON_ERROR} == 1 ]]; then return "${code}"; fi
        fi
    done
    printf '\nGPU %s / seed %s finished: %d succeeded, %d failed.\n' "${gpu}" "${seed}" "${success}" "${failed}"
    (( failed == 0 ))
}

# Internal worker mode: each child owns a separate process group, including training and tee.
if [[ ${1:-} == --worker ]]; then
    [[ ${2:-} == 0 || ${2:-} == 1 ]] && [[ -n ${BATCH_LOG_DIR:-} ]] || exit 2
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # Keep the worker in its own process group while mirroring logs to the terminal.
    # The logger belongs to this worker, so the parent's wait -n still tracks only workers.
    exec 3>&1 4>&2
    exec > >(tee "${BATCH_LOG_DIR}/worker_${2}.log" | awk -v prefix="[GPU ${gpus[$2]} / seed ${seeds[$2]}] " '{ print prefix $0; fflush(); }') 2>&1
    logger_pid=$!
    run_worker "$2"
    worker_code=$?
    exec 1>&3 2>&4 3>&- 4>&-
    if wait "${logger_pid}"; then logger_code=0; else logger_code=$?; fi
    if (( worker_code == 0 && logger_code != 0 )); then worker_code=${logger_code}; fi
    exit "${worker_code}"
fi
if (( $# > 0 )); then echo "Unknown argument: $1" >&2; exit 2; fi
printf 'Configs: %d; total experiments: %d\n' "${#configs[@]}" "$(( ${#configs[@]} * 2 ))"
printf 'GPU %s -> seed %s; GPU %s -> seed %s\n' "${gpus[0]}" "${seeds[0]}" "${gpus[1]}" "${seeds[1]}"
if [[ ${DRY_RUN} == 1 ]]; then
    run_worker 0
    run_worker 1
    echo 'Dry run complete; no training started.'
    exit 0
fi
command -v setsid >/dev/null || { echo 'setsid is required for worker cleanup.' >&2; exit 2; }
run_id="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_$$"
export BATCH_LOG_DIR="${LOG_ROOT}/${run_id}"
mkdir -p "${BATCH_LOG_DIR}" || exit 1
if ! printf 'write check\n' > "${BATCH_LOG_DIR}/.write_check"; then
    echo 'Log directory is not writable; check free disk space.' >&2
    exit 74
fi
rm -f "${BATCH_LOG_DIR}/.write_check"
# Pass all selected settings to the child invocations, including shell-local defaults.
export PYTHON_BIN OUTPUT_ROOT LOG_ROOT SUMMARY_ROOT DRY_RUN STOP_ON_ERROR
export SEEDS="${seeds[*]}" GPU_IDS="${gpus[*]}"
echo "Batch logs: ${BATCH_LOG_DIR}"
pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; fi
    done
    for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'echo "Interrupted; stopping both GPU workers." >&2; exit 130' INT
trap 'echo "Terminated; stopping both GPU workers." >&2; exit 143' TERM
for index in 0 1; do
    setsid bash "${BASH_SOURCE[0]}" --worker "${index}" &
    pids+=("$!")
done
result=0
for index in 0 1; do
    if wait -n; then code=0; else code=$?; fi
    if (( code != 0 )); then
        result=${code}
        if [[ ${STOP_ON_ERROR} == 1 || ${code} == 130 || ${code} == 143 || ${code} == 74 ]]; then
            cleanup
            break
        fi
    fi
done
summary="${BATCH_LOG_DIR}/summary.tsv"
printf 'dataset\tconfig\tseed\tgpu\texit_code\toutput_dir\tlog\tstage\n' > "${summary}" || { echo "Cannot write summary: ${summary}" >&2; exit 74; }
for index in 0 1; do
    if [[ -f ${BATCH_LOG_DIR}/worker_${index}.tsv ]]; then
        tail -n +2 "${BATCH_LOG_DIR}/worker_${index}.tsv" >> "${summary}" || { echo "Cannot update summary: ${summary}" >&2; exit 74; }
    fi
done
echo "Both workers finished. Exit code: ${result}; summary: ${summary}"
exit "${result}"
