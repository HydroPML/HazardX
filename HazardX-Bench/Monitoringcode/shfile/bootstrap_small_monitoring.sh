#!/usr/bin/env bash
# Evaluate existing main experiments only; resume complete bootstrap results.
set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$PROJECT_ROOT" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
SKYSENSEPP_PYTHON_BIN="${SKYSENSEPP_PYTHON_BIN:-/root/anaconda3/envs/disaster/bin/python}"
RESULT_ROOT="${PROJECT_ROOT}/output/bootstrap_results"
failed=0
found=0
skipped=0
scheduled=0
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$RESULT_ROOT"
    printf 'dataset\texperiment\tpython\texit_code\n' > "$RESULT_ROOT/bootstrap_failures.tsv"
fi
for dataset in s2wcd floodplanet landref; do
    for experiment in "${PROJECT_ROOT}/output/${dataset}"/*; do
        [[ -f "$experiment/config.yaml" && -f "$experiment/best.pth" ]] || continue
        found=$((found + 1))
        experiment_name="$(basename -- "$experiment")"
        result_dir="${RESULT_ROOT}/${dataset}/${experiment_name}"
        # Skip only complete outputs matching the requested settings and current checkpoint/config.
        if [[ "${FORCE:-0}" != 1 ]] && "$PYTHON_BIN" - "$result_dir" "$experiment" "${BOOTSTRAP_REPS:-10000}" "${BOOTSTRAP_SEED:-2026}" <<'PY'
import json
import sys
from pathlib import Path
out, exp = map(Path, sys.argv[1:3])
try:
    files = [out / name for name in ('test_bootstrap.json', 'test_bootstrap.csv',
             'test_tile_confusion_matrices.npy', 'test_bootstrap_distributions.npz')]
    data = json.loads(files[0].read_text())
    fresh = min(p.stat().st_mtime for p in files) >= max(
        (exp / 'best.pth').stat().st_mtime, (exp / 'config.yaml').stat().st_mtime)
    complete = {r['metric'] for r in data['metrics']} == {'mIoU', 'CSI', 'F1'}
    sys.exit(0 if fresh and complete and data['repetitions'] == int(sys.argv[3])
             and data['seed'] == int(sys.argv[4]) else 1)
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
PY
        then
            skipped=$((skipped + 1))
            continue
        fi
        experiment_python="$PYTHON_BIN"
        if [[ "$experiment_name" == *skysensepp* ]]; then
            experiment_python="$SKYSENSEPP_PYTHON_BIN"
        fi
        scheduled=$((scheduled + 1))
        echo "[$dataset] $experiment_name | python=$experiment_python"
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            continue
        fi
        "$experiment_python" test.py --exp_path "$experiment" \
            --bootstrap "${BOOTSTRAP_REPS:-10000}" --bootstrap-seed "${BOOTSTRAP_SEED:-2026}" \
            --bootstrap-output-dir "$result_dir"
        status=$?
        if (( status != 0 )); then
            failed=$((failed + 1))
            printf '%s\t%s\t%s\t%s\n' "$dataset" "$experiment_name" "$experiment_python" "$status" >> "$RESULT_ROOT/bootstrap_failures.tsv"
        fi
    done
done
# Evaluate only external seed43/44 checkpoints; reuse complete bootstrap outputs.
EXTERNAL_PROJECT_ROOT="${EXTERNAL_PROJECT_ROOT:-/root/workcsc/diasterother}"
PYTHON_BIN="${EXTERNAL_PYTHON_BIN:-/root/anaconda3/envs/disasterother/bin/python}" \
    RESULT_ROOT="$RESULT_ROOT" bash "$EXTERNAL_PROJECT_ROOT/bootstrap_monitoring.sh" || failed=$((failed + 1))
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON_BIN" scripts/summarize_bootstrap_results.py || failed=$((failed + 1))
    "$PYTHON_BIN" scripts/export_bootstrap_seed_tables.py || failed=$((failed + 1))
    "$PYTHON_BIN" scripts/plot_bootstrap_uncertainty.py || failed=$((failed + 1))
fi
echo "Experiments: $found; skipped: $skipped; scheduled: $scheduled; failures: $failed"
(( found > 0 && failed == 0 ))
