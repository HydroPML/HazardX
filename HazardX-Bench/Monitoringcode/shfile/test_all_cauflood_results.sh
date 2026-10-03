#!/bin/sh

set -u

PROJECT_ROOT="/root/workcsc/AnyDisasterMapping"
OUTPUT_ROOT="/root/autodl-tmp/output/cau"

cd "${PROJECT_ROOT}" || exit 1

failed=""
tested=0

for exp_path in "${OUTPUT_ROOT}"/*; do
  [ -d "${exp_path}" ] || continue

  exp_name=$(basename "${exp_path}")
  tested=$((tested + 1))
  echo "[$(date '+%F %T')] Starting test: ${exp_name}"

  if python test.py --exp_path "${exp_path}"; then
    echo "[$(date '+%F %T')] Finished test: ${exp_name}"
  else
    status=$?
    echo "[$(date '+%F %T')] Failed test: ${exp_name} (exit code ${status})" >&2
    failed="${failed} ${exp_name}:${status}"
  fi
done

if [ "${tested}" -eq 0 ]; then
  echo "No result directories found under ${OUTPUT_ROOT}." >&2
  exit 1
fi

if [ -n "${failed}" ]; then
  echo "Failed test jobs:${failed}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] All ${tested} cau result directories tested successfully."
