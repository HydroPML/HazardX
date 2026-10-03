#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/root/workcsc/AnyDisasterMapping"
CONFIG_ROOT="${PROJECT_ROOT}/configs/wildfire/s2wcd_sen2_norm"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/disaster/bin/python}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/results/wildfire/s2wcd_sen2_norm/batch_logs}"
START_FROM="${START_FROM:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Disable the Albumentations network version check on every new training job.
export NO_ALBUMENTATIONS_UPDATE=1
# Avoid multi-gigabyte core dumps if a native CUDA extension crashes.
ulimit -c 0


configs="
cd_unet.yaml
cd_segformer_b0.yaml
cd_segformer_b1.yaml
cd_segformer_b2.yaml
cd_segformer_b3.yaml
cd_segformer_b4.yaml
cd_segformer_b5.yaml
cd_sam2_fpn_s.yaml
cd_sam2_fpn_bplus.yaml
cd_sam2_fpn_l.yaml
cd_sam2_fpn_t.yaml
cd_dinov3_dpt_vitb_lvd.yaml
cd_dinov3_dpt_vitl_lvd.yaml
cd_prithvieo2.yaml
cd_satmae.yaml
cd_spectralgpt.yaml
cd_dofav2_base.yaml
cd_anysat_base.yaml
cd_skysensepp.yaml
cd_terramind.yaml
cd_galileo_nano.yaml
cd_clay_v15_large.yaml
cd_siamcrnn_r50.yaml
cd_dsifn.yaml
cd_bit_r50.yaml
cd_changeos_r50.yaml
cd_changemamba.yaml
"

cd "${PROJECT_ROOT}" || exit 1

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

# Fail before starting a multi-day run if a config, dataset split, registry
# entry, or the ChangeMamba CUDA extension is unavailable.
"${PYTHON_BIN}" - "${CONFIG_ROOT}" ${configs} <<'PY'
import importlib.util
import sys
from pathlib import Path

import yaml
from src.core.registry import dataset_libs, model_libs

config_root = Path(sys.argv[1])
errors = []
for name in sys.argv[2:]:
    path = config_root / name
    if not path.is_file():
        errors.append(f"{name}: config not found: {path}")
        continue
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        errors.append(f"{name}: invalid YAML: {exc}")
        continue
    model_name = (cfg.get("model") or {}).get("name")
    dataset_name = (cfg.get("dataset") or {}).get("name")
    if model_name not in model_libs:
        errors.append(f"{name}: unknown model '{model_name}'")
    if dataset_name not in dataset_libs:
        errors.append(f"{name}: unknown dataset '{dataset_name}'")
    model_kwargs = (cfg.get("model") or {}).get("kwargs") or {}
    for key, value in model_kwargs.items():
        key_lower = key.lower()
        if not isinstance(value, str) or not any(
            marker in key_lower for marker in ("weight", "checkpoint", "pretrained")
        ):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.exists():
            errors.append(f"{name}: missing model.{key}: {value}")
    for split in ("train", "val", "test"):
        split_cfg = (cfg.get("dataset") or {}).get(split) or {}
        for key in ("dataset_path", "data_list_path"):
            value = split_cfg.get(key)
            if value and not Path(value).exists():
                errors.append(f"{name}: missing {split}.{key}: {value}")
    if model_name == "changemamba" and importlib.util.find_spec("selective_scan_cuda_oflex") is None:
        errors.append(f"{name}: selective_scan_cuda_oflex is not installed")

if errors:
    print("Preflight failed:", file=sys.stderr)
    print("\n".join(f"  - {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Preflight passed for {len(sys.argv) - 2} S2-WCD configurations.")
PY
preflight_status=$?
if [ "${preflight_status}" -ne 0 ]; then
  exit "${preflight_status}"
fi

if [ -n "${START_FROM}" ] && ! printf '%s\n' ${configs} | grep -Fxq "${START_FROM}"; then
  echo "START_FROM does not match a configured job: ${START_FROM}" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run completed; no training jobs were started."
  exit 0
fi

failed=""
started=0

for config in ${configs}; do
  if [ "${started}" -eq 0 ] && [ -n "${START_FROM}" ]; then
    if [ "${config}" != "${START_FROM}" ]; then
      continue
    fi
  fi
  started=1

  config_path="${CONFIG_ROOT}/${config}"
  log_path="${LOG_ROOT}/${config%.yaml}.log"
  echo "[$(date '+%F %T')] Starting ${config}"

  if [ ! -f "${config_path}" ]; then
    echo "[$(date '+%F %T')] Missing config ${config_path}" >&2
    failed="${failed} ${config}:missing"
    continue
  fi

  "${PYTHON_BIN}" train.py --config "${config_path}" 2>&1 | tee "${log_path}"
  pipeline_status=("${PIPESTATUS[@]}")
  train_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  if [ "${train_status}" -ne 0 ]; then
    status=${train_status}
  else
    status=${tee_status}
  fi
  if [ "${status}" -eq 0 ]; then
    echo "[$(date '+%F %T')] Finished ${config}"
  else
    echo "[$(date '+%F %T')] Failed ${config} (exit code ${status})" >&2
    failed="${failed} ${config}:${status}"
    if [ "${STOP_ON_ERROR}" = "1" ]; then
      break
    fi
  fi
done

if [ -n "${failed}" ]; then
  echo "Failed training jobs:${failed}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] All S2-WCD training jobs finished successfully."
