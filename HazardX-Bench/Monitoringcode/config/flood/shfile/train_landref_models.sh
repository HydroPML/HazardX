#!/bin/sh

set -u

PROJECT_ROOT="/root/workcsc/AnyDisasterMapping"
CONFIG_ROOT="${PROJECT_ROOT}/configs/landslide/landref_cd"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/disaster/bin/python}"

# Disable the Albumentations network version check on every new training job.
export NO_ALBUMENTATIONS_UPDATE=1

# configs="
# cd_unet.yaml
# cd_segformer_b0.yaml
# cd_segformer_b1.yaml
# cd_segformer_b2.yaml
# cd_segformer_b3.yaml
# cd_segformer_b4.yaml
# cd_segformer_b5.yaml
# cd_sam2_fpn_s.yaml
# cd_sam2_fpn_bplus.yaml
# cd_sam2_fpn_l.yaml
# cd_sam2_fpn_t.yaml
# cd_dinov3_dpt_vitb_lvd.yaml
# cd_dinov3_dpt_vitl_lvd.yaml
# cd_prithvieo2.yaml
# cd_satmae.yaml
# cd_spectralgpt.yaml
# cd_dofav2_base.yaml
# cd_anysat_base.yaml
# cd_skysensepp.yaml
# cd_terramind.yaml
# cd_galileo_nano.yaml
# cd_clay_v15_large.yaml
# "

configs="
cd_siamcrnn_r50.yaml
cd_dsifn.yaml
cd_bit_r50.yaml
cd_changemamba.yaml
"

cd "${PROJECT_ROOT}" || exit 1

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

failed=""

for config in ${configs}; do
  config_path="${CONFIG_ROOT}/${config}"
  echo "[$(date '+%F %T')] Starting ${config}"

  if [ ! -f "${config_path}" ]; then
    echo "[$(date '+%F %T')] Missing config ${config_path}" >&2
    failed="${failed} ${config}:missing"
    continue
  fi

  if "${PYTHON_BIN}" train.py --config "${config_path}"; then
    echo "[$(date '+%F %T')] Finished ${config}"
  else
    status=$?
    echo "[$(date '+%F %T')] Failed ${config} (exit code ${status})" >&2
    failed="${failed} ${config}:${status}"
  fi
done

if [ -n "${failed}" ]; then
  echo "Failed training jobs:${failed}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] All LANDREF-CD training jobs finished successfully."
