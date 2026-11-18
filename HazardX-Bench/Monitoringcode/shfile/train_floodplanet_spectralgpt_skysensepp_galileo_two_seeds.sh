#!/usr/bin/env bash
# FloodPlanet (PlanetScope) segmentation with SpectralGPT, SkySense++, and
# Galileo Nano. Runs seeds 43 and 44 sequentially on one GPU, tests every
# successful experiment, and refreshes the FloodPlanet result tables.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export CONFIG_NAMES="seg_spectralgpt.yaml seg_skysensepp.yaml seg_galileo_nano.yaml"
export SEEDS="${SEEDS:-43 44}"
export GPU_ID="${GPU_ID:-0}"

exec bash "${PROJECT_ROOT}/shfile/train_floodplanet_two_seeds.sh" "$@"
