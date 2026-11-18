
<h1 align="center">🌍 HazardX-Bench: Forecasting</h1>

## 🔭Overview

This repository brings together **FloodCastBench, FloodSimBench, LandslideCast, and WildfireSpreadTS**, along with representative deep learning models for **flood, landslide, and wildfire forecasting**. It provides a unified training and evaluation interface for 36 selected experiments while preserving the original model implementations, configurations, and evaluation protocols.

## Installation

Requires Linux, Python 3.11, and a CUDA GPU. Validated with PyTorch 2.5.0, CUDA 12.4, and an RTX A6000.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For PDE-Transformer, use a separate environment with the same PyTorch version and `requirements-pde.txt`. See [environment details](docs/environments.md).

## Dataset Preparation

Download datasets from their official sources:

| Dataset | Official source | Expected local input |
| --- | --- | --- |
| FloodCastBench | [Dataset](https://zenodo.org/records/14017092), [official code](https://github.com/HydroPML/FloodCastBench) | Existing split `.pt` trajectories and original DEM rasters |
| FloodSimBench | [Official dataset](https://huggingface.co/datasets/chrimerss/FloodSimBench) | Native 1 m water-depth and DEM GeoTIFFs |
| LandslideCast | [Official project](https://github.com/HydroPML/Landslidecast), [paper](https://doi.org/10.1029/2023EA003417) | Canonical `2d_t10.h5` with scenarios `x1` through `x12000` |
| WildfireSpreadTS | [Official dataset](https://zenodo.org/records/8006177), [official code](https://github.com/SebastianGer/WildfireSpreadTS) | Official daily TIFFs or the original lossless HDF5 conversion |

See the [preparation guide](scripts/data_prep/README.md) for layouts, channels, splits, and conversion commands. Public access to the canonical LandslideCast HDF5 and FloodCastBench raw-to-trajectory conversion remains unverified; the guide documents these dependencies.

| Dataset | Physical model input (excluding batch and positional encodings) | Prediction protocol |
| --- | --- | --- |
| FloodCastBench | `1 × 3 × 128 × 128`: current depth, rainfall, DEM | Native 128/96 overlapping tiled scene inference; autoregressive depth |
| FloodSimBench | `1 × 12 × 1024 × 1024`: five depths, six rainfall fields, DEM | Next-step prediction; complete evaluation rolls out 67 steps |
| LandslideCast | Three static/initial 200 × 200 fields; architecture-specific nine temporal positions | Eight future depth/velocity frames at 10–80 s; GPhyT uses its original Euler adapter |
| WildfireSpreadTS | `5 × 40 × 128 × 128` | Next-day fire segmentation from five daily observations |

LandslideCast's temporal positions are conditioning positions, not nine observed
future states. The original GPhyT adapter also reads the initial velocity from
the same scenario; the other models retain their original static-field inputs.
See the selected YAML and model adapter for the exact tensor interface.

## Pretrained Weights

Pretrained initialization weights are **not** included or uploaded. Download
only the weights required by the chosen model from their official publishers:

| Model | Official pretrained file | Local filename under `pretrained/` |
| --- | --- | --- |
| PDE-Transformer SC-S | [diffusion_pytorch_model.safetensors](https://huggingface.co/thuerey-group/pde-transformer/blob/main/sc-s/diffusion_pytorch_model.safetensors) | `pde_sc_s.safetensors` |
| DPOT-S | [model_S.pth](https://huggingface.co/hzk17/DPOT/blob/main/model_S.pth) | `dpot_s.pth` |
| Poseidon-T | [pytorch_model.bin](https://huggingface.co/camlab-ethz/Poseidon-T/blob/main/pytorch_model.bin) | `poseidon_t.bin` |
| GPhyT-S | [gphyt-S.pth](https://huggingface.co/flwi/Physics-Foundation-Model/blob/main/gphyt-S.pth) | `gphyt_s.pth` |
| VideoMAEv2 | [vit_b_k710_dl_from_giant.pth](https://huggingface.co/OpenGVLab/VideoMAE2/blob/main/distill/vit_b_k710_dl_from_giant.pth) | `vit_b_k710_dl_from_giant.pth` |
| Prithvi-EO-2.0 300M | [Prithvi_EO_V2_300M.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/Prithvi_EO_V2_300M.pt) | `Prithvi_EO_V2_300M.pt` |

U-Net, ConvLSTM, FNO, EarthFormer, and U-TAE use the original scratch
initializations in these experiments. The retained PT–FT constructors initialize
their backbone before loading the full benchmark state, so evaluation also
requires the corresponding official initialization file. Exact upstream
revisions and weight hashes are in [third-party provenance](docs/third_party.md).

The optional helper downloads only the chosen official initialization at its
audited repository revision and checks its SHA256:

```bash
python scripts/download_pretrained.py --model prithvi
```

For Prithvi, also place its audited official source/config next to the weight
file. These two small companion files are included as source, not weights:

```bash
python scripts/setup_prithvi.py --pretrained-root pretrained
```

## Quick Start

Run commands from this directory. Paths in YAMLs are relative to this release;
CLI path overrides allow another data, DEM, initialization, or output location.

### Download benchmark checkpoints

The selected trained weights are in the
[Forecasting checkpoint folder](https://drive.google.com/drive/folders/1pdkteoAp0TyW5NUtx-q_i-Nwi9hT8axo).
[The checkpoint manifest](checkpoints/manifest.csv) gives each model's seed,
metrics, file size, SHA256, and observed Drive links.

```bash
python scripts/download_checkpoints.py --dataset FloodSimBench --model unet
# Or download all 36 selected checkpoints:
python scripts/download_checkpoints.py --all
```

The connected upload service accepts at most 100 MiB per file. Larger
checkpoints are therefore stored as ordered 90 MiB **byte parts**, with
`parts.json`; the helper verifies every part and restores the exact original
`best.pth`. These are not separately trained shards. Files are never silently
overwritten. For manually downloaded parts, use `--offline --parts-dir PATH`.

### Independent evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config config/FloodSimBench/unet.yaml \
  --data-root /path/to/FloodSimBench \
  --output-dir outputs/eval_floodsim_unet
```

The same command supports every dataset/model by changing `--config` and data
paths. `test.py` loads the selected weights and normalization, enters eval mode,
and runs the original evaluator without creating an optimizer. Wildfire's
decision threshold comes from the saved validation selection, never the test
labels. The original AP tie handling is retained.

For a quick evaluation limit, add `--max-batches 1`; for flood datasets also add
`--max-steps 1`. A limited run does not reproduce full-test metrics. FloodCast's
default is within-source testing; `--protocol mozambique` runs the original
one-/two-day trajectory regrouping from Mozambique/576. Mozambique/1728 is
excluded by that published protocol, as documented in
[reproducibility details](docs/reproducibility.md).

### Training and resume

The YAML's `distributed.world_size` is part of the original training recipe.
For example, FloodSimBench U-Net uses two ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29541 \
  train.py --config config/FloodSimBench/unet.yaml \
  --data-root /path/to/FloodSimBench --output-dir outputs/train_floodsim_unet
```

Resume an interrupted run with `--resume /path/to/last.pt`. The published
`best.pth` is the selected best-validation snapshot; it is not the final
interrupted-run cursor. Exact per-model training commands and rank counts are
in [training commands](docs/train_commands.md). Use an independent master port
for each simultaneously running group. Training is never launched by `test.py`.

### Small-batch check

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py \
  --config config/LandslideCast/unet.yaml --data-root /path/to/LandslideCast \
  --output-dir outputs/smoke_landslide_unet
```

This reads real data, checks a checkpoint forward and native batch metrics,
then executes one temporary training step. Modified weights are discarded.
[The release validation report](docs/validation_report.md) records all 36
original/release comparisons and the limits of this check.

## Repository Layout

```text
Forecastingcode/
├── config/<dataset>/       # 36 verified experiment YAMLs
├── src/
│   ├── data.py             # Shared data/evaluation interface
│   ├── models/             # Original adapters and official backbone sources
│   └── core/               # Shared CLI, lifecycle, and audited native math
├── scripts/
│   ├── data_prep/          # Preparation guide and official HDF5 wrapper
│   ├── download_checkpoints.py
│   └── smoke_test.py
├── checkpoints/            # Download manifests; no weights in the code ZIP
├── results/                # Selected original final evaluation JSONs
├── docs/                   # Validation, provenance, environments, commands
├── train.py
├── test.py
├── requirements.txt
└── requirements-pde.txt
```

## Reproducibility

The first formal seed is **42** for FloodCastBench, **1** for FloodSimBench,
and **0** for LandslideCast and WildfireSpreadTS. Wildfire's Prithvi selection
uses the **second experiment, seed 1**, as requested. Selection was checked
against the actual checkpoint metadata and saved result files.

[The experiment catalog](results/catalog.json) preserves all saved metrics,
selected checkpoint epochs, and configurations. [The manifest](checkpoints/manifest.json)
connects each selected experiment to its uploaded checkpoint. Original numerical
source hashes are retained in `docs/native_source_manifest.json`. No full
retraining or full-test reevaluation was performed during release preparation;
the archived final metrics remain separate from small-batch validation metrics.

The [selected-experiment table](docs/selected_experiments.md) lists each model's
actual seed, best-checkpoint epoch, and original main test metrics. Public Drive
download hashes and folder audits are included with the release evidence.

## Licensing

See [licensing and attribution](docs/licensing.md). Dataset, backbone source,
pretrained initialization, and trained benchmark checkpoint permissions are
separate. Existing license notices are preserved. A repository-wide code or
benchmark-weight license was not found in the audited project, and no new
license is invented here. DPOT/Poseidon source redistribution permissions and
the canonical LandslideCast data access must be resolved by the maintainers
before an unrestricted public release. Poseidon pretrained weights carry a
noncommercial restriction. This archive does not override upstream terms.
