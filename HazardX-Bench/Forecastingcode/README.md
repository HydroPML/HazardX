<h1 align="center">🌍 HazardX-Bench: Forecasting</h1>

## 🔭Overview

This repository brings together **FloodCastBench, FloodSimBench, LandslideCast, and WildfireSpreadTS**, along with representative deep learning models for **flood, landslide, and wildfire forecasting**. It provides a unified training and evaluation interface for 36 selected experiments while preserving the original model implementations, configurations, and evaluation protocols.

## 🛠️Installation

Requires Linux, Python 3.11, and a CUDA GPU. Validated with PyTorch 2.5.0, CUDA 12.4, and an RTX A6000.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For PDE-Transformer, use a separate environment with the same PyTorch version and `requirements-pde.txt`. See [environment details](docs/environments.md).

## 🧪Dataset Preparation

Download datasets from their official sources:

| Dataset | Official source | Expected local input |
| --- | --- | --- |
| FloodCastBench | [Dataset](https://zenodo.org/records/14017092), [official code](https://github.com/HydroPML/FloodCastBench) | Existing split `.pt` trajectories and original DEM rasters |
| FloodSimBench | [Official dataset](https://huggingface.co/datasets/chrimerss/FloodSimBench) | Native 1 m water-depth and DEM GeoTIFFs |
| LandslideCast | [Official project](https://github.com/HydroPML/Landslidecast), [paper](https://doi.org/10.1029/2023EA003417) | Canonical `2d_t10.h5` with scenarios `x1` through `x12000` |
| WildfireSpreadTS | [Official dataset](https://zenodo.org/records/8006177), [official code](https://github.com/SebastianGer/WildfireSpreadTS) | Official daily TIFFs or the original lossless HDF5 conversion |


| Dataset | Physical model input (excluding batch and positional encodings) | Prediction protocol |
| --- | --- | --- |
| FloodCastBench | `1 × 3 × 128 × 128`: current depth, rainfall, DEM | Native 128/96 overlapping tiled scene inference; autoregressive depth |
| FloodSimBench | `1 × 12 × 1024 × 1024`: five depths, six rainfall fields, DEM | Next-step prediction; complete evaluation rolls out 67 steps |
| LandslideCast | Three static/initial 200 × 200 fields; architecture-specific nine temporal positions | Eight future depth/velocity frames at 10–80 s; GPhyT uses its original Euler adapter |
| WildfireSpreadTS | `5 × 40 × 128 × 128` | Next-day fire segmentation from five daily observations |

See the [preparation guide](scripts/data_prep/README.md) for layouts, channels, splits, and conversion commands.

## 📦Pretrained Weights

Place the required initialization weights in `pretrained/`. These files are needed for both training and evaluation of the corresponding pretrained models.

| Model | Official pretrained file | Local filename under `pretrained/` |
| --- | --- | --- |
| PDE-Transformer SC-S | [diffusion_pytorch_model.safetensors](https://huggingface.co/thuerey-group/pde-transformer/blob/main/sc-s/diffusion_pytorch_model.safetensors) | `pde_sc_s.safetensors` |
| DPOT-S | [model_S.pth](https://huggingface.co/hzk17/DPOT/blob/main/model_S.pth) | `dpot_s.pth` |
| Poseidon-T | [pytorch_model.bin](https://huggingface.co/camlab-ethz/Poseidon-T/blob/main/pytorch_model.bin) | `poseidon_t.bin` |
| GPhyT-S | [gphyt-S.pth](https://huggingface.co/flwi/Physics-Foundation-Model/blob/main/gphyt-S.pth) | `gphyt_s.pth` |
| VideoMAEv2 | [vit_b_k710_dl_from_giant.pth](https://huggingface.co/OpenGVLab/VideoMAE2/blob/main/distill/vit_b_k710_dl_from_giant.pth) | `vit_b_k710_dl_from_giant.pth` |
| Prithvi-EO-2.0 300M | [Prithvi_EO_V2_300M.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/Prithvi_EO_V2_300M.pt) | `Prithvi_EO_V2_300M.pt` |

U-Net, ConvLSTM, FNO, EarthFormer, and U-TAE are trained from scratch.

Alternatively, use the download helper, which verifies the file hash. Prithvi also requires its companion source and configuration:

```bash
python scripts/download_pretrained.py --model prithvi
python scripts/setup_prithvi.py --pretrained-root pretrained
```

## 🚀Quick Start

Run commands from this directory. Change `--config` and data paths to select another experiment.


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

For a limited check, add `--max-batches 1` and, for flood datasets, `--max-steps 1`. Limited runs do not reproduce full-test metrics. FloodCastBench uses within-source testing by default; add `--protocol mozambique` for the original Mozambique protocol.

### Training

Use the GPU count specified by `distributed.world_size` in the YAML. For example, FloodSimBench U-Net uses two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29541 \
  train.py --config config/FloodSimBench/unet.yaml \
  --data-root /path/to/FloodSimBench --output-dir outputs/train_floodsim_unet
```

Resume with `--resume /path/to/last.pt`; `best.pth` is the best-validation snapshot. See [per-model training commands](docs/train_commands.md) for GPU counts and settings. Use different master ports for concurrent runs.

### Small-batch check

Check checkpoint inference, native metrics, and one temporary training step:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py \
  --config config/LandslideCast/unet.yaml --data-root /path/to/LandslideCast \
  --output-dir outputs/smoke_landslide_unet
```

Modified weights are discarded. See the [validation report](docs/validation_report.md) for coverage and limitations.

## 🗂️Repository Layout

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

## 🏗️Reproducibility

See the [selected experiments](docs/selected_experiments.md) for seeds, checkpoint epochs, and metrics; the [experiment catalog](results/catalog.json) for saved configurations; and [third-party provenance](docs/third_party.md) for source revisions and weight hashes. Protocol details are in the [reproducibility guide](docs/reproducibility.md).

## License

Original code in this repository is licensed under the [Apache License 2.0](LICENSE). Third-party code, datasets, and pretrained weights remain subject to their respective licenses.

