<div align="center">

<h1>🌍 HazardX-Bench: Monitoring</h1>

[**Installation**](#installation) | [**Dataset Preparation**](#dataset-preparation) | [**Pretrained Weights**](#pretrained-weights) | [**Released Assets**](#released-assets) | [**Quick Start**](#quick-start) | [**Repo Layout**](#repository-layout)

</div>

<!-- 
## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Dataset Preparation](#dataset-preparation)
- [Architecture And Extension](#architecture-and-extension) -->

## 🔭Overview
*HazardX* is the official repository for our paper, *Multimodal Foundation Models for Natural Hazard Monitoring and Forecasting: Methods, Benchmarks, and Perspectives*.

One of our key motivations is that current disaster mapping research is highly fragmented: benchmarks, tasks, and model implementations are often inconsistent across papers, making fair evaluation, reproduction, and further development unnecessarily difficult.

This repo unifies widely used disaster mapping benchmarks and representative deep learning models across major research directions, and provides a consistent training and evaluation pipeline for:
- Flood mapping
- Landslide segmentation
- Wildfire analysis

It is designed to help researchers:
- Reproduce the results reported in our paper
- Evaluate models under a unified protocol
- Use strong baselines out of the box
- Build and test their own improvements with minimal engineering overhead

## 🛠️Installation

Base environment and optional model-specific extras.

```bash
# NOTE: --index-url should match the version of your local CUDA toolkit for compiling ChangeMamba kernels (cu126 is just an example)
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

Some models require optional extras:

- ChangeMamba selective scan kernel:
  ```bash
  # run `conda install -c conda-forge gcc=13 gxx=13 -y` if you meet GCC issues
  cd src/models/ChangeMamba/kernels/selective_scan
  pip install . --no-build-isolation
  ```
- Local pretrained checkpoints under `pretrained_weight/` for model families such as SegFormer, HRNet, SAM/SAM2, DINOv3, HyperSigma, SkySense, SpectralGPT, and ChangeMamba. See [Pretrained Weights](#pretrained-weights) below.

## 🧪Dataset Preparation

Dataset preparation guides are organized by disaster domain:

- Flood: [scripts/data_prep/flood/README.md](scripts/data_prep/flood/README.md)
- Landslide: [scripts/data_prep/landslide/README.md](scripts/data_prep/landslide/README.md)
- Wildfire: [scripts/data_prep/wildfire/README.md](scripts/data_prep/wildfire/README.md)


## 📦Pretrained Weights

For a local checkout, pretrained checkpoints used by this project are expected under:

```text
/root/workcsc/HazardX_Monitoring/pretrained_weight
```

The YAML files under `configs/` use this directory directly. Before starting an experiment, verify that the required checkpoint exists:

```bash
ls -lh /root/workcsc/HazardX_Monitoring/pretrained_weight
```

If the directory has not been created yet:

```bash
mkdir -p /root/workcsc/HazardX_Monitoring/pretrained_weight
```

### Released Assets

The Hugging Face repository contains the following experiment assets:

- **Pretrained weights** for the model backbones and components required by the released configurations.
- **Datasets** used by the HazardX Monitoring experiments.
- **Experiment outputs**, including model checkpoints, test metrics, logs, and summary tables where available.

Download or synchronize the required files from the Hugging Face repository before running an experiment. Keep the local directory layout consistent with the paths referenced by the configuration files, especially `pretrained_weight/`, dataset roots, and `output/`.

The Hugging Face repository is the canonical distribution location for these released assets. The repository source code and the asset repository are maintained separately.

### Direct Downloads

The current checkpoint directory contains:

```text
/root/workcsc/HazardX_Monitoring/pretrained_weight/
├── AnySat.pth
├── clay-v1.5.ckpt
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
├── dofav2_vit_base_e150.pth
├── dofav2_vit_large_e150.pth
├── Galileo_nano_encoder.pt
├── Prithvi_EO_V2_300M.pt
├── pretrain-vit-base-e199.pth
├── sam2.1_hiera_base_plus.pt
├── sam2.1_hiera_large.pt
├── sam2.1_hiera_small.pt
├── sam2.1_hiera_tiny.pt
├── segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth
├── segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth
├── segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth
├── segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth
├── segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth
├── segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth
├── skysense_model_backbone_hr.pth
├── skysensepp_release_hr.pth
├── SpectralGPT+.pth
├── TerraMind_v1_base.pt
└── vssm_tiny_0230_ckpt_epoch_262.pth
```

## 🚀 Quick Start

The pretrained weights, datasets, and experiment outputs for HazardX Monitoring are released on Hugging Face:

<https://huggingface.co/liyao0719/HazardX_Monitoring>

You can use this repository to reproduce the experiments and download the released assets from the Hugging Face repository above.

The runnable batch scripts are stored in:

```text
/root/workcsc/HazardX_Monitoring/shfile
```

Run all LANDREF change-detection models sequentially on GPU 0 with seeds 43 and 44. Each completed model is evaluated automatically, and the result tables are updated after testing:

```bash
cd /root/workcsc/HazardX_Monitoring
bash shfile/train_landref_cd_two_seeds.sh
```

Preview all commands without starting training:

```bash
DRY_RUN=1 bash shfile/train_landref_cd_two_seeds.sh
```

Run the Landslide4Sense experiments:

```bash
bash shfile/train_landslide4sensen_two_seeds.sh
```

LANDREF checkpoints and test metrics are written to `output/landref/`. The Excel and TSV summary tables are written to `output/tables/`.

## 🗂️Repository Layout

 Main directories and what they are responsible for:

- `src/core/`: trainer, config loader, registry, augmentation, metrics
- `src/tasks/`: task handlers for segmentation, change detection, and semantic change detection
- `src/datasets/`: dataset adapters and runtime data contracts
- `src/models/`: model wrappers and vendored third-party implementations
- `configs/`: experiment configs grouped by domain and dataset
- `pretrained_weight/`: local pretrained checkpoints
- `shfile/`: executable batch-training scripts
- `scripts/data_prep/`: dataset preparation guides and helper scripts
- `docs/`: architecture notes and extension guidance


## 🏗️Architecture And Extension

Internal runtime design and extension entry points.

- Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Extension guide: [docs/EXTENSION_GUIDE.md](docs/EXTENSION_GUIDE.md)

## 🙋Q & A
For questions, please open an issue in this repository.
