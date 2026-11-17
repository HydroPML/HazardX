<p align="center">
  <img src="Figs/HazardX%20Multimodal%20Hazard%20Monitoring%20Banner.png"
       alt="Multimodal Foundation Models for Natural Hazard Monitoring and Forecasting"
       width="100%">
</p>




# HazardX
<p align="center">
  <img src="Figs/fig11_GRSM.png"
       alt="HazardX Framework Overview"
       width="100%">
</p>
HazardX focuses on floods, landslides, and wildfires, while the other hazards illustrate potential future extensions.



# HazardX-Bench for natural hazard monitoring

<p align="center">
  <img src="Figs/monitoring.jpg" alt="HazardX Monitoring" width="100%">
</p>

The currently released monitoring implementation is located in [`HazardX-Bench/Monitoringcode/`](HazardX-Bench/Monitoringcode/).

## 🛠️ Installation

```bash
git clone https://github.com/HydroPML/HazardX.git
cd HazardX/HazardX-Bench/Monitoringcode
# Choose a PyTorch build matching your hardware and CUDA environment (cu126 is only an example).
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
# Install the additional dependencies required by your chosen model.
```

Some models require optional extras. For ChangeMamba's selective scan kernel:

```bash
# If you encounter GCC issues, consider: conda install -c conda-forge gcc=13 gxx=13 -y
cd src/models/ChangeMamba/kernels/selective_scan
pip install . --no-build-isolation
cd ../../../..
```

Some models also require local upstream weights in `pretrained_weight/` (details below).

## 🧪 Dataset Preparation

Dataset preparation guides are organized by disaster domain:

- Flood: [scripts/data_prep/flood/README.md](HazardX-Bench/Monitoringcode/scripts/data_prep/flood/README.md)
- Landslide: [scripts/data_prep/landslide/README.md](HazardX-Bench/Monitoringcode/scripts/data_prep/landslide/README.md)
- Wildfire: [scripts/data_prep/wildfire/README.md](HazardX-Bench/Monitoringcode/scripts/data_prep/wildfire/README.md)

## 📦 Pretrained Weights

Create the local weights directory inside `HazardX-Bench/Monitoringcode/`:

```bash
mkdir -p pretrained_weight
```

### Direct Downloads

```bash
# pretrain-vit-base-e199.pth
wget -O pretrained_weight/pretrain-vit-base-e199.pth \
  https://zenodo.org/records/7338613/files/pretrain-vit-base-e199.pth

# SpectralGPT+.pth
wget -O "pretrained_weight/SpectralGPT+.pth" \
  "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1"

# spec-vit-base-ultra-checkpoint-1599.pth
wget -O pretrained_weight/spec-vit-base-ultra-checkpoint-1599.pth \
  https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main/spec-vit-base-ultra-checkpoint-1599.pth

huggingface-cli download UTokyo-Yokoya-Lab/AnyDisaster-Pretrained_Weight \
  vssm_tiny_0230_ckpt_epoch_262.pth --local-dir pretrained_weight
```

### Additional Upstream Checkpoints

- **DINOv3:** https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  - `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16, LVD-1689M)
  - `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16, LVD-1689M)
  - `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (ViT-L/16, SAT-493M)
- **SAM 2.1:** https://github.com/facebookresearch/sam2
  - `sam2.1_hiera_small.pt`
  - `sam2.1_hiera_base_plus.pt`
- **SegFormer MiT:** https://github.com/NVlabs/SegFormer
  - `mit_b0.pth`, `mit_b1.pth`, `mit_b2.pth`, `mit_b3.pth`, `mit_b4.pth`, `mit_b5.pth`
- **SkySense++ backbone:** [repository](https://github.com/Jack-bo1220/SkySense) and [checkpoints](https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509)
  - Select the `hr` (high-resolution RGB / RGBNIR) variant, not the `s2` Sentinel-2 variant.
  - Save as `skysense_model_backbone_hr.pth` (confirm the filename expected by the chosen config).
  - For commercial use, contact the authors (`yansheng.li@whu.edu.cn`).

Depending on the selected model, your `pretrained_weight/` directory may include:

```text
pretrained_weight/
├── clay-v1.5.ckpt
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
├── dofav2_vit_base_e150.pth
├── dofav2_vit_large_e150.pth
├── Galileo_nano_encoder.pt
├── Prithvi_EO_V2_300M/
├── TerraMind_v1_base/
├── HSI_spatial_checkpoint-1600.pth
├── mit_b0.pth
├── mit_b1.pth
├── mit_b2.pth
├── mit_b3.pth
├── mit_b4.pth
├── mit_b5.pth
├── pretrain-vit-base-e199.pth
├── sam2.1_hiera_base_plus.pt
├── sam2.1_hiera_small.pt
├── skysensepp_model_backbone_hr.pth
├── spec-vit-base-ultra-checkpoint-1599.pth
├── SpectralGPT+.pth
└── vssm_tiny_0230_ckpt_epoch_262.pth
```

For upstream checkpoints not directly linked above, consult the corresponding model's official repository or documentation. Comply with the license for each model and dataset.

## 🚀 Quick Start

From `HazardX-Bench/Monitoringcode/`, after preparing the dataset and any required weights:

### Train

```bash
python train.py --config config/flood/cau/seg_unet.yaml
```

### Evaluate

```bash
python test.py --exp_path results/flood/cau/seg_unet
```

## 🗂️ Repository Layout

```text
HazardX/
├── README.md
└── HazardX-Bench/
    └── Monitoringcode/
        ├── README.md
        ├── config/
        ├── docs/
        ├── scripts/data_prep/
        ├── src/core/
        ├── src/datasets/
        ├── src/models/
        ├── src/tasks/
        ├── train.py
        └── test.py
```

Within `Monitoringcode/`:

- `src/core/`: trainer, config loader, registry, augmentation and metrics
- `src/tasks/`: segmentation, change detection and semantic change detection task handlers
- `src/datasets/`: dataset adapters and runtime data contracts
- `src/models/`: model wrappers and third-party implementations
- `config/`: experiment configurations grouped by domain and dataset
- `scripts/data_prep/`: dataset preparation instructions and utilities
- `docs/`: architecture and extension guidance

## 🏗️ Architecture and Extension

- [Architecture overview](HazardX-Bench/Monitoringcode/docs/ARCHITECTURE.md)
- [Extension guide](HazardX-Bench/Monitoringcode/docs/EXTENSION_GUIDE.md)

# HazardX-Bench for natural hazard forecasting
<p align="center">
  <img src="Figs/forecasting.jpg" alt="HazardX Forecasting" width="100%">
</p>





