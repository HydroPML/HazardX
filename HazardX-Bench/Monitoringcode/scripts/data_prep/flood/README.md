# Flood Dataset Preparation

This directory contains preparation guides for **FloodPlanet** and **CAU-Flood**. Run all commands from the HazardX project root. The preprocessing scripts use only the Python standard library; no model installation or weight downloads are required in advance.

## FloodPlanet

Keep the original dataset layout; no TIFF conversion is required:

```text
<FLOODPLANET_ROOT>/
└── CSDAP_complete/
    ├── <region_a>/
    │   ├── PS/<id>.tif
    │   └── labels/<id>.tif
    └── <region_b>/
        ├── PS/<id>.tif
        └── labels/<id>.tif
```

Images and labels must have matching names within the same region. The default sensor is four-channel PlanetScope (PS). If train/validation/test lists are provided by an experiment or by the dataset authors, use them directly instead of creating new splits.

To create custom random splits, run:

```bash
python scripts/data_prep/flood/prepare_segmentation_splits.py \
  /root/autodl-tmp/datasetcsc/flood/FloodPlanet \
  --sensor PS --train-ratio 0.7 --val-ratio 0.15 --seed 42
```

The script writes `train.txt`, `validation.txt`, and `test.txt` to the dataset root. Each line contains a sample ID without a file extension. Running the script again overwrites these three lists but does not modify the source images. The test ratio is the remaining fraction, 15% by default.

The script raises an error for missing labels, duplicate IDs across regions, invalid ratios, or empty splits caused by too few samples. It checks file pairing only; it does not inspect pixel values inside TIFF files.

The default split is random at the sample level and does not guarantee region-level independence. For cross-region generalization evaluation, provide region-independent split lists. Because the loader selects samples by filename stem, IDs must be unique across regions.

In the relevant `configs/flood/floodplanet/*.yaml` files, set `dataset_path` for train/val/test to the actual dataset root and set `data_list_path` to the three lists above. Keep `sensor: PS` and `channels: [0, 1, 2, 3]` consistent with the data. The current configuration uses `scale: 10000.0`; adjust it if the source data has already been scaled to avoid scaling twice.

The loader maps labels as follows: raw `0` → ignore value `255`, `1` → non-flood `0`, and `2` → flood `1`. No pre-binarization is required.

## CAU-Flood

After extraction, prepare the following layout. Filenames must match across the three directories in each split:

```text
<CAU_RAW>/
├── train/
│   ├── flood_vv/<id>.png
│   ├── opt/<id>.png
│   └── vv/<id>.png
└── test/
    ├── flood_vv/<id>.png
    ├── opt/<id>.png
    └── vv/<id>.png
```

```bash
python scripts/data_prep/flood/reorganize_cau_flood.py \
  --input-root /root/autodl-tmp/datasetcsc/flood/CAU_Flood/raw \
  --output-dir /root/autodl-tmp/datasetcsc/flood/CAU_Flood \
  --val-ratio 0.1 --seed 42
```

The default paths are `/root/autodl-tmp/datasetcsc/flood/CAU_Flood/raw` and `/root/autodl-tmp/datasetcsc/flood/CAU_Flood`. The script copies the files and generates:

```text
<CAU_ROOT>/
├── PRE/*.png     # from opt
├── POST/*.png    # from vv
├── GT/*.png      # from flood_vv
├── train.txt
├── validation.txt
└── test.txt
```

The original test set is preserved; only the original training set is split to create validation data. Use `--val-ratio 0` to disable validation splitting; if you do so, configure training accordingly to handle an empty validation split. The output `PRE`, `POST`, and `GT` directories must be empty before running the script to prevent stale files from being mixed in. Overlapping sample IDs between the original train and test sets cause an error before copying. The original data is never moved or deleted.

In the relevant `configs/flood/cau/*.yaml` files, set `dataset_path` for train/val/test to `<CAU_ROOT>` and set `data_list_path` to the corresponding lists. The CAU-Flood loader supports this PNG directory layout for segmentation and change detection. Ensure that model input channels and normalization match the actual optical/SAR data and the configured input mode.

For the unified data layout and preparation order, see the top-level [README](../README.md).
