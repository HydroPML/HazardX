# Wildfire Dataset Preparation

Store all wildfire datasets under `/root/autodl-tmp/datasetcsc/wildfire`. Run the commands below from the project root. Install the project dependencies before running the conversion scripts.

## FLOGA

Place the original-year HDF5 files for FLOGA at `/root/autodl-tmp/datasetcsc/wildfire/floga/raw_h5/` and the experiment split CSV at `/root/autodl-tmp/datasetcsc/wildfire/floga/data_split.csv`.

```bash
python scripts/data_prep/wildfire/floga/create_dataset.py \
  --out_size 256 256 --sample 1 --random_seed 999
python scripts/data_prep/wildfire/floga/data_convert.py --patch_size 256
```

By default, the first command creates `floga/converted_data_256/`, and the second creates T1, T2, GT, `train.txt`, `val.txt`, and `test.txt` under `floga/patch256/`. The `--src_dir` and `--tgt_dir` arguments of `data_convert.py` should point to the parent directory containing `floga/`; both default to `/root/autodl-tmp/datasetcsc/wildfire`.

Set the training dataset root to `/root/autodl-tmp/datasetcsc/wildfire/floga`; the lists are stored in its `patch256/` subdirectory. Optical data keeps its original scaling. GT values are `0` = non-fire, `1` = fire, and `2` = other event; the loader maps `2` to the ignore value `255`. If patch files are moved, regenerate any `.pkl` index that may contain the old paths; see the top-level documentation for details.

## S2-WCD

Extract the dataset to `/root/autodl-tmp/datasetcsc/wildfire/raw/S2-WCD/` and preserve the original event directories.

```bash
python scripts/data_prep/wildfire/s2_wcd/data_convert.py --mode all
```

The default output is `/root/autodl-tmp/datasetcsc/wildfire/s2_wcd/`. It contains T1, T2, T1_sen2, T2_sen2, GT, and the generated `train.txt`, `val.txt`, and `test.txt` lists. Use `--mode sen2` to generate only the Sentinel-2 data; use `all` for the complete initial preparation. The training configuration should already point its dataset root and lists to this output directory.

For the unified data layout, see the top-level [README](../README.md). The conversion scripts do not move the source data when paths are configured.
