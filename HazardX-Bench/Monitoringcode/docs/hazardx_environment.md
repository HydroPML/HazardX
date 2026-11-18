# hazardx Training Environment

The Conda environment on the current machine is `/root/miniconda3/envs/hazardx`.
It uses Python 3.10, PyTorch 2.5.1 + CUDA 11.8, torchvision 0.20.1, and NumPy 1.26.4.
The system driver and system CUDA installation have not been modified; the CUDA build tools are installed inside the environment.

## Running the Landslide Experiments

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hazardx
cd /root/workcsc/HazardX

# Preview the 96 training commands for 48 configurations and two seeds
DRY_RUN=1 bash shfile/train_landslide4sensen_two_seeds.sh

# Run formally: GPU 0 uses seed 43, GPU 1 uses seed 44, with two queues in parallel
bash shfile/train_landslide4sensen_two_seeds.sh
```

Python can also be selected explicitly:

```bash
PYTHON_BIN=/root/miniconda3/envs/hazardx/bin/python \
bash /root/workcsc/HazardX/shfile/train_landslide4sensen_two_seeds.sh
```

The output root directory is `/root/autodl-tmp/output/landslide`; logs and batch summaries are stored in its `batch_logs` directory.

## Compatibility Handling

- The PyTorch 1.13.1 version originally pinned in `pyproject.toml` is older than the official SAM2 requirement. This environment uses `requirements-hazardx.txt` and `constraints-hazardx.txt`; do not run `pip install -e .`, as it may reinstall an older PyTorch version.
- MMCV uses `mmcv-full 1.7.2` with operator extensions; the pure-Python `mmcv` package of the same version must not be installed at the same time.
- ChangeMamba's `selective_scan` is compiled from source and includes the `core` and `oflex` CUDA extensions. The compiled copy only generates V100 SM70 kernels; recompile it for another GPU architecture.
- AnySat source is at `/root/workcsc/AnySat`, and DOFA source is at `/root/workcsc/DOFA`.
  AnySat commit: `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`;
  DOFA commit: `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9`.
- `/root/workcsc/AnyDisasterMapping` links to HazardX; `pretrained_weight` links to `/root/autodl-tmp/pretrained_weight`.
- The dataset package's `__init__.py` no longer imports the deleted `infra` datasets; they are still imported on demand through the registry.

This document records the current machine's environment and execution method. The dependency lists do not replace compilation of source-code extensions.

## Local Verification Results

- `pip check`: no dependency conflicts; all required model modules import successfully.
- V100 CUDA tensor operations pass; U-Net forward and backward passes pass on real 14-channel L4S and 34-channel LANDREF-CD samples.
- Both `core` and `oflex` `selective_scan` extensions pass CUDA forward and backward passes.
- Batch-script `DRY_RUN` confirms 48 configurations, seeds 43/44, and 96 experiments total.
- The 96 complete training runs were not executed, nor was every model's complete training workflow individually verified.

The installed-package snapshot is in `hazardx-installed-packages.txt`; local extension build paths are recorded only as installation-source information.

## Dual-GPU Scheduling

The script defaults to `GPU_IDS="0 1"` and `SEEDS="43 44"`, paired by position.
Each GPU trains at most one model at a time, and both GPUs traverse all configurations in order.
Each worker sets its own `CUDA_VISIBLE_DEVICES`; within the process, `cuda:0` refers to the assigned physical GPU.
There is no need to set `CUDA_VISIBLE_DEVICES` outside the script.

Each batch log directory contains `worker_0.log`, `worker_1.log`, individual experiment logs, and the completed `summary.tsv`.
`STOP_ON_ERROR=1` stops both queues when any experiment fails; by default, failures are recorded and execution continues.
Ctrl+C or terminating the main script cleans up both training process groups.

```bash
# Change GPU and seed assignments (paired by position)
GPU_IDS="0 1" SEEDS="43 44" bash shfile/train_landslide4sensen_two_seeds.sh
```

## Automatic Testing After Each Model Training Run

The current batch script only iterates over L4S. After successful training, each GPU queue immediately runs
`python test.py --exp_path <the model's actual output directory> --checkpoint best.pth --force`,
then starts the next model. Training failures do not test old weights; test failures are recorded in the summary.
Execution continues by default, while `STOP_ON_ERROR=1` stops both queues.

The test outputs `test_metrics.json` and `test_log.txt` are saved in the experiment directory; the batch directory also stores `*_test.log`.
The `stage` column in `summary.tsv` distinguishes `train` from `test`.
The Trainer avoids appending the seed directory suffix again when testing from the saved configuration.
Changes affect only subsequently started batches and do not modify already-running old scripts.

## Automatic Routing for Existing Checkpoints

Each L4S model and seed independently checks its output directory:

1. If `best.pth` exists: skip training and run `test.py --checkpoint best.pth --force`.
2. If only `latest.pth` exists: run `train.py --resume latest.pth`, then test the new `best.pth`.
3. If neither exists: train from scratch, then test `best.pth`.

Restoring `latest` is handled by the Trainer and includes the model, optimizer, scheduler, epoch/iteration, and `best_metric`.
An existing `best` checkpoint is tested directly even if it comes from incomplete training.
Testing still requires the corresponding `config.yaml`; corrupted files or missing saved configurations are recorded as failures, and weights are not overwritten.
The `stage` in `summary.tsv` is `train`, `resume`, or `test`. `DRY_RUN=1` shows each experiment's selected branch.

## L4S Results Summary

After each successful test, this directory is updated automatically: `/root/autodl-tmp/output/landslide/l4s_summary/`.

- `l4s_results.xlsx`: separate sheets for seed42, seed43, and seed44, with multi-level merged headers; two rows per model, one for Class0 and one for Class1.
- `l4s_seed<N>.csv` and `l4s_seed<N>.md`: CSV and Markdown files with the same content.

Columns are ordered as model, class, IoU, F1, Precision, Recall, OA, mIoU. The table excludes encoder and parameter count.
Metrics are percentages rounded to two decimal places. Class0 is non-landslide and Class1 is landslide; OA and mIoU are global metrics.
Parameter count remains auxiliary information in `model_info.json` and is not included in the results table.
Projects without `test_metrics.json` are shown as —; example values are never substituted for real results.

Manual update:

```bash
python scripts/summarize_l4s_results.py
```

Automatic updates use the script's `OUTPUT_ROOT`; `SUMMARY_ROOT` changes the table output directory.
The summary process uses a file lock and atomic replacement to prevent concurrent GPU jobs from overwriting incomplete tables.
Only newly started `test.py` processes record `model_info.json`; historical metrics without parameter counts can be regenerated by testing again.

