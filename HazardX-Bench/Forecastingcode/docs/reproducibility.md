# Reproducibility

## Selected experiments

Each dataset–model has one selected formal seed. The catalog and checkpoint
manifest are keyed by dataset plus model slug; global IDs remain 01–11 with
dataset-specific omissions. FloodCastBench uses seed 42, FloodSimBench seed 1,
LandslideCast seed 0, and WildfireSpreadTS seed 0 except Prithvi seed 1. The
Wildfire Prithvi selection is the second experimental seed, not a newly trained
replacement. All seeds were read from actual checkpoint configurations.

`results/<dataset>/<model>/` preserves the original final evaluation JSONs.
`metrics.json` additionally joins those results to the checkpoint SHA256 and
selection metadata. `results/catalog.json` records each best-validation epoch.
Epoch numbering follows the original implementation and is not standardized
across datasets. The selected checkpoint need not be the final training epoch.

## Exact implementation boundaries

The common interface resolves filesystem paths and forwards the saved settings
to the original numerical implementation. Native files remain byte-identical;
`native_source_manifest.json` records their SHA256. Dataset/model factories,
losses, sampler rules, state dictionaries, normalization constants, padding,
overlap assembly, thresholds, and metric calculations remain the originals.
Formatting and explanatory comments were added to the new shared interface.

The YAML's `native_args`, `model_config`, and saved normalization reproduce the
selected run's configuration. Paths to datasets, original initialization weights,
official code, and new outputs are the intended user-controlled overrides.
Derived schedules are reconstructed by native validators, not approximated by
a generic optimizer/scheduler. Training checks `distributed.world_size` so that
a single-GPU command cannot silently change a multi-rank effective batch.

cuDNN deterministic/benchmark flags are configured only when the original
experiment explicitly specifies them. LandslideCast and WildfireSpreadTS keep
their native backend defaults; the shared wrapper does not add a new policy.

The FloodSim PDE checkpoint belongs to the four-GPU accelerated continuation
with validation every 10 epochs; its original run resumed an earlier single-GPU
snapshot at epoch 3. The public YAML reproduces the selected continuation's
settings. A new epoch 0 run of this final-phase configuration is not a claim of
bitwise reproduction of that historical two-phase training trajectory. Historical
resume lineage is retained in `experiment_lineage.json`; intermediate cursors
are not published as extra model seeds.

## Evaluation protocols

- **FloodCastBench:** fixed FP32 evaluation, physical depth reconstruction,
  nonnegative projection, original masked metrics, 128×128 windows/stride96,
  and original overlap assembly. Source scores macro-average Australia and
  Pakistan as in the native evaluator. Mozambique one-/two-day scores use only
  the three 576-state files: six 288-state segments for one day and three 576-state
  segments for two days. The 1728 directory is excluded from that protocol.
- **FloodSimBench:** FP32 evaluation of original 1024×1024 spatial crops/padding;
  12 physical channels; five historical states and six rain fields; original
  depth cleanup and masks. Complete evaluation predicts67 autoregressive steps
  and retains the original city aggregation. Training precision remains model
  specific in YAML. No patch-level score is relabeled as full-domain evaluation.
- **LandslideCast:** original 200×200 scenario grid and x/y transpose; static and
  initial conditioning; eight future depth/velocity frames. The fixed scenario
  ranges prevent train/validation/test overlap. Original relative losses and
  per-lead evaluation are retained, including the GPhyT Euler adapter and the
  model-specific FP32/FP16/BF16 recipe.
- **Landslide post-hoc cap200:** `test.py --metric-protocol cap200-pooled`
  reuses the original separate evaluator, including target masks at 200,
  thresholds 0.5/5, and pooled contingency counts. Saved post-hoc reports remain
  separate from native uncapped training-run metrics. The historical GPhyT
  `test_metrics_bounded_200_csi_0p5_5.json` averages per-scenario CSI instead of
  pooling, and is explicitly retained under that different definition. Those
  two CSI definitions must not be substituted for one another.
- **WildfireSpreadTS:** original year fold, normalization, feature expansion,
  crops, binary Dice loss, and next-day evaluation. The threshold is selected
  on validation and stored with the original results. The original stable-sort
  AP implementation, including tie handling, is preserved; it is not silently
  replaced by another library's AP definition. Training uses original BF16
  autocast; independent evaluation uses FP32.

`--max-batches` and `--max-steps` are explicit debugging limits. Their outputs
are not replacements for the saved complete-test results.

## Validation scope

See `validation_report.md` and `validation_results.json`. Every selected weight
was strictly loaded; genuine data inputs, predictions, and batch metrics were
compared between original modules and the release in separate processes. A
temporary optimizer step was executed. The smoke test uses a representative
batch of size1; it does not replace the original training batch/accumulation
recipe or prove long-run optimization convergence. Modified smoke-test weights
were discarded and original checkpoints were never overwritten.

FloodSim validation used bitwise-verified cached original raw1024 pixels because
the original GeoTIFF directory was empty. Original GeoTIFF file parsing remains
unverified in this release session. Wildfire validation used a small subset of
official TIFFs/HDF5, not its complete held-out set. No full retraining or full
benchmark reevaluation was performed. The Wildfire PDE batch had finite
gradients near1e-26 and no detectable parameter update; the original loss and
precision were retained and that case is marked with a warning.

## Checkpoint integrity and downloads

Only selected trained benchmark checkpoints are uploaded. Source data and
third-party pretrained initialization weights are excluded. A checkpoint larger
than the connector's 100 MiB limit is split into lossless 90 MiB byte parts. The
manifest records each part's exact byte count, local SHA256, and observed Drive
ID/link. The download helper checks every part and the restored whole-file
SHA256 before using `best.pth`.

Remote Drive metadata was read back to verify file IDs, sizes, and parent
folders. All 250 weight files (13 whole checkpoints plus 237 byte parts),
72 configuration/result files, and 23 parts manifests were then independently
downloaded through their public Drive links; every SHA256 matched the local
source. The API does not expose checksums, so this evidence comes from actual
downloaded bytes. All 23 local part streams also reconstruct their original
checkpoint hashes, and the public offline helper successfully restored and
reloaded a checkpoint. The 36 model folders contain exactly the expected files,
with no duplicate names or extra checkpoints. No sharing permissions were
changed by the release tools.
