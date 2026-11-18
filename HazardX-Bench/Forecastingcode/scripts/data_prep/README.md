# Dataset preparation

Download from the official sources linked below. Keep the existing event and
scenario splits. The release does not regenerate random splits, resample spatial
grids, or substitute an easier forecast horizon.

## FloodCastBench

Sources: [official archive](https://zenodo.org/records/14017092) and
[generation project](https://github.com/HydroPML/FloodCastBench).

The selected experiments consume trusted `.pt` trajectories with shape
`[T,H,W,3]`, ordered water depth (m), rainfall, and terrain. The loader also
accepts a mapping containing `data`, optional masks/times, and grid spacing.
It reads DEM rasters separately and uses the original centered-DEM transform.
Keep the original nodata values and the existing 288-state trajectories:

```text
data/FloodCastBench/
├── Australia/288/{train,valid,test}/*.pt
├── Pakistan/288/{train,valid,test}/*.pt
├── UK_30/288/train/*.pt
├── UK/288/train/*.pt
├── Test/Mozambique/{576,1728}/*.pt
└── DEM/                         # Original geographical DEM rasters
```

`UK` is excluded from the selected training protocol; retain it if executing the
original split audit. Training uses the original 21-scene native tile trajectory
sampler. Model inputs are 128×128; evaluation uses 128×128 windows with stride
96 and the original overlap assembly at every autoregressive step. Within-source
validation/test use Australia and Pakistan. Mozambique is geographically held
out and evaluated separately.

Set `--data-root` to the directory above and `--dem-root` to its DEM directory.
The local archive contains prepared `.pt` trajectories, but the exact historical
raw-archive-to-`.pt` conversion script was not located. Public users must obtain
the matching prepared trajectories from the dataset maintainers; this release
does not claim a new converter is equivalent. The original generation project
provides simulation generation, which is different from this missing conversion.

## FloodSimBench

Source: [official FloodSimBench dataset](https://huggingface.co/datasets/chrimerss/FloodSimBench).
Use the native **1 m** data. Do not use the unrelated 3 m preparation script.

```text
data/FloodSimBench/
├── HOU001/
│   ├── HOU001_DEM.tif
│   ├── HOU001_<P>mm_WaterDepth_30000.tif
│   ├── ...
│   └── HOU001_<P>mm_WaterDepth_2160000.tif
└── ...                         # Other original domain directories
```

The parser requires 72 observed depth rasters at five-minute intervals. It
ignores `6hr_max` products. Train cities are HOU/DAL/OKC/NYC/ATL/ORL; validation
uses AUS/MIA; testing uses LA/SF. The saved normalizer records the original
train-only water/rain statistics and centered terrain statistics.

The physical input is six rain fields, DEM, and five preceding water depths
(12 channels in the exact order built by `build_context_input`). The primary
output is the next depth field. The original domain-balanced training sampler
selects one scenario per each of 17 training regions per round, rather than
iterating every possible scenario/window each epoch. Spatial windows are
1024×1024 with the original padding and masks. Complete evaluation uses the
unchanged spatial crop/padding policy and 67 autoregressive steps; it must not be
reinterpreted as a 128×128 benchmark.

During release validation the raw GeoTIFF directory was unavailable. A private
lossless 1024×1024 HDF5 cache of actual original rasters was used to check model
inputs and calculations. This cache is not a supported replacement for public
GeoTIFF preparation and is not included in the ZIP.

## LandslideCast

Sources: [official repository](https://github.com/HydroPML/Landslidecast) and
[associated paper](https://doi.org/10.1029/2023EA003417).

Place the canonical file at `data/LandslideCast/2d_t10.h5`. Each dataset `xN`
must have shape `[200,200,9,4]`, optionally with a leading singleton. Channels
are friction angle, elevation, depth, and velocity. The native loader preserves
the original x/y transpose and fixed scaling constants (elevation 2133, depth
141). Scenario IDs define the immutable split:

| Split | Scenario IDs | Independent scenarios | Future frame targets |
| --- | --- | ---: | ---: |
| Train | `x1`–`x10000` | 10,000 | 80,000 |
| Val | `x10001`–`x11000` | 1,000 | 8,000 |
| Test | `x11001`–`x12000` | 1,000 | 8,000 |

Each scenario contains nine positions at 0,10,…,80 seconds. Initial/static
conditions condition the model; the eight future depth/velocity frames are
targets. Temporal broadcasting in a spatiotemporal model does not expose those
future targets. GPhyT retains its original Euler initialization and lead chunk.

The current official README has no verified download URL for this exact canonical
HDF5 file. Obtain it from the dataset authors through the official project.
This missing public data-access dependency remains a release limitation.

## WildfireSpreadTS

Sources: [official archive](https://zenodo.org/records/8006177) and
[official preprocessing](https://github.com/SebastianGer/WildfireSpreadTS).

Extract the original archive into `data/WildfireSpreadTS/Data/`, keeping
`<year>/<fire_id>/<YYYY-MM-DD>.tif`. Convert it with the unchanged official
converter, through the provided isolated wrapper:

```bash
python scripts/data_prep/prepare_wildfire.py \
  --data-dir data/WildfireSpreadTS/Data \
  --target-dir data/WildfireSpreadTS/HDF5
```

The converter writes `<year>/<fire_id>.hdf5`, dataset `data`, and original date
and geolocation attributes without changing raster values. Selected YAMLs use
this HDF5 backend. The original fold 0 trains on 2018/2019, validates on 2020,
and tests on 2021. All normalization statistics come from training years.

The original 23 raw features expand to 40 model channels through the original
feature selection/categorical encoding. Five preceding days form one input;
the next-day active fire is the target. Keep native 375 m resolution, daily
sampling, 128×128 target-independent training crops/augmentations, and the
original deterministic evaluation crop. No test-selected classification
threshold is introduced.
