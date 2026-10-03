# LANDREF change detection

These configurations formulate LANDREF as bi-temporal, multi-source landslide
change detection. Each branch uses ten Sentinel-2 bands, six Sentinel-1 SAR
features, and the shared DEM and slope channels:

`B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`

`gVV, gVH, COHVV, ALPHA, ANISOTROPY, ENTROPY`

`None_DEM, None_SLOPE`

The target is `None_MASK`. It marks landslide pixels, not every visual change
between acquisitions, so this is landslide-specific rather than generic change
detection.

- Native dual-input CD models receive two tensors with 18 channels each; each
  independently encoded branch therefore has access to the shared terrain.
- ChangeOS also retains two 18-channel branches because it splits the
  concatenated input internally.
- Segmentation backbones used as early-fusion CD baselines receive 18 channels
  on the pre branch (including DEM/slope) and 16 on the post branch. `TaskCD`
  concatenates them into 34 channels, so static terrain is included only once.
- S2 reflectance, SAR alpha, DEM, and slope are divided by 10000, 90, 5000,
  and 90 respectively. Other SAR features retain their native scale, and all
  input channels are clipped to `[0, 1]` after scaling.
- LANDREF chips are 128 x 128; model-specific patch divisibility settings are
  retained where necessary.

Example:

```bash
python train.py --config configs/landslide/landref_cd/cd_unet.yaml
python train.py --config configs/landslide/landref_cd/cd_bit_r50.yaml
```

Additional geospatial foundation-model early-fusion baselines are provided for
Prithvi-EO-2.0, SatMAE, SpectralGPT, DoFA+, AnySat, SkySense++, TerraMind,
Galileo, and Clay. Their configurations use the `cd_<model>.yaml` naming
scheme in this directory.
