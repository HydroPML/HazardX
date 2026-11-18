# LANDREF configs

These configs use 22 channels from the official HDF5 files: ten PRE1
Sentinel-2 bands, ten POST1 Sentinel-2 bands, DEM, and slope. Reflectance is
scaled by 10000, DEM by 5000, and slope by 90 in the dataset adapter.

AnySat and PrithviEO2 use the complete 22-channel tensor through configurable
input projections. Clay and DOFAV2 use the 20 optical channels only, with the
correct Sentinel-2 wavelength list repeated for pre- and post-event imagery;
DEM and slope are excluded from wavelength embedding.

Galileo uses the post-event Sentinel-2 bands plus real slope and DEM. LANDREF
does not contain B1 or B9, so those two channels are explicitly zero-filled.

The default independent test split is `testind_n3_s1s2.h5`. Change it to
`testspt_n3_s1s2.h5` in a config when evaluating the spatial test split.
