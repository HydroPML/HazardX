# Environments

Validation used Python 3.11.16, PyTorch 2.5.0+cu124 and torchvision 0.20.0+cu124 on an RTX A6000. Installed package inventories are in environment_main.json and environment_pde.json. These are observations of the original environments, not a claim that every vendored example was executed.

The main model environment used transformers 4.46.2. The separate PDE environment used pdetransformer 0.1.11, diffusers 0.40.0 and transformers 4.42.3. Keeping them separate avoids replacing the PDE dependency stack.

A private rasterio 1.4.3 overlay supplied missing GeoTIFF-reader dependencies to the PDE environment during verification and was also in the shared validation launcher path. The base main environment had rasterio 1.4.4 installed; the release requirements select the validated 1.4.3 reader. The overlay is not packaged, and existing training environments were not modified.

Prithvi uses TerraTorch 0.99.5 at official revision ca289f0663a84952247213958cf3727a07359a87. The original editable working tree was checked against its Git index and had no modified tracked files. requirements.txt pins that archive revision and its original compatible Torch version is installed by the README command.

The audited Prithvi source/config beside the official initialization weight must be set up with scripts/setup_prithvi.py. No third-party weight is included in the release or uploaded to Drive.

Complete clean-environment installation and full distributed training were not rerun. Numerical checks were performed in the existing verified environments. Exact source/weight hashes and the two requirement files support independent installation and inspection.
