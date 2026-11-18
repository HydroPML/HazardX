# Release validation report

All 36 selected checkpoints were strictly loaded and tested on physical GPU 4 (NVIDIA RTX A6000, 49,140 MiB; driver 535.183.06; PyTorch 2.5.0 / CUDA 12.4). Existing GPU 5/6 jobs were left untouched.

Original and release modules ran in separate processes. Every test used genuine inputs at the original model resolution, eval-mode forward and native metrics, followed by one temporary training step. All 36 input tensors match exactly, and predictions/metrics satisfy the declared tolerances (maximum prediction absolute difference 0; maximum metric difference 0). Tolerances were rtol=1e-4 / atol=1e-5 for predictions and 1e-4 for metric differences.

**Training-step result:** all 36 executed an optimizer step with finite gradients; 35 produced a detected parameter change. Wildfire PDE had maximum gradient 9.794496999224083e-27 and parameter change=False on this batch. The finite-gradient/no-change case is recorded as a warning; loss and precision were not changed to force a parameter update.

All 36 public training dispatches were parsed and checked without starting a complete training run. Independent test.py smoke evaluations passed for FloodCastBench, LandslideCast, and WildfireSpreadTS. The official Wildfire converter processed three real events successfully. The optional original Landslide cap200 evaluator passed a real ConvLSTM batch with internal metric cross-check delta 0.0.

| Dataset | ID | Model | Seed | Data / forward / native metrics / parity | Optimizer executed | Parameter change |
| --- | --- | --- | ---: | --- | --- | --- |
| FloodCastBench | 01 | UNet | 42 | PASS | PASS | PASS |
| FloodCastBench | 02 | ConvLSTM-Scratch | 42 | PASS | PASS | PASS |
| FloodCastBench | 04 | FNO | 42 | PASS | PASS | PASS |
| FloodCastBench | 05 | EarthFormer-Scratch-fp32-lr3e-4 | 42 | PASS | PASS | PASS |
| FloodCastBench | 06 | PDE-T-SC-S-PT-FT | 42 | PASS | PASS | PASS |
| FloodCastBench | 07 | DPOT-S-PT-FT | 42 | PASS | PASS | PASS |
| FloodCastBench | 08 | Poseidon-T-PT-FT | 42 | PASS | PASS | PASS |
| FloodCastBench | 09 | GPhyT-S-PT-FT | 42 | PASS | PASS | PASS |
| FloodCastBench | 11 | Prithvi-EO2-300M-PT-FT | 42 | PASS | PASS | PASS |
| FloodSimBench | 01 | UNet | 1 | PASS | PASS | PASS |
| FloodSimBench | 04 | FNO | 1 | PASS | PASS | PASS |
| FloodSimBench | 02 | ConvLSTM-Scratch | 1 | PASS | PASS | PASS |
| FloodSimBench | 05 | EarthFormer-Scratch | 1 | PASS | PASS | PASS |
| FloodSimBench | 06 | PDE-T-SC-S-PT-FT | 1 | PASS | PASS | PASS |
| FloodSimBench | 07 | DPOT-S-PT-FT | 1 | PASS | PASS | PASS |
| FloodSimBench | 08 | Poseidon-T-PT-FT | 1 | PASS | PASS | PASS |
| FloodSimBench | 09 | GPhyT-S-PT-FT | 1 | PASS | PASS | PASS |
| FloodSimBench | 11 | Prithvi-EO2-300M-PT-FT | 1 | PASS | PASS | PASS |
| LandslideCast | 02 | ConvLSTM-Scratch | 0 | PASS | PASS | PASS |
| LandslideCast | 04 | FNO3D-Scratch | 0 | PASS | PASS | PASS |
| LandslideCast | 06 | PDE-T-SC-S-PT-FT | 0 | PASS | PASS | PASS |
| LandslideCast | 08 | Poseidon-T-PT-FT | 0 | PASS | PASS | PASS |
| LandslideCast | 01 | UNet3D-Scratch | 0 | PASS | PASS | PASS |
| LandslideCast | 05 | EarthFormer-Scratch | 0 | PASS | PASS | PASS |
| LandslideCast | 07 | DPOT-S-PT-FT | 0 | PASS | PASS | PASS |
| LandslideCast | 11 | Prithvi-EO2-300M-PT-FT | 0 | PASS | PASS | PASS |
| LandslideCast | 09 | GPhyT-S-PT-FT-Euler | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 01 | UNet2D-Scratch | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 03 | UTAE-Scratch | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 02 | ConvLSTM-Scratch | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 04 | FNO2D-Scratch | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 05 | EarthFormer-Scratch | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 06 | PDE-T-SC-S-PT-FT | 0 | PASS | PASS | No change; warning |
| WildfireSpreadTS | 07 | DPOT-S-PT-FT | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 10 | VideoMAEv2-PT-FT | 0 | PASS | PASS | PASS |
| WildfireSpreadTS | 11 | Prithvi-EO2-300M-PT-FT | 1 | PASS | PASS | PASS |

## Data and scope limits

- FloodCastBench: real prepared trajectories and DEMs; representative 128×128 windows. Formal evaluation uses FP32 and original overlap assembly. All nine comparisons were rerun after checking this fixed evaluation precision.
- FloodSimBench: genuine raw 1024×1024 HDF5 cache tiles, with recorded bitwise equality to the original rasters; exact native rain/context/terrain/normalization/cleanup. The original GeoTIFF directory was empty, so GeoTIFF I/O and a complete public test.py rollout were not exercised. No resized or synthetic data was substituted.
- LandslideCast: real canonical HDF5 scenarios, full 200×200 grids, all eight forecast frames, original transpose and scale.
- WildfireSpreadTS: genuine official Zenodo TIFF subset converted to HDF5, original five-day/40-channel128×128 windows and native crop/augmentation. This is a small official subset, not full-test coverage.
- No complete retraining, full held-out score regeneration, distributed training run, or long-run resume was performed. The original saved final metrics are preserved separately in results/. One-step checks do not prove identical convergence across hardware.
- Case-level results, input shapes, training losses, and compared metrics are in validation_results.json. Full private logs remain with the release preparation records; temporary logs/data/weights are excluded from the public code ZIP.

## Remaining public-release dependencies

- Canonical Landslide HDF5 public access and the historical FloodCast raw-to-trajectory converter were not found; prepared local data was usable and tested.
- Project-wide code/benchmark-weight licensing and missing DPOT/Poseidon source grants require maintainer resolution. Existing upstream notices are preserved.
- Drive upload is limited to 100 MiB per file; larger checkpoints use lossless 90 MiB byte parts and verified reconstruction, explicitly labeled in the manifest.
