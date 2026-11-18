# Selected formal experiments

One selected best-validation checkpoint per dataset–model. Epochs retain native numbering. Metrics below are copied from original full-test artifacts; they are not the release small-batch scores. Checkpoint filenames/downloads/SHA256/configs are joined in checkpoints/manifest.csv.

| Dataset | ID | Model | Seed | Best checkpoint epoch | Original main test metrics |
| --- | --- | --- | ---: | ---: | --- |
| FloodCastBench | 01 | UNet | 42 | 24 | RMSE=1.4282488; MAE=0.74785944 |
| FloodCastBench | 02 | ConvLSTM-Scratch | 42 | 19 | RMSE=0.98739332; MAE=0.49657727 |
| FloodCastBench | 04 | FNO | 42 | 89 | RMSE=0.11119338; MAE=0.063825368 |
| FloodCastBench | 05 | EarthFormer-Scratch-fp32-lr3e-4 | 42 | 49 | RMSE=0.85742593; MAE=0.46407674 |
| FloodCastBench | 06 | PDE-T-SC-S-PT-FT | 42 | 4 | RMSE=1.4337883; MAE=0.90026528 |
| FloodCastBench | 07 | DPOT-S-PT-FT | 42 | 34 | RMSE=1.0001756; MAE=0.48912405 |
| FloodCastBench | 08 | Poseidon-T-PT-FT | 42 | 94 | RMSE=0.45853522; MAE=0.21791731 |
| FloodCastBench | 09 | GPhyT-S-PT-FT | 42 | 74 | RMSE=0.066794252; MAE=0.026766502 |
| FloodCastBench | 11 | Prithvi-EO2-300M-PT-FT | 42 | 14 | RMSE=1.2313929; MAE=0.69976246 |
| FloodSimBench | 01 | UNet | 1 | 14 | RMSE=0.12378675; MAE=0.066810103 |
| FloodSimBench | 04 | FNO | 1 | 34 | RMSE=0.10836735; MAE=0.03094599 |
| FloodSimBench | 02 | ConvLSTM-Scratch | 1 | 89 | RMSE=0.16248582; MAE=0.054431889 |
| FloodSimBench | 05 | EarthFormer-Scratch | 1 | 74 | RMSE=0.088697761; MAE=0.027766737 |
| FloodSimBench | 06 | PDE-T-SC-S-PT-FT | 1 | 69 | RMSE=0.063856189; MAE=0.011378274 |
| FloodSimBench | 07 | DPOT-S-PT-FT | 1 | 24 | RMSE=0.081432303; MAE=0.012353057 |
| FloodSimBench | 08 | Poseidon-T-PT-FT | 1 | 44 | RMSE=0.19845009; MAE=0.057089909 |
| FloodSimBench | 09 | GPhyT-S-PT-FT | 1 | 79 | RMSE=0.059291411; MAE=0.021172905 |
| FloodSimBench | 11 | Prithvi-EO2-300M-PT-FT | 1 | 34 | RMSE=0.071789788; MAE=0.022488206 |
| LandslideCast | 02 | ConvLSTM-Scratch | 0 | 59 | uncapped RMSE_h=9.6446363; RelL2_h=0.69134784 |
| LandslideCast | 04 | FNO3D-Scratch | 0 | 99 | uncapped RMSE_h=8.5686648; RelL2_h=0.52096753 |
| LandslideCast | 06 | PDE-T-SC-S-PT-FT | 0 | 89 | uncapped RMSE_h=8.0641516; RelL2_h=0.29349992 |
| LandslideCast | 08 | Poseidon-T-PT-FT | 0 | 79 | uncapped RMSE_h=7.982119; RelL2_h=0.27307022 |
| LandslideCast | 01 | UNet3D-Scratch | 0 | 99 | uncapped RMSE_h=10.142342; RelL2_h=0.9915743 |
| LandslideCast | 05 | EarthFormer-Scratch | 0 | 99 | uncapped RMSE_h=8.0438654; RelL2_h=0.27699192 |
| LandslideCast | 07 | DPOT-S-PT-FT | 0 | 89 | uncapped RMSE_h=8.2158361; RelL2_h=0.37415412 |
| LandslideCast | 11 | Prithvi-EO2-300M-PT-FT | 0 | 59 | uncapped RMSE_h=8.0989754; RelL2_h=0.305975 |
| LandslideCast | 09 | GPhyT-S-PT-FT-Euler | 0 | 99 | uncapped RMSE_h=8.1885138; RelL2_h=0.37451348 |
| WildfireSpreadTS | 01 | UNet2D-Scratch | 0 | 100 | AP=0.4769344; F1=0.58792002 |
| WildfireSpreadTS | 03 | UTAE-Scratch | 0 | 90 | AP=0.33162552; F1=0.27576764 |
| WildfireSpreadTS | 02 | ConvLSTM-Scratch | 0 | 60 | AP=0.50910634; F1=0.57108836 |
| WildfireSpreadTS | 04 | FNO2D-Scratch | 0 | 65 | AP=0.474655; F1=0.57365144 |
| WildfireSpreadTS | 05 | EarthFormer-Scratch | 0 | 15 | AP=0.36050621; F1=0.50736698 |
| WildfireSpreadTS | 06 | PDE-T-SC-S-PT-FT | 0 | 85 | AP=0.47923917; F1=0.56735429 |
| WildfireSpreadTS | 07 | DPOT-S-PT-FT | 0 | 90 | AP=0.44481778; F1=0.56894115 |
| WildfireSpreadTS | 10 | VideoMAEv2-PT-FT | 0 | 85 | AP=0.35074961; F1=0.51734009 |
| WildfireSpreadTS | 11 | Prithvi-EO2-300M-PT-FT | 1 | 95 | AP=0.44702885; F1=0.51915642 |

All original additional metrics remain in results/catalog.json and per-model JSON files. Landslide cap200 and mean-/pooled-CSI post-hoc reports are labeled separately; uncapped scores must not be replaced by those values without changing the stated evaluation definition.
