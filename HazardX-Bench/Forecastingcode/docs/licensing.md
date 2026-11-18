# Licensing and attribution

This archive preserves original source notices and does not grant a new blanket
license. Code, datasets, pretrained initialization, and trained benchmark weights
have separate ownership and terms. The maintainers must publish the project's
own code/checkpoint license before describing it as unrestricted open source.
A repository-wide license was not found in the audited HazardX project.

## Third-party source

| Component | Source | License evidence in the retained snapshot |
| --- | --- | --- |
| EarthFormer | [amazon-science/earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer) | Apache-2.0; `src/models/third_party/earthformer/LICENSE` |
| PDE-Transformer | [tum-pbs/pde-transformer](https://github.com/tum-pbs/pde-transformer) | Apache-2.0; `pde-transformer/LICENSE.txt` |
| GPhyT | [FloWsnr/General-Physics-Transformer](https://github.com/FloWsnr/General-Physics-Transformer) | MIT; `GPhyT/LICENSE` |
| WildfireSpreadTS | [SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) | MIT; `WildfireSpreadTS/LICENSE` |
| VideoMAEv2 | [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) | MIT; `VideoMAEv2/LICENSE` |
| Wildfire forecasting components | [Retained upstream snapshot](../src/models/third_party/wildfire-forecasting-main/README.md) | MIT; `wildfire-forecasting-main/LICENSE` |
| Prithvi official companion source | [ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | Apache-2.0 source header; common license text retained in `PrithviEO2/LICENSE` |
| TerraTorch dependency | [torchgeo/terratorch](https://github.com/torchgeo/terratorch) | Apache-2.0 in the original dependency; pinned source revision documented |
| DPOT | [thu-ml/DPOT](https://github.com/thu-ml/DPOT) | No source LICENSE found in this snapshot or repository root during audit; redistribution terms require clarification |
| Poseidon | [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) | No source LICENSE found in this snapshot or repository root during audit; redistribution terms require clarification |
| Landslide FNO source | [HydroPML/Landslidecast](https://github.com/HydroPML/Landslidecast) | No license grant is inferred from public availability |

The DPOT and Poseidon weight model-card licenses are not treated as automatic
licenses for their separate source repositories. Retained code is provided for
authorship/provenance review; resolve these missing source grants before an
unrestricted public redistribution. This package is not a legal permission
statement on behalf of the upstream authors.

## Pretrained weights

The official model-card labels observed during preparation are:

| Initialization | Model card / terms |
| --- | --- |
| PDE-Transformer SC-S | [MIT](https://huggingface.co/thuerey-group/pde-transformer) |
| DPOT-S | [Apache-2.0](https://huggingface.co/hzk17/DPOT) |
| Poseidon-T | [CC-BY-NC-4.0](https://huggingface.co/camlab-ethz/Poseidon-T); noncommercial restriction |
| GPhyT-S | [MIT](https://huggingface.co/flwi/Physics-Foundation-Model) |
| VideoMAEv2 | [Apache-2.0 model-card label](https://huggingface.co/OpenGVLab/VideoMAE2) |
| Prithvi-EO-2.0 | [Apache-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) |

No third-party initialization weights are included in this ZIP or reuploaded to
Drive. Cite the respective models and honor their applicable terms when using
benchmark models initialized from them.

## Data and benchmark checkpoints

Use the licenses attached to each official dataset record:
[FloodCastBench](https://zenodo.org/records/14017092),
[FloodSimBench](https://huggingface.co/datasets/chrimerss/FloodSimBench),
[LandslideCast](https://github.com/HydroPML/Landslidecast), and
[WildfireSpreadTS](https://zenodo.org/records/8006177). No dataset is redistributed
here. The absence of a verified canonical Landslide HDF5 download/license is
listed as an unresolved data-access dependency.

The uploaded benchmark files are the selected trained experiment snapshots.
Their availability does not itself establish a new checkpoint license or
remove upstream initialization restrictions. The project maintainers must
provide the intended benchmark-checkpoint grant and citations.
