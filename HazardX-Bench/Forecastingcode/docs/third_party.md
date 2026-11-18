# Third-party provenance

One compatibility-only edit moves a newline join outside an optional upstream
PDE metric's `__repr__` f-string so the snapshot parses on Python 3.11. Its text
representation is unchanged. The exact before/after hashes are recorded in
`compatibility_changes.json`; no model, loss, or benchmark metric calculation
is changed.

Official backbone implementations are reused rather than replaced by compact
approximations. Audited local numerical-source hashes are in
`native_source_manifest.json`; vendored file hashes are in
`third_party_source_manifest.json`. License notices remain with each snapshot.

| Component | Original pinned source revision |
| --- | --- |
| ConvLSTM reference | [OpenSTL eecf8a3](https://github.com/chengtan9907/OpenSTL/tree/eecf8a3078f0a178dbc7b28723da20f94ce36985) |
| EarthFormer | [7732b03](https://github.com/amazon-science/earth-forecasting-transformer/tree/7732b03bdb366110563516c3502315deab4c2026) |
| PDE-Transformer | [850e09d](https://github.com/tum-pbs/pde-transformer/tree/850e09d352487922e4a853ee90a2624717769cc7) |
| DPOT | [dcd2f9a](https://github.com/thu-ml/DPOT/tree/dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17) |
| Poseidon | [b8fa28f](https://github.com/camlab-ethz/poseidon/tree/b8fa28f59bd7f7673323f28d11a12c6f3a215c61) |
| GPhyT | [df1527c](https://github.com/FloWsnr/General-Physics-Transformer/tree/df1527c4773ababc6f8af0d117fb5803fa3bf74b) |
| Prithvi-EO-2.0 | [7583112](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/tree/758311280f935d4775c05b26d792768bf1fda841) |
| TerraTorch registry | [ca289f0](https://github.com/torchgeo/terratorch/tree/ca289f0663a84952247213958cf3727a07359a87), installed original version 0.99.5 |

For WildfireSpreadTS, VideoMAEv2, and the auxiliary wildfire source snapshot,
an original immutable Git revision was not recoverable from the local archive;
their exact retained files are identified by SHA256 instead of assigning the
current upstream HEAD retroactively. Prithvi's companion source/config are
also explicitly SHA256-checked by its original adapter.

`pretrained_weights.json` records official Hugging Face repository revisions,
file paths, model-card license labels, and LFS SHA256 values observed during
preparation. These are download references, not bundled weights. The original
checkpoint loading reports/hash checks remain active. Renaming a downloaded
weight to the local filename in the README does not alter its content.
