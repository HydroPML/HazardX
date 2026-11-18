#!/usr/bin/env python3
"""Create seed43/seed44 S2WCD result tables from test_metrics.json files."""

from __future__ import annotations

import argparse
from pathlib import Path

import summarize_landref_cd_results as table


table.MODELS = [
    ("cd_unet", "UNet", ""),
    ("cd_segformer_b0", "Segformer", "MiT-B0"),
    ("cd_segformer_b1", "Segformer", "MiT-B1"),
    ("cd_segformer_b2", "Segformer", "MiT-B2"),
    ("cd_segformer_b3", "Segformer", "MiT-B3"),
    ("cd_segformer_b4", "Segformer", "MiT-B4"),
    ("cd_segformer_b5", "Segformer", "MiT-B5"),
    ("cd_sam2_fpn_bplus", "SAM2", "Sam2.1_hiera_base_plus"),
    ("cd_sam2_fpn_s", "SAM2", "Sam2.1_hiera_small"),
    ("cd_sam2_fpn_l", "SAM2", "Sam2.1_hiera_large"),
    ("cd_sam2_fpn_t", "SAM2", "Sam2.1_hiera_tiny"),
    ("cd_dinov3_dpt_vitb_lvd", "DINOV3", "ViT-B16"),
    ("cd_dinov3_dpt_vitl_lvd", "DINOV3", "ViT-L16"),
    ("cd_prithvieo2", "Prithvi-EO-2.0", "prithvi-eo-2.0-300m"),
    ("cd_satmae_b", "SatMAE", "base-patch16"),
    ("cd_spectralgpt", "SpectralGPT", ""),
    ("cd_dofav2_base", "DoFA+", "base_e150"),
    ("cd_anysat_base", "AnySat", "base"),
    ("cd_skysensepp", "SkySense++", ""),
    ("cd_terramind", "TerraMind", "terramind-v1-base"),
    ("cd_galileo_nano", "Galileo", "galileo-nano"),
    ("cd_clay_v15_large", "Clay", "large"),
    ("cd_siamcrnn_r50", "SiamCRNN", "变化检测"),
    ("cd_dsifn", "DSIFN", ""),
    ("cd_bit_r50", "BIT-CD", ""),
    ("cd_changemamba", "ChangeMamba", ""),
    ("cd_changeos_r50", "ChangeOS", "ResNet-50"),
]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=repo / "output" / "s2wcd")
    parser.add_argument("--table-dir", type=Path, default=repo / "output" / "s2wcd" / "tables")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()
    table.generate(
        args.output_root,
        args.table_dir,
        list(dict.fromkeys(args.seeds)),
        title_prefix="S2WCD",
        file_prefix="s2wcd",
    )


if __name__ == "__main__":
    main()
