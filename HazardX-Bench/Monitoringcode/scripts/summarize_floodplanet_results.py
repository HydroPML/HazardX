#!/usr/bin/env python3
"""Create seed43/seed44 FloodPlanet result tables from test_metrics.json files."""

from __future__ import annotations

import argparse
from pathlib import Path

import summarize_landref_cd_results as table


table.MODELS = [
    ("seg_unet", "UNet", ""),
    ("seg_segformer_b0", "Segformer", "MiT-B0"),
    ("seg_segformer_b1", "Segformer", "MiT-B1"),
    ("seg_segformer_b2", "Segformer", "MiT-B2"),
    ("seg_segformer_b3", "Segformer", "MiT-B3"),
    ("seg_segformer_b4", "Segformer", "MiT-B4"),
    ("seg_segformer_b5", "Segformer", "MiT-B5"),
    ("seg_sam2fpn_hiera_bp", "SAM2", "Sam2.1_hiera_base_plus"),
    ("seg_sam2fpn_hiera_s", "SAM2", "Sam2.1_hiera_small"),
    ("seg_sam2fpn_hiera_l", "SAM2", "Sam2.1_hiera_large"),
    ("seg_sam2fpn_hiera_t", "SAM2", "Sam2.1_hiera_tiny"),
    ("seg_dinov3dpt_vitb16_lvd", "DINOV3", "ViT-B16"),
    ("seg_dinov3dpt_vitl16_lvd", "DINOV3", "ViT-L16"),
    ("seg_prithvieo2", "Prithvi-EO-2.0", "prithvi-eo-2.0-300m"),
    ("seg_satmae", "SatMAE", "base-patch16"),
    ("seg_spectralgpt", "SpectralGPT", ""),
    ("seg_dofav2_base", "DoFA+", "base_e150"),
    ("seg_anysat_base", "AnySat", "base"),
    ("seg_skysensepp", "SkySense++", ""),
    ("seg_terramind", "TerraMind", "terramind-v1-base"),
    ("seg_galileo_nano", "Galileo", "galileo-nano"),
    ("seg_clay_v15_large", "Clay", "large"),
]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=repo / "output" / "floodplanet")
    parser.add_argument(
        "--table-dir", type=Path, default=repo / "output" / "floodplanet" / "tables"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()
    table.generate(
        args.output_root,
        args.table_dir,
        list(dict.fromkeys(args.seeds)),
        title_prefix="FloodPlanet",
        file_prefix="floodplanet",
    )


if __name__ == "__main__":
    main()
