#!/usr/bin/env python3
"""Create LANDREF-CD result tables from test_metrics.json files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


MODELS = [
    ("unet", "UNet", ""),
    ("segformer_b0", "Segformer", "MiT-B0"),
    ("segformer_b1", "Segformer", "MiT-B1"),
    ("segformer_b2", "Segformer", "MiT-B2"),
    ("segformer_b3", "Segformer", "MiT-B3"),
    ("segformer_b4", "Segformer", "MiT-B4"),
    ("segformer_b5", "Segformer", "MiT-B5"),
    ("sam2_fpn_bplus", "SAM2", "Sam2.1_hiera_base_plus"),
    ("sam2_fpn_s", "SAM2", "Sam2.1_hiera_small"),
    ("sam2_fpn_l", "SAM2", "Sam2.1_hiera_large"),
    ("sam2_fpn_t", "SAM2", "Sam2.1_hiera_tiny"),
    ("dinov3_dpt_vitb_lvd", "DINOV3", "ViT-B16"),
    ("dinov3_dpt_vitl_lvd", "DINOV3", "ViT-L16"),
    ("prithvieo2", "Prithvi-EO-2.0", "prithvi-eo-2.0-300m"),
    ("satmae_b", "SatMAE", "base-patch16"),
    ("spectralgpt", "SpectralGPT", ""),
    ("dofav2_base", "DoFA+", "base_e150"),
    ("anysat_base", "AnySat", "base"),
    ("skysensepp", "SkySense++", ""),
    ("terramind", "TerraMind", "terramind-v1-base"),
    ("galileo_nano", "Galileo", "galileo-nano"),
    ("clay_v15_large", "Clay", "large"),
    # These two rows are requested in the table but currently have no config.
    ("olmoearth_nano", "OLMOEarth", "olmoearth_nano"),
    ("satlas_tiny", "SatlasPretrain", "satlas_tiny"),
    ("siamcrnn_r50", "SiamCRNN", "变化检测"),
    ("dsifn", "DSIFN", ""),
    ("bit_r50", "BIT-CD", ""),
    ("changemamba", "ChangeMamba", ""),
]

HEADERS = ["model", "encoder", "class", "IoU", "F1", "Precision", "Recall", "OA", "mIoU"]
ARRAY_METRICS = ["per_class_iou", "per_class_f1", "per_class_precision", "per_class_recall"]


def load_metrics(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def percent(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


def rows_for_seed(output_root: Path, seed: int):
    rows = []
    ready = 0
    for dirname, model, encoder in MODELS:
        metrics = load_metrics(output_root / f"{dirname}_seed{seed}" / "test_metrics.json")
        if metrics:
            ready += 1
        for class_id in (0, 1):
            values = []
            for array_name in ARRAY_METRICS:
                array = metrics.get(array_name, [])
                values.append(percent(array[class_id]) if len(array) > class_id else None)
            rows.append([
                model if class_id == 0 else "",
                encoder if class_id == 0 else "",
                f"Class{class_id}",
                *values,
                percent(metrics.get("OA")) if class_id == 0 else None,
                percent(metrics.get("mIoU")) if class_id == 0 else None,
            ])
    return rows, ready


def write_csv(path: Path, title: str, rows) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model", "encoder", "class", title, "", "", "", "", ""])
        writer.writerow(["", "", "", "IoU", "F1", "Precision", "Recall", "OA", "mIoU"])
        writer.writerows([["" if value is None else value for value in row] for row in rows])
    temporary.replace(path)


def generate(
    output_root: Path,
    table_dir: Path,
    seeds: list[int],
    title_prefix: str = "Landslide4sense",
    file_prefix: str = "landref_cd",
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    table_dir.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for seed in seeds:
        rows, ready = rows_for_seed(output_root, seed)
        title = f"{title_prefix}（seed{seed}）"
        write_csv(table_dir / f"{file_prefix}_seed{seed}.tsv", title, rows)

        sheet = workbook.create_sheet(f"seed{seed}")
        for column, header in enumerate(HEADERS[:3], 1):
            sheet.cell(1, column, header)
            sheet.merge_cells(start_row=1, start_column=column, end_row=2, end_column=column)
        sheet.cell(1, 4, title)
        sheet.merge_cells(start_row=1, start_column=4, end_row=1, end_column=9)
        for column, header in enumerate(HEADERS[3:], 4):
            sheet.cell(2, column, header)
        for row in rows:
            sheet.append(row)

        # Each encoder and its global metrics span Class0/Class1 rows.
        for row_index in range(3, len(rows) + 3, 2):
            sheet.merge_cells(start_row=row_index, start_column=2, end_row=row_index + 1, end_column=2)
            for column in (8, 9):
                sheet.merge_cells(start_row=row_index, start_column=column, end_row=row_index + 1, end_column=column)
        # Merge the model label across all of its encoders.
        start = 3
        while start <= len(rows) + 2:
            model = sheet.cell(start, 1).value
            end = start + 1
            cursor = start + 2
            while cursor <= len(rows) + 2 and sheet.cell(cursor, 1).value == model:
                end = cursor + 1
                cursor += 2
            sheet.merge_cells(start_row=start, start_column=1, end_row=end, end_column=1)
            start = end + 1

        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"
                if cell.row <= 2:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill("solid", fgColor="D9EAF7")
        widths = [20, 28, 12, 12, 12, 12, 12, 12, 12]
        for column, width in enumerate(widths, 1):
            sheet.column_dimensions[chr(64 + column)].width = width
        sheet.freeze_panes = "D3"
        print(f"seed {seed}: {ready}/{len(MODELS)} rows have test results")

    handle, temporary_name = tempfile.mkstemp(suffix=".xlsx", dir=table_dir)
    os.close(handle)
    try:
        workbook.save(temporary_name)
        os.replace(temporary_name, table_dir / f"{file_prefix}_results.xlsx")
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(f"Table: {table_dir / f'{file_prefix}_results.xlsx'}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=repo / "output" / "landref")
    parser.add_argument("--table-dir", type=Path, default=repo / "output" / "tables")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()
    generate(args.output_root, args.table_dir, list(dict.fromkeys(args.seeds)))


if __name__ == "__main__":
    main()
