#!/usr/bin/env python3
"""Export Class0/Class1 L4S test results as merged Excel, CSV and Markdown tables."""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import yaml

HEADERS = ['model', 'class', 'IoU', 'F1', 'Precision', 'Recall', 'OA', 'mIoU']
METRICS = [('per_class_iou', 'IoU'), ('per_class_f1', 'F1'),
           ('per_class_precision', 'Precision'), ('per_class_recall', 'Recall')]


def load_json(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f'Cannot read {path}: {exc}')
        return {}


def model_label(config):
    model = config['model']; name = model['name']; k = model.get('kwargs', {})
    aliases = {'sam2fpn': 'sam2', 'dinov3dpt': 'dinov3', 'satmaedpt': 'satmae',
               'prithvieo2': 'prithvi-eo-2.0', 'dofav2': 'dofa+', 'skysensepp': 'skysense++'}
    name = aliases.get(name, name)
    encoder = k.get('backbone', k.get('model_size', k.get('variant', '')))
    if name == 'sam2':
        match = re.search(r'hiera_(b\+|s|l|t)', k.get('config_file', ''))
        encoder = {'b+': 'hiera-B+', 's': 'hiera-s', 'l': 'hiera-l', 't': 'hiera-t'}.get(match[1] if match else '', '')
    elif name == 'segformer': encoder = str(encoder).replace('mit_', 'mit-')
    elif name == 'dinov3': encoder = {'dinov3_vitb16': 'vit-b16', 'dinov3_vitl16': 'vit-l16'}.get(encoder, encoder)
    elif name == 'prithvi-eo-2.0' and '300m' in str(encoder): encoder = 'vit-l'
    elif name == 'satmae': encoder = 'base-p16' if 'base-patch16' in str(encoder) else encoder
    elif name == 'dofa+' and encoder == 'base' and 'e150' in k.get('checkpoint_path', ''): encoder = 'base-e150'
    elif name == 'clay': encoder = str(encoder).removeprefix('clay-v1.5-')
    elif name == 'galileo': encoder = str(encoder).removeprefix('galileo-')
    elif name == 'terramind': encoder = str(encoder).removeprefix('terramind-')
    return name, encoder or '—'


def number(value, percent=False):
    if value is None:
        return '—'
    value = float(value)
    if not math.isfinite(value):
        return '—'
    return f'{value * (100 if percent else 1):.2f}'


def build_rows(configs, experiments, seed):
    rows = []; ready = 0
    for config_path in configs:
        fallback = yaml.safe_load(config_path.read_text())
        # Trainer uses the bare model directory only for the default seed 42.
        dirname = config_path.stem + (f'_seed{seed}' if seed != 42 else '')
        exp = experiments / dirname
        saved = exp / 'config.yaml'
        config = yaml.safe_load(saved.read_text()) if saved.is_file() else fallback
        name, _ = model_label(config)
        metrics = load_json(exp / 'test_metrics.json')
        if metrics: ready += 1
        for class_id in (0, 1):
            values = []
            for array_key, scalar_key in METRICS:
                array = metrics.get(array_key)
                value = array[class_id] if isinstance(array, list) and len(array) > class_id else metrics.get(f'{scalar_key}_{class_id}')
                values.append(number(value, percent=True))
            rows.append([name if class_id == 0 else '', f'Class{class_id}', *values,
                         number(metrics.get('OA'), True) if class_id == 0 else '',
                         number(metrics.get('mIoU'), True) if class_id == 0 else ''])
    return rows, ready


def atomic_text(path, content):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(content, encoding='utf-8-sig' if path.suffix == '.csv' else 'utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(config_root, experiments, output, seeds):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    configs = sorted(list(config_root.glob('*.yaml')) + list(config_root.glob('*.yml')))
    if not configs:
        raise ValueError(f'No YAML configs found under {config_root}')
    order = ['unet', 'segformer', 'sam2', 'dinov3', 'prithvi-eo-2.0', 'satmae',
             'spectralgpt', 'dofa+', 'anysat', 'skysense++', 'terramind', 'galileo', 'clay']
    def sort_key(path):
        name, _ = model_label(yaml.safe_load(path.read_text()))
        return (order.index(name) if name in order else len(order), path.stem)
    configs.sort(key=sort_key)
    output.mkdir(parents=True, exist_ok=True)
    # Serialize concurrent GPU workers and atomically publish complete files.
    with (output / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        workbook = Workbook(); workbook.remove(workbook.active)
        for seed in seeds:
            rows, ready = build_rows(configs, experiments, seed)
            title = f'Landslide4sense（seed{seed}）'
            csv_data = io.StringIO(); writer = csv.writer(csv_data)
            writer.writerow(HEADERS); writer.writerows(rows)
            atomic_text(output / f'l4s_seed{seed}.csv', csv_data.getvalue())
            md = [f'# {title}', '', '指标单位：%，保留两位小数。',
                  'Class0=非滑坡，Class1=滑坡；OA 和 mIoU 为全局指标。— 表示测试结果尚未生成。', '',
                  '| ' + ' | '.join(HEADERS) + ' |', '| ' + ' | '.join(['---'] * len(HEADERS)) + ' |']
            md.extend('| ' + ' | '.join(row) + ' |' for row in rows)
            atomic_text(output / f'l4s_seed{seed}.md', '\n'.join(md) + '\n')
            sheet = workbook.create_sheet(f'seed{seed}')
            for col, header in enumerate(HEADERS[:2], 1):
                sheet.cell(1, col, header); sheet.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
            sheet.cell(1, 3, title); sheet.merge_cells('C1:H1')
            for col, header in enumerate(HEADERS[2:], 3): sheet.cell(2, col, header)
            for row in rows:
                sheet.append([float(v) if v not in ('', '—') and col not in (0, 1) else v for col, v in enumerate(row)])
            for row in range(3, len(rows) + 3, 2):
                for col in (1, 7, 8):
                    sheet.merge_cells(start_row=row, start_column=col, end_row=row + 1, end_column=col)
            for row in sheet:
                for cell in row:
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                    if isinstance(cell.value, float): cell.number_format = '0.00'
                    if cell.row <= 2:
                        cell.font = Font(bold=True); cell.fill = PatternFill('solid', fgColor='D9EAF7')
            for col, width in zip('ABCDEFGH', [22, 12, 13, 13, 13, 13, 13, 13]): sheet.column_dimensions[col].width = width
            sheet.freeze_panes = 'C3'
            print(f'seed {seed}: {ready}/{len(configs)} model test results; {output / f"l4s_seed{seed}.md"}')
        fd, temporary = tempfile.mkstemp(suffix='.xlsx', dir=output); os.close(fd)
        try:
            workbook.save(temporary); os.replace(temporary, output / 'l4s_results.xlsx')
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f'Excel: {output / "l4s_results.xlsx"}')


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, default=repo / 'configs/landslide/l4s')
    parser.add_argument('--exp-root', type=Path, default=Path('/root/autodl-tmp/output/landslide/l4s'))
    parser.add_argument('--output-dir', type=Path, default=Path('/root/autodl-tmp/output/landslide/l4s_summary'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    generate(args.config_root, args.exp_root, args.output_dir, list(dict.fromkeys(args.seeds)))


if __name__ == '__main__':
    main()
