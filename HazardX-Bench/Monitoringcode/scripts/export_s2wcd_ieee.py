#!/usr/bin/env python3
"""Export a two-row task-specific / foundation-model efficiency figure."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter
from plot_s2wcd_efficiency import load_rows, COLORS, FAMILIES


def place_labels(ax, rows, key, scale, fontsize=7):
    """Place labels outside the full memory bubbles and previously placed text."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer).padded(-4)
    font = FontProperties(size=fontsize)
    circles = [(ax.transData.transform((r[key], r['throughput_pairs_s_mean'])),
                math.sqrt(r['peak_allocated_mib'] * scale) / 2 * fig.dpi / 72 + 4)
               for r in rows]
    used, annotations = [], []
    for r, ((px, py), radius) in zip(rows, circles):
        label = r['_name']
        width, height, descent = renderer.get_text_width_height_descent(label, font, False)
        width += 6
        height = fontsize * fig.dpi / 72 * 1.5
        candidates = []
        for distance in range(14, 500, 6):
            for angle in range(0, 360, 15):
                a = math.radians(angle)
                cx, cy = px + distance*math.cos(a), py + distance*math.sin(a)
                box = Bbox.from_bounds(cx-width/2, cy-height/2, width, height)
                if not bounds.contains(box.x0, box.y0) or not bounds.contains(box.x1, box.y1):
                    continue
                if any(box.overlaps(b) for b in used):
                    continue
                if any(math.hypot(max(box.x0-qx, 0, qx-box.x1),
                                  max(box.y0-qy, 0, qy-box.y1)) < qr
                       for ((qx, qy), qr) in circles):
                    continue
                candidates.append((distance + abs(cy-py)*.12, cx, cy, box))
            if candidates:
                break
        if not candidates:
            raise RuntimeError(f'No unobstructed label position: {label}')
        _, cx, cy, box = min(candidates, key=lambda c:c[0])
        used.append(box)
        annotations.append(ax.annotate(label, (r[key], r['throughput_pairs_s_mean']),
            xytext=((cx-px)*72/fig.dpi, (cy-py)*72/fig.dpi), textcoords='offset points',
            ha='center', va='center', fontsize=fontsize, color='#18232D',
            bbox=dict(facecolor='none', edgecolor='none', pad=.3),
            arrowprops=dict(arrowstyle='-', color='#8B969E', linewidth=.5,
                            shrinkA=2, shrinkB=radius*72/fig.dpi), zorder=5))
    return annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='output/efficiency/s2wcd_256_fp32_all29')
    p.add_argument('--output-dir')
    args = p.parse_args()
    root = Path(args.results)
    out = Path(args.output_dir) if args.output_dir else root / 'ieee_final'
    out.mkdir(parents=True, exist_ok=True)
    rows, versions, contended = load_rows(root, allow_contended=False, allow_mixed=True)
    assert len(rows) == 29 and not contended
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8,
                         'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .8})
    fig, axes = plt.subplots(1, 4, figsize=(14.32, 4.8), dpi=200)
    fig.subplots_adjust(left=.045, right=.99, top=.855, bottom=.24, wspace=.31)
    shapes = {tuple(r['input_shape_per_time']) for r in rows}
    assert shapes == {(1, 13, 256, 256)}, shapes
    groups = [rows[:12], rows[12:]]
    scale = .025
    annotations = []
    for i, group in enumerate(groups):
        for j, key in enumerate(['total_params_m', 'gops_fma1_per_pair']):
            ax = axes[i*2+j]
            for r in sorted(group, key=lambda r: -r['peak_allocated_mib']):
                c = COLORS[r['_family']]
                ax.scatter(r[key], r['throughput_pairs_s_mean'], s=r['peak_allocated_mib']*scale,
                           facecolors=to_rgba(c, .24), edgecolors=c, linewidths=.9, zorder=3)
            ax.set_xscale('log')
            xs = [r[key] for r in group]
            ax.set_xlim(min(xs)/1.9, max(xs)*2.1)
            ax.set_ylim(-12, 200)
            ax.set_yticks([0, 40, 80, 120, 160, 200])
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=[1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
            ax.tick_params(which='major', length=3, width=.8)
            ax.tick_params(which='minor', length=2, width=.6)
            ax.grid(axis='y', color='#DADADA', linewidth=.6)
            ax.grid(axis='x', which='major', color='#EBEBEB', linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_xlabel('Total parameters (M)' if j == 0 else 'Forward operations (G; FMA=1)', labelpad=5)
            ax.set_ylabel('Throughput (pairs/s)', labelpad=5)
            group_name = 'Task-specific' if i == 0 else 'FM'
            metric = 'Parameters' if j == 0 else 'Operations'
            ax.set_title(f'({chr(97+i*2+j)}) {group_name}\n{metric} versus throughput',
                         loc='left', fontsize=10, pad=9, linespacing=1.4)
            labels = place_labels(ax, group, key, scale, fontsize=7)
            annotations.append((ax, labels))
    # One horizontal legend row; wrapped labels keep the native paper width readable.
    legend_ax = fig.add_axes([.02, .025, .96, .105])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')
    labels = ['Task-specific\nCNN', 'Task-specific\nTransformer', 'Change\ndetection',
              'General vision\nFM', 'Earth observation\nFM']
    positions = [.005, .135, .295, .405, .55]
    for family, label, x in zip(FAMILIES, labels, positions):
        legend_ax.scatter(x+.006, .5, s=19, facecolors=to_rgba(COLORS[family], .24),
                          edgecolors=COLORS[family], linewidths=.8, clip_on=False)
        legend_ax.text(x+.021, .5, label, va='center', fontsize=8, linespacing=1.35)
    legend_ax.text(.72, .5, 'Bubble area:\npeak allocated GPU\nmemory (MiB)',
                   va='center', fontsize=8, linespacing=1.2)
    for x, memory in zip([.877, .925, .98], [500, 2000, 10000]):
        legend_ax.scatter(x, .62, s=memory*scale, facecolors='#DDDDDD',
                          edgecolors='#888888', linewidths=.8, clip_on=False)
        legend_ax.text(x, .17, f'{memory:,}', ha='center', va='center', fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = 0
    for ax, labels in annotations:
        boxes = [a.get_bbox_patch().get_window_extent(renderer) for a in labels]
        overlaps += sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    if overlaps:
        raise RuntimeError(f'Overlapping model labels: {overlaps}')
    stem = out / 'fig_efficiency_ieee'
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=600, facecolor='white')
    (out / 'figure_validation.json').write_text(json.dumps({
        'width_inches': 14.32, 'height_inches': 4.8, 'panel_layout': '1x4', 'png_dpi': 600,
        'plot_label_overlaps': int(overlaps), 'models': len(rows),
        'group_models': {'Task-specific': len(groups[0]), 'FM': len(groups[1])},
        'label_placement': 'excludes full bubble extents; transparent text background',
        'dataset': 'S2WCD', 'input_shape_per_time': [1, 13, 256, 256], 'temporal_images': 2,
        'error_bars': False, 'direct_model_labels': True,
        'bubble_area': 'peak_allocated_mib', 'software_groups': versions,
        'strict_same_framework_comparison': False}, indent=2))
    print(stem)

if __name__ == '__main__':
    main()
