"""Export IEEE-ready descriptive bootstrap table and vector figure from JSON results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1] / 'output/bootstrap_results'
DATASETS = [('s2wcd', 'S2-WCD'), ('floodplanet', 'FloodPlanet'), ('landref', 'LANDREF')]
METRICS = ['mIoU', 'CSI', 'F1']


def main():
    (ROOT / 'figures').mkdir(exist_ok=True)
    (ROOT / 'latex').mkdir(exist_ok=True)
    stats = []
    missing = {}
    widths = {}
    counts = {}
    for key, label in DATASETS:
        missing[key] = sum(1 for p in (ROOT / key).glob('*/test_metrics.json') if not (p.parent / 'test_bootstrap.json').exists())
        entries = [json.loads(p.read_text()) for p in sorted((ROOT / key).glob('*/test_bootstrap.json'))]
        if not entries:
            raise ValueError(f'No results: {key}')
        sizes = {x['n_test_tiles'] for x in entries}
        assert len(sizes) == 1, (key, sizes)
        assert all(x['repetitions'] == 10000 and x['confidence'] == .95 for x in entries)
        counts[key] = (sizes.pop(), len(entries))
        for metric in METRICS:
            rows = [next(m for m in x['metrics'] if m['metric'] == metric) for x in entries]
            w = np.array([100 * (m['ci_upper'] - m['ci_lower']) for m in rows])
            v = np.array([m['bootstrap_variance'] for m in rows])
            assert np.all(np.isfinite(w)) and np.all(w >= 0)
            q1, median, q3 = np.quantile(w, [.25, .5, .75])
            widths[key, metric] = w
            stats.append(dict(dataset=label, n_test=counts[key][0], checkpoints=len(entries), metric=metric,
                              ci_width_median_pp=float(median), ci_width_q1_pp=float(q1),
                              ci_width_q3_pp=float(q3), bootstrap_variance_median=float(np.median(v))))
    with (ROOT / 'bootstrap_uncertainty_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    tex = [r'\begin{table*}[t]', r'\centering',
           r'\caption{Bootstrap uncertainty on small test sets. $N$ is the number of original test samples and $K$ the number of fitted checkpoints. CI widths are in percentage points (pp); Q1 and Q3 denote the 25th and 75th percentiles across checkpoints. Variances use the original $[0,1]$ metric scale. These are descriptive summaries across checkpoints, not confidence intervals for an average model or statistical comparisons between models.}',
           r'\label{tab:bootstrap_uncertainty}', r'\small',
           r'\begin{tabular}{lrrlrrr}', r'\toprule',
           r'Dataset & $N$ & $K$ & Metric & Median CI width (pp) & Q1--Q3 (pp) & Median variance ($\times10^{-3}$) \\', r'\midrule']
    for i, row in enumerate(stats):
        if i and i % 3 == 0:
            tex.append(r'\midrule')
        lead = [row['dataset'], str(row['n_test']), str(row['checkpoints'])] if i % 3 == 0 else ['', '', '']
        tex.append(' & '.join(lead + [row['metric'], f"{row['ci_width_median_pp']:.2f}",
                   f"{row['ci_width_q1_pp']:.2f}--{row['ci_width_q3_pp']:.2f}",
                   f"{1000*row['bootstrap_variance_median']:.3f}"]) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (ROOT / 'latex/bootstrap_uncertainty_table.tex').write_text('\n'.join(tex) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65), sharey=True)
    colors = ['#4477AA', '#EE9944', '#228866']
    for ax, metric in zip(axes, METRICS):
        boxes = ax.boxplot([widths[key, metric] for key, _ in DATASETS], widths=.48,
                           patch_artist=True, showfliers=True,
                           medianprops={'color': 'black', 'linewidth': 1.2},
                           flierprops={'marker': 'o', 'markersize': 2.5, 'markerfacecolor': 'white'})
        for patch, color in zip(boxes['boxes'], colors):
            patch.set(facecolor=color, alpha=.7, linewidth=.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f'{label}\n$N={counts[key][0]}$\n$K={counts[key][1]}$' for key, label in DATASETS], fontsize=7)
        ax.set_title(metric)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, 5 * np.ceil(max(w.max() for w in widths.values()) / 5))
    axes[0].set_ylabel('95% bootstrap CI width (pp)')
    fig.tight_layout(pad=.7, w_pad=1.0)
    fig.savefig(ROOT / 'figures/bootstrap_ci_widths.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'figures/bootstrap_ci_widths.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    inclusion = r'''% Preamble:
% \usepackage{graphicx}
% \usepackage{booktabs}
% Copy the table .tex and the figure .pdf into your Overleaf/project directory.
\input{bootstrap_uncertainty_table.tex}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{bootstrap_ci_widths.pdf}
\caption{Distribution of 95\% percentile bootstrap confidence interval widths across fitted checkpoints on three small test sets ($B=10{,}000$). Boxes span the first and third quartiles, center lines indicate medians, whiskers extend to the most extreme observations within 1.5 interquartile ranges, and circles mark observations beyond the whiskers. $N$ denotes original test samples; $K$ denotes fitted checkpoints, with training seeds evaluated separately. Widths are expressed in percentage points (pp). The plot describes test-set sampling uncertainty conditional on each checkpoint and does not establish model rankings.}
\label{fig:bootstrap_ci_widths}
\end{figure*}

Table~\ref{tab:bootstrap_uncertainty} and
Fig.~\ref{fig:bootstrap_ci_widths} summarize the substantial
sampling uncertainty on the three small test sets. The median
95\% CI widths for mIoU/CSI/F1 are 9.65/13.06/7.67 percentage
points on S2-WCD, 14.90/22.58/19.68 on FloodPlanet, and
8.93/16.58/10.87 on LANDREF. These distributions are descriptive
summaries across fitted checkpoints rather than confidence
intervals for a model-averaged score. Accordingly, we treat
these datasets as diagnostic evidence and avoid definitive
model-ranking conclusions based on their point estimates.
'''
    inclusion += '\n% Existing external test metrics without bootstrap CIs excluded from uncertainty figures: ' + str(missing) + '\n'
    import re
    for index, (_, label) in enumerate(DATASETS):
        medians = '/'.join(f"{stats[index * 3 + j]['ci_width_median_pp']:.2f}" for j in range(3))
        inclusion = re.sub(r'\d+\.\d+/\d+\.\d+/\d+\.\d+( percentage\s+points on ' + re.escape(label) + ')',
                           lambda match: medians + match.group(1), inclusion)
        # The later datasets omit the repeated unit in the prose.
        inclusion = re.sub(r'\d+\.\d+/\d+\.\d+/\d+\.\d+( on ' + re.escape(label) + ')',
                           lambda match: medians + match.group(1), inclusion)
    (ROOT / 'latex/bootstrap_visuals_include.tex').write_text(inclusion)
    methods = ROOT / 'latex/bootstrap_methods.tex'
    if methods.exists():
        text = methods.read_text()
        sizes = [str(counts[key][0]) for key, _ in DATASETS]
        ks = [str(counts[key][1]) for key, _ in DATASETS]
        text = re.sub(r'The analysis covers [\d, and]+ fitted checkpoints, respectively;',
                      'The bootstrap analysis covers ' + ', '.join(ks[:2]) + ', and ' + ks[2] + ' fitted checkpoints, respectively;', text)
        text = re.sub(r'The bootstrap analysis covers [\d, and]+ fitted checkpoints, respectively;',
                      'The bootstrap analysis covers ' + ', '.join(ks[:2]) + ', and ' + ks[2] + ' fitted checkpoints, respectively;', text)
        for index, (_, label) in enumerate(DATASETS):
            medians = '/'.join(f"{stats[index * 3 + j]['ci_width_median_pp']:.2f}" for j in range(3))
            text = re.sub(r'\d+\.\d+/\d+\.\d+/\d+\.\d+((?: percentage\s+points)? on ' + re.escape(label) + ')',
                          lambda match: medians + match.group(1), text)
        marker = '% Imported external results note'
        text = text.split(marker)[0].rstrip()
        if any(missing.values()):
            note = 'External checkpoints without bootstrap intervals are marked with dashes and excluded from uncertainty-width summaries and figures.'
        else:
            note = 'OLMoEarth Nano and Satlas Tiny use training seeds 43 and 44 only. Their bootstrap intervals and variances use the same original-sample resampling protocol and are included in all tables and uncertainty figures.'
        text += '\n\n' + marker + '\n' + note + '\n'
        methods.write_text(text)
    print(json.dumps(stats, indent=2))
    print('Saved table, CSV, PDF, PNG, and LaTeX inclusion snippet.')


if __name__ == '__main__':
    main()
