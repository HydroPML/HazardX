"""Descriptive mean (sample variance) across training seeds; CI endpoints are summarized, not pooled."""
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from summarize_bootstrap_results import load_results

ROOT = Path(__file__).resolve().parents[1] / 'output/bootstrap_results'
METRICS = ('mIoU', 'CSI', 'F1')


def model_name(name, dataset):
    name = re.sub(r'_seed\d+$', '', name)
    for prefix in (dataset + '_', 'landref_cd_', 'cd_', 'seg_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def moments(values):
    if any(value is None for value in values):
        return None, None
    return statistics.mean(values), statistics.variance(values) if len(values) > 1 else None


def cell(mean, variance, scientific=False):
    if mean is None:
        return '--'
    fmt = '.2e' if scientific else '.2f'
    low = '--' if variance is None else format(variance, fmt)
    return r'\shortstack{' + format(mean, fmt) + r'\\(' + low + ')}'


def main():
    out = ROOT / 'latex'
    out.mkdir(exist_ok=True)
    original = out / 'bootstrap_checkpoint_tables.tex'
    archive = out / 'bootstrap_checkpoint_tables_per_seed.tex'
    if original.exists() and not archive.exists():
        archive.write_text(original.read_text())
    lines = ['% Requires graphicx. First line: mean; second line: (sample variance) across training seeds.',
             '% CI endpoints summarized separately; these are NOT CIs for the seed-averaged score.']
    csvrows = []
    summary = []
    checkpoint_lines = ['% Checkpoint-level bootstrap results; training seeds evaluated separately.']
    for dataset, title in [('s2wcd', 'S2-WCD'), ('floodplanet', 'FloodPlanet'), ('landref', 'LANDREF')]:
        groups = defaultdict(list)
        for name, data in load_results(ROOT, dataset):
            groups[model_name(name, dataset)].append((name, data))
        models = sorted(groups)
        entries = sorted(entry for group in groups.values() for entry in group)
        for start in range(0, len(entries), 24):
            checkpoint_lines += [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
                r'\caption{' + title + ' checkpoint-level bootstrap results (part ' + str(start // 24 + 1) +
                r'). Scores and 95\% CI endpoints are percentages; variances use the $[0,1]$ scale. Training seeds are evaluated separately.}',
                r'\begin{tabular}{lcccccc}', r'\hline',
                r'Checkpoint & mIoU [95\% CI] & Var. & CSI [95\% CI] & Var. & F1 [95\% CI] & Var. \\', r'\hline']
            for name, data in entries[start:start + 24]:
                cells = [r'\texttt{' + name.replace('_', r'\_') + '}']
                for metric in METRICS:
                    m = next(m for m in data['metrics'] if m['metric'] == metric)
                    interval = (f"[{m['ci_lower']*100:.2f}, {m['ci_upper']*100:.2f}]"
                                if m['ci_lower'] is not None else '[--]')
                    variance = f"{m['bootstrap_variance']:.3e}" if m['bootstrap_variance'] is not None else '--'
                    cells += [f"{m['estimate']*100:.2f} {interval}", variance]
                checkpoint_lines.append(' & '.join(cells) + r' \\')
            checkpoint_lines += [r'\hline', r'\end{tabular}', r'\end{table*}', '']
        summary.append(f'{title}: {len(models)} models, seed counts {[len(groups[m]) for m in models]}')
        for start in [0]:
            table_start = len(lines)
            lines += [r'\begin{table*}[t]', r'\centering',
                '\\caption{' + title + ' bootstrap results summarized across available training seeds. Each cell reports the mean on the first line and sample variance across training seeds in parentheses on the second line. Scores and CI endpoints are percentages, with between-seed variances in squared percentage points. Bootstrap variances use the $[0,1]$ metric scale; their second-line values are between-seed variances of bootstrap variances. The lower and upper endpoints of the seed-specific 95\\% bootstrap CIs are summarized separately; their means do not constitute a 95\\% CI for the seed-averaged score. A dash indicates unavailable bootstrap statistics or insufficient checkpoints for between-seed variance. External models use training seeds 43 and 44; bootstrap statistics are conditional on each fitted checkpoint.}',
                '\\label{tab:bootstrap_' + dataset + '_seeds}',
                r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.25}',
                r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lccc ccc ccc}', r'\hline',
                r'Model & \multicolumn{3}{c}{mIoU} & \multicolumn{3}{c}{CSI} & \multicolumn{3}{c}{F1} \\',
                r' & Estimate & CI endpoints [L, U] & Bootstrap var. & Estimate & CI endpoints [L, U] & Bootstrap var. & Estimate & CI endpoints [L, U] & Bootstrap var. \\', r'\hline']
            for model in models:
                entries = groups[model]
                seeds = [data.get('training_seed') if data.get('training_seed') is not None
                         else (re.search(r'_seed(\d+)$', name).group(1)
                               if re.search(r'_seed(\d+)$', name) else 'unspecified')
                         for name, data in entries]
                assert len(set(map(str, seeds))) == len(entries), (dataset, model, seeds)
                cells = [model.replace('_', r'\_')]
                for metric in METRICS:
                    metrics = [next(m for m in data['metrics'] if m['metric'] == metric) for _, data in entries]
                    values = {key: moments([m[key] * (1 if key == 'bootstrap_variance' else 100) if m[key] is not None else None for m in metrics])
                              for key in ('estimate', 'ci_lower', 'ci_upper', 'bootstrap_variance')}
                    cells.append(cell(*values['estimate']))
                    lo, hi = values['ci_lower'], values['ci_upper']
                    spread = '--' if lo[1] is None else f'{lo[1]:.2f}, {hi[1]:.2f}'
                    cells.append('--' if lo[0] is None else r'\shortstack{[' + f'{lo[0]:.2f}, {hi[0]:.2f}' + r']\\(' + spread + ')}')
                    cells.append(cell(*values['bootstrap_variance'], scientific=True))
                    row = dict(dataset=title, model=model, n_seeds=len(entries), metric=metric)
                    for key, (mean, variance) in values.items():
                        row[key + '_mean'] = mean
                        row[key + '_variance'] = variance
                    csvrows.append(row)
                lines.append(' & '.join(cells) + r' \\')
            lines += [r'\hline', r'\end{tabular}%', '}', r'\end{table*}', '']
            (out / f'bootstrap_{dataset}_mean_var_table.tex').write_text('\n'.join(lines[table_start:]) + '\n')
    archive.write_text('\n'.join(checkpoint_lines) + '\n')
    text = '\n'.join(lines) + '\n'
    original.write_text(text)
    (out / 'bootstrap_model_mean_var_tables.tex').write_text(text)
    with (ROOT / 'bootstrap_model_mean_var.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    print('\n'.join(summary))
    print(f'Saved {len(csvrows)} model-metric rows; checkpoint-level tables updated.')


if __name__ == '__main__':
    main()
