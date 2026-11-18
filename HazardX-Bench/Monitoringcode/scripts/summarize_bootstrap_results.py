"""Collect bootstrap outputs and existing external seed43/44 test metrics."""
import csv
import json
import os
import shutil
from pathlib import Path

DATASETS = ('s2wcd', 'floodplanet', 'landref')
EXTERNAL_MODELS = ('olmoearth_nano', 'satlas_tiny')


def load_results(root, dataset):
    """Normalize existing test metrics in memory; never fabricate bootstrap outputs."""
    entries = []
    for directory in sorted((root / dataset).iterdir()):
        bootstrap = directory / 'test_bootstrap.json'
        metrics = directory / 'test_metrics.json'
        if bootstrap.is_file():
            entries.append((directory.name, json.loads(bootstrap.read_text())))
        elif metrics.is_file():
            data = json.loads(metrics.read_text())
            rows = []
            for metric, key in [('mIoU', 'mIoU'), ('CSI', 'IoU_1'), ('F1', 'F1_1')]:
                rows.append(dict(metric=metric, estimate=data[key], ci_lower=None, ci_upper=None,
                                 bootstrap_variance=None, zero_denominator_replicates=None))
            entries.append((directory.name, dict(n_test_tiles={'s2wcd': 8, 'floodplanet': 47, 'landref': 44}[dataset],
                benchmark_role='diagnostic', repetitions=None, seed=None,
                training_seed=int(directory.name.rsplit('_seed', 1)[1]),
                metrics=rows, source='existing_test_metrics', checkpoint=data.get('checkpoint'))))
    return entries


def import_external_results(project, destination):
    external = Path(os.environ.get('EXTERNAL_PROJECT_ROOT', str(project.parent / 'diasterother'))) / 'outputs'
    for dataset in DATASETS:
        prefix = 'landref_cd' if dataset == 'landref' else dataset
        for model in EXTERNAL_MODELS:
            # Old seed42 results must not enter active summaries.
            legacy = destination / dataset / f'{prefix}_{model}'
            if legacy.exists():
                archive = destination / 'legacy_seed42' / dataset / legacy.name
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    raise ValueError(f'Duplicate legacy result: {legacy}; archive already exists: {archive}')
                legacy.rename(archive)
            for seed in (43, 44):
                name = f'{prefix}_{model}_seed{seed}'
                source = external / name / 'test_metrics.json'
                data = json.loads(source.read_text())
                assert data['split'] == 'test', source
                assert all(key in data for key in ('mIoU', 'IoU_1', 'F1_1')), source
                target = destination / dataset / name
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / source.name)
                (target / 'result_source.json').write_text(json.dumps(dict(
                    source=str(source), training_seed=seed, model=model,
                    bootstrap_available=(target / 'test_bootstrap.json').exists(),
                    note=('Bootstrap intervals and variance computed from original test samples.'
                          if (target / 'test_bootstrap.json').exists() else
                          'Imported existing test metrics; missing bootstrap CI/variance remain blank.')), indent=2) + '\n')


def main():
    project = Path(__file__).resolve().parents[1]
    destination = project / 'output/bootstrap_results'
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for source in (project / 'output' / dataset).glob('*/test_bootstrap.json'):
            if any(model in source.parent.name for model in EXTERNAL_MODELS):
                continue
            target = destination / dataset / source.parent.name
            if (target / source.name).exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for name in ('test_bootstrap.json', 'test_bootstrap.csv',
                         'test_tile_confusion_matrices.npy', 'test_bootstrap_distributions.npz'):
                if (source.parent / name).exists():
                    shutil.copy2(source.parent / name, target / name)
    import_external_results(project, destination)
    fields = ['dataset', 'experiment', 'n_test_tiles', 'benchmark_role',
              'repetitions', 'seed', 'metric', 'estimate', 'ci_lower', 'ci_upper',
              'bootstrap_variance', 'zero_denominator_replicates']
    count = 0
    with (destination / 'bootstrap_summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for experiment, result in load_results(destination, dataset):
                for metric in result['metrics']:
                    row = dict(dataset=dataset, experiment=experiment,
                               **{key: result[key] for key in ('n_test_tiles', 'benchmark_role', 'repetitions', 'seed')},
                               **metric)
                    writer.writerow({key: row[key] for key in fields})
                    count += 1
    print(f'Saved {count} metric rows: {destination / "bootstrap_summary.csv"}')


if __name__ == '__main__':
    main()
