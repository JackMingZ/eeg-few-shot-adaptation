"""Export small, inspectable result tables; never copies EEG, weights or images."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from src.experiment_profiles import PROFILES


SUBJECTS = list(range(1, 10))
SEEDS = [7, 17, 27]
BUDGETS = [1, 3, 5, 10]
RUNS = {(target, seed) for target in SUBJECTS for seed in SEEDS}
METHOD_BUDGETS = {('source', 0), ('csp_lda', 0)} | {
    (method, budget) for method in ('head', 'full', 'scale_head', 'affine_head')
    for budget in BUDGETS
}
TRAINING = {'subjects': SUBJECTS, 'epochs': 100, 'patience': 20,
            'adapt_epochs': 50, 'adapt_lr': 0.001, 'input_scale': 1.0,
            'budgets': BUDGETS}
PROFILE = {key: getattr(PROFILES['paper'], key) for key in
           ('samples', 'sampling_rate', 'epoch_start', 'epoch_end', 'temporal_kernel', 'dropout')}


def validate_matrix(configs, frame, plan):
    """Require the exact published design, not just the same number of rows."""
    for key, expected in {**PROFILE, **TRAINING, 'targets': SUBJECTS, 'seeds': SEEDS}.items():
        if plan.get(key) != expected:
            raise ValueError(f'Batch plan mismatch: {key}')
    if plan.get('euclidean_alignment', False):
        raise ValueError('Batch plan must not enable alignment')
    identities = [(cfg.get('target'), cfg.get('seed')) for cfg in configs]
    if len(identities) != len(RUNS) or set(identities) != RUNS:
        raise ValueError('Configuration target/seed matrix mismatch')
    for cfg in configs:
        expected = {**PROFILE, **TRAINING, 'status': 'complete', 'skip_csp': False,
                    'input_scale_microvolt': 1.0,
                    'validation_subject': cfg['target'] % 9 + 1,
                    'adaptation_batchnorm': 'frozen source statistics for all methods'}
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f'Run configuration mismatch: {key}')
        # Early snapshots predate the optional EA flag; absence means disabled.
        if cfg.get('euclidean_alignment', False):
            raise ValueError('Run must not enable alignment')
    keys = ['target', 'seed', 'method', 'budget_per_class']
    if not set(keys) <= set(frame.columns):
        raise ValueError('Missing metric identity columns')
    expected_rows = {(t, s, m, b) for t, s in RUNS for m, b in METHOD_BUDGETS}
    actual_rows = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or actual_rows != expected_rows:
        raise ValueError('Metric method/budget matrix mismatch')


def export(source, output):
    paths = sorted(source.glob('target_*/metrics.json'))
    if not paths:
        raise ValueError('No completed runs')
    rows, provenance = [], []
    configs = []
    for path in paths:
        config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        configs.append(config)
        run_rows = json.loads(path.read_text(encoding='utf-8'))
        if path.parent.name != f'target_{config["target"]:02d}_seed_{config["seed"]}':
            raise ValueError('Directory identity differs from configuration')
        if any((r.get('target'), r.get('seed')) != (config['target'], config['seed']) for r in run_rows):
            raise ValueError('Metric identity differs from its run configuration')
        rows.extend(run_rows)
        provenance.append({'run': path.parent.name,
                           'metrics_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'configuration': {k: v for k, v in config.items() if k not in ('output', 'data_root')}})
    frame = pd.DataFrame(rows)
    plan = json.loads((source/'batch_plan.json').read_text(encoding='utf-8'))
    validate_matrix(configs, frame, plan)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'all_metrics.csv', index=False)
    subjects = frame.groupby(['target','method','budget_per_class'], as_index=False)[['accuracy','macro_f1','kappa']].mean()
    subjects.to_csv(output/'subject_means.csv', index=False)
    subjects.groupby(['method','budget_per_class'])[['accuracy','macro_f1','kappa']].agg(['mean','std','count']).to_csv(output/'summary.csv')
    frame.groupby(['method','budget_per_class'])[['parameters','trainable_parameters','adaptation_seconds','inference_seconds','peak_cuda_bytes']].mean().to_csv(output/'cost_summary.csv')
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (output/'configuration.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'Exported {len(frame)} rows; no trial data or weights copied.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/real/paper_reproduction'))
    parser.add_argument('--output', type=Path, default=Path('reports/paper'))
    args = parser.parse_args()
    export(args.source, args.output)
