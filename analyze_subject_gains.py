"""Paired subject-level bootstrap and Wilcoxon summaries for one locked run."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def analyze(root: Path, seed: int = 20261010, samples: int = 20000) -> pd.DataFrame:
    metrics = pd.read_csv(root / 'all_metrics.csv')
    subject = metrics.groupby(['target', 'method', 'budget_per_class'], as_index=False).accuracy.mean()
    source = subject[subject.method == 'source'][['target', 'accuracy']].rename(columns={'accuracy': 'source_accuracy'})
    rng = np.random.default_rng(seed)
    rows = []
    for budget in sorted(subject.loc[subject.budget_per_class > 0, 'budget_per_class'].unique()):
        for method in ['head', 'full', 'scale_head', 'affine_head']:
            part = subject[(subject.method == method) & (subject.budget_per_class == budget)].merge(source, on='target')
            delta = (part.accuracy - part.source_accuracy).to_numpy()
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)])
            p = float(wilcoxon(delta, alternative='two-sided').pvalue) if np.any(delta != 0) else 1.0
            rows.append({
                'method': method,
                'budget_per_class': int(budget),
                'mean_accuracy_gain': float(delta.mean()),
                'median_accuracy_gain': float(np.median(delta)),
                'improved_subjects': int((delta > 0).sum()),
                'n_subjects': int(len(delta)),
                'bootstrap_ci_95_low': float(np.quantile(boot, .025)),
                'bootstrap_ci_95_high': float(np.quantile(boot, .975)),
                'wilcoxon_p_unadjusted': p,
            })
    result = pd.DataFrame(rows)
    result.to_csv(root / 'paired_subject_statistics.csv', index=False)
    return result


def compare_methods(root: Path, seed: int = 20261011, samples: int = 20000) -> pd.DataFrame:
    """Paired comparisons among methods, with Holm correction across all tests."""
    metrics = pd.read_csv(root / 'all_metrics.csv')
    subject = metrics.groupby(['target', 'method', 'budget_per_class'], as_index=False).accuracy.mean()
    comparisons = [('head', 'full'), ('head', 'scale_head'), ('head', 'affine_head'),
                   ('full', 'scale_head'), ('full', 'affine_head'), ('scale_head', 'affine_head')]
    rng = np.random.default_rng(seed)
    rows = []
    for budget in sorted(subject.loc[subject.budget_per_class > 0, 'budget_per_class'].unique()):
        for method_a, method_b in comparisons:
            a = subject[(subject.method == method_a) & (subject.budget_per_class == budget)][['target', 'accuracy']]
            b = subject[(subject.method == method_b) & (subject.budget_per_class == budget)][['target', 'accuracy']]
            pair = a.merge(b, on='target', suffixes=('_a', '_b'))
            delta = (pair.accuracy_a - pair.accuracy_b).to_numpy()
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)])
            p = float(wilcoxon(delta, alternative='two-sided').pvalue) if np.any(delta != 0) else 1.0
            rows.append({'budget_per_class': int(budget), 'method_a': method_a, 'method_b': method_b,
                         'mean_accuracy_difference_a_minus_b': float(delta.mean()),
                         'median_accuracy_difference_a_minus_b': float(np.median(delta)),
                         'subjects_a_better': int((delta > 0).sum()), 'n_subjects': int(len(delta)),
                         'bootstrap_ci_95_low': float(np.quantile(boot, .025)),
                         'bootstrap_ci_95_high': float(np.quantile(boot, .975)),
                         'wilcoxon_p_unadjusted': p})
    result = pd.DataFrame(rows)
    # Holm step-down family-wise correction over the complete planned comparison family.
    order = np.argsort(result.wilcoxon_p_unadjusted.to_numpy())
    pvals = result.wilcoxon_p_unadjusted.to_numpy()
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adjusted[idx] = running
    result['wilcoxon_p_holm'] = adjusted
    result.to_csv(root / 'paired_method_comparisons.csv', index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--bootstrap-samples', type=int, default=20000)
    parser.add_argument('--method-comparisons', action='store_true')
    args = parser.parse_args()
    result = compare_methods(Path(args.root), args.seed, args.bootstrap_samples) if args.method_comparisons else analyze(Path(args.root), args.seed, args.bootstrap_samples)
    print(result.to_string(index=False))
