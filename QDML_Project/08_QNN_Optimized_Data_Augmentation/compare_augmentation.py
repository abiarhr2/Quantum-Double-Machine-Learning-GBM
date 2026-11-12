import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_qnn_augmented import iteration_keys, load_selection, record_path, validate_result
from config import BOOTSTRAPS, METHODS, run_directory
from data_utils import check_reference_folds, load_data
from storage import checked, exclusive, experiment_identity


def interval(values):
    return np.quantile(values, [0.025, 0.975])


def summarize(frame):
    results = []
    base = frame.loc[frame.method == 'baseline'].set_index('b').sort_index()
    for (method, percent), rows in frame.groupby(['method', 'percent'], sort=False):
        rows = rows.set_index('b').sort_index()
        if not rows.index.equals(base.index):
            raise ValueError('The paired bootstrap series have different iterations')
        delta = rows.theta - base.theta
        lo, hi = interval(rows.theta)
        delta_lo, delta_hi = interval(delta)
        results.append(dict(method=method, percent=percent, n=len(rows),
                            theta_mean=rows.theta.mean(), theta_boot_sd=rows.theta.std(),
                            ci_low=lo, ci_high=hi, g_logloss=rows.g_logloss.mean(),
                            m_logloss=rows.m_logloss.mean(), elapsed_s=rows.elapsed_s.mean(),
                            training_s=rows.training_s.mean(), total_elapsed_s=rows.elapsed_s.sum(),
                            paired_delta=delta.mean(), paired_delta_sd=delta.std(),
                            paired_ci_low=delta_lo, paired_ci_high=delta_hi))
    return pd.DataFrame(results)


def figures(out, selection, boot, summary):
    baseline = selection['rows'][0]
    for targets, name in [(('g_logloss',), 'validation_outcome_logloss'),
                          (('g_logloss', 'm_logloss'), 'validation_method_comparison')]:
        fig, axes = plt.subplots(1, len(targets), squeeze=False, figsize=(5 * len(targets), 4))
        for metric, ax in zip(targets, axes[0]):
            for method in METHODS:
                rows = sorted((r for r in selection['rows'] if r['method'] == method),
                              key=lambda r: r['percent'])
                if rows:
                    points = [baseline, *rows]
                    ax.errorbar([r['percent'] for r in points], [r[metric] for r in points],
                                yerr=[r[metric + '_sd'] for r in points], marker='o', label=method)
            ax.axhline(baseline[metric], color='black', linestyle=':', label='Baseline')
            ax.set(xlabel='Synthetic augmentation (%)',
                   ylabel='Outcome log loss' if metric[0] == 'g' else 'Treatment log loss')
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots()
    bins = np.histogram_bin_edges(boot.theta, bins='auto')
    for (method, percent), rows in boot.groupby(['method', 'percent'], sort=False):
        ax.hist(rows.theta, bins=bins, histtype='step', label=f'{method} {percent}%')
    ax.set(xlabel='Edema effect estimate', ylabel='Bootstrap count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'bootstrap_distributions.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    labels = [f'{r.method} {r.percent}%' for r in summary.itertuples()]
    positions = np.arange(len(labels))
    ax.hlines(positions, summary.ci_low, summary.ci_high)
    ax.scatter(summary.theta_mean, positions)
    ax.set(yticks=positions, yticklabels=labels, xlabel='Bootstrap mean and 95% percentile interval')
    fig.tight_layout()
    fig.savefig(out / 'bootstrap_intervals.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='main')
    parser.add_argument('--iterations', type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    if not 2 <= args.iterations <= BOOTSTRAPS:
        parser.error('At least two bootstrap iterations are required')
    out = run_directory(args.run)
    X, Y, D, patient_hash = load_data()
    check_reference_folds(X, D)
    identity = experiment_identity(X, Y, D, patient_hash)
    _, selection, cases = load_selection(out, identity)
    rows = []
    for b in range(args.iterations):
        _, _, keys = iteration_keys(b, X, D, identity, cases)
        for key in keys:
            record = checked(record_path(out, key), key,
                             lambda result: validate_result(result, key['valid']))
            if key['valid']:
                rows.append({k: v for k, v in (key | record).items() if k not in
                             ('identity', 'folds', 'ci')})
    frame = pd.DataFrame(rows)
    if len(frame) < 2 * len(cases):
        raise ValueError('At least two valid paired iterations are required')
    summary = summarize(frame)
    comparison = out / f'comparison_{args.iterations}'
    comparison.mkdir(exist_ok=True)
    with exclusive(comparison / 'comparison.lock'):
        with TemporaryDirectory(dir=comparison) as temp:
            staging = Path(temp)
            frame.to_csv(staging / 'paired_bootstrap.csv', index=False)
            summary.to_csv(staging / 'summary.csv', index=False)
            pd.DataFrame([{k: v for k, v in r.items() if k != 'folds'}
                          for r in selection['rows']]).to_csv(staging / 'validation.csv', index=False)
            figures(staging, selection, frame, summary)
            for path in staging.iterdir():
                path.replace(comparison / path.name)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
