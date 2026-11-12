import argparse
import time

import numpy as np

from config import BASELINE, BOOTSTRAPS, run_directory
from data_utils import boot_indices, boot_valid, check_reference_folds, digest, fold_labels, load_data, splits
from qnn_dml import dml, fit_fold, losses
from select_augmentation import validate_selection
from storage import cached, experiment_identity, finite_fields, read_json


def load_selection(out, identity=None):
    record = read_json(out / 'selection.json')
    key, result = record['key'], record['result']
    if identity is not None and key['identity'] != identity:
        raise ValueError('Selection and bootstrap have different experiment identities')
    validate_selection(result, key['plan'])
    cases = [BASELINE, *((method, percent) for method, percent in result['selected'].items()
                         if percent is not None)]
    return key, result, cases


def iteration_keys(b, X, D, identity, cases):
    idx = boot_indices(b, len(D))
    valid = boot_valid(idx, D)
    folds = splits(X[idx], D[idx], b, idx) if valid else []
    fold_hash = digest(fold_labels(folds, len(idx))) if valid else None
    keys = [dict(identity=identity, b=b, idx_hash=digest(idx), fold_hash=fold_hash,
                 seed=b, method=method, percent=percent, valid=valid)
            for method, percent in cases]
    return idx, folds, keys


def validate_result(result, valid):
    finite_fields(result, ('elapsed_s', 'training_s'))
    if valid:
        finite_fields(result, ('g_logloss', 'm_logloss', 'theta', 'se'))
        if len(result['ci']) != 2 or not np.isfinite(result['ci']).all():
            raise ValueError('Missing confidence interval')
        if len(result['folds']) != 5:
            raise ValueError('Missing fold results')


def evaluate_bootstrap(X, Y, D, folds, method, percent, seed):
    start = time.perf_counter()
    if not folds:
        return dict(elapsed_s=time.perf_counter() - start, training_s=0)
    g, m = np.empty(len(Y)), np.empty(len(Y))
    rows = []
    for fold, (train, test) in enumerate(folds):
        g[test], m[test], metrics = fit_fold(X[train], Y[train], D[train], X[test],
                                             method, percent, seed, fold)
        rows.append(dict(fold=fold, n_real_train=len(train), n_real_test=len(test),
                         **losses(Y[test], D[test], g[test], m[test]), **metrics))
    return dict(**dml(Y, D, g, m), **losses(Y, D, g, m), folds=rows,
                training_s=sum(row['training_s'] for row in rows),
                elapsed_s=time.perf_counter() - start)


def record_path(out, key):
    return out / f"boot_{key['b']:04d}_{key['method']}_{key['percent']}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='main')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= BOOTSTRAPS:
        parser.error('Invalid bootstrap range')
    out = run_directory(args.run)
    X, Y, D, patient_hash = load_data()
    check_reference_folds(X, D)
    identity = experiment_identity(X, Y, D, patient_hash)
    _, _, cases = load_selection(out, identity)
    for b in range(args.start, args.end):
        idx, folds, keys = iteration_keys(b, X, D, identity, cases)
        for key in keys:
            cached(record_path(out, key), key,
                   lambda: evaluate_bootstrap(X[idx], Y[idx], D[idx], folds,
                                              key['method'], key['percent'], b),
                   lambda result: validate_result(result, key['valid']))
        print(f'Bootstrap {b}: {len(cases)} records validated')


if __name__ == '__main__':
    main()
