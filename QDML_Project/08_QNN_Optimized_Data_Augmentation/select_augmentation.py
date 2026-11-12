import argparse
from statistics import mean, stdev

from augmentation_seq import MISSING_SPECIFICATION
from config import BASELINE, METHODS, RATIOS, SELECTION_SEED, planned_cases, run_directory, synthetic_count
from data_utils import check_reference_folds, digest, fold_labels, load_data, splits
from qnn_dml import fit_fold, losses
from storage import cached, experiment_identity, finite_fields


def validate_fold(result, count):
    finite_fields(result, ('g_logloss', 'm_logloss', 'training_s', 'elapsed_s'))
    if result['n_synthetic'] != count:
        raise ValueError('Synthetic count mismatch')
    if count and not isinstance(result['synthetic_diagnostics'], dict):
        raise ValueError('Missing synthetic diagnostics')


def summarize(method, percent, rows):
    return dict(method=method, percent=percent,
                g_logloss=mean(row['g_logloss'] for row in rows),
                g_logloss_sd=stdev(row['g_logloss'] for row in rows),
                m_logloss=mean(row['m_logloss'] for row in rows),
                m_logloss_sd=stdev(row['m_logloss'] for row in rows),
                training_s=sum(row['training_s'] for row in rows),
                elapsed_s=sum(row['elapsed_s'] for row in rows),
                folds=rows)


def choose(rows, method):
    baseline = next(row for row in rows if row['method'] == BASELINE[0])
    eligible = [row for row in rows if row['method'] == method
                and row['g_logloss'] < baseline['g_logloss']
                and row['m_logloss'] <= baseline['m_logloss']]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row['g_logloss'], row['m_logloss'], row['percent']))['percent']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='main')
    parser.add_argument('--ratios', type=int, nargs='+', default=RATIOS)
    args = parser.parse_args()
    if len(set(args.ratios)) != len(args.ratios) or min(args.ratios) <= 0:
        parser.error('Ratios must be distinct positive integers')
    out = run_directory(args.run)
    X, Y, D, patient_hash = load_data()
    check_reference_folds(X, D)
    identity = experiment_identity(X, Y, D, patient_hash)
    folds = splits(X, D, SELECTION_SEED)
    fold_hash = digest(fold_labels(folds, len(Y)))
    plan = [[method, percent] for method, percent in planned_cases(args.ratios)]
    key = dict(identity=identity, plan=plan, seed=SELECTION_SEED, fold_hash=fold_hash)
    cached(out / 'plan.json', key, lambda: {'seq_blocked': MISSING_SPECIFICATION},
           lambda result: result['seq_blocked'])

    def evaluate_plan():
        rows = []
        for method, percent in plan:
            if method == 'SEQ':
                continue
            fold_rows = []
            for fold, (train, test) in enumerate(folds):
                def compute():
                    g, m, timing = fit_fold(X[train], Y[train], D[train], X[test],
                                            method, percent, SELECTION_SEED, fold)
                    return dict(fold=fold, n_real_train=len(train), n_real_test=len(test),
                                **losses(Y[test], D[test], g, m), **timing)

                fold_key = dict(identity=identity, seed=SELECTION_SEED, fold_hash=fold_hash,
                                fold=fold, method=method, percent=percent)
                count = synthetic_count(len(train), percent)
                row = cached(out / f'cv_{method}_{percent}_{fold}.json', fold_key,
                             compute, lambda result: validate_fold(result, count))
                fold_rows.append(row)
            rows.append(summarize(method, percent, fold_rows))
        return dict(rows=rows, selected={method: choose(rows, method) for method in METHODS},
                    seq_blocked=MISSING_SPECIFICATION)

    result = cached(out / 'selection.json', key, evaluate_plan,
                    lambda result: validate_selection(result, plan))
    print(f"Selected augmentation percentages: {result['selected']}; SEQ unavailable")


def validate_selection(result, plan):
    expected = [(method, percent) for method, percent in plan if method != 'SEQ']
    rows = result['rows']
    if [(row['method'], row['percent']) for row in rows] != expected:
        raise ValueError('Selection is missing configurations')
    if result['selected'] != {method: choose(rows, method) for method in METHODS}:
        raise ValueError('Selection does not match the validation rule')
    for row in rows:
        finite_fields(row, ('g_logloss', 'g_logloss_sd', 'm_logloss', 'm_logloss_sd',
                            'training_s', 'elapsed_s'))
        if len(row['folds']) != 5:
            raise ValueError('Selection is missing folds')


if __name__ == '__main__':
    main()
