import ast
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from config import FOLDS, REFERENCE_FOLD_HASH, original_source, workbook_path


def load_data():
    # The historical executable starts training at import time. Read its feature
    # list without importing it.
    tree = ast.parse(original_source().read_text(encoding='utf-8-sig'))
    features = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SELECTED_RADIO'
                            for t in node.targets))
    frame = pd.read_excel(workbook_path(), header=3)
    frame.columns = frame.columns.str.strip()
    frame = frame.dropna(subset=['name']).dropna(axis=1, how='all').reset_index(drop=True)
    X = frame[features].apply(pd.to_numeric, errors='coerce').to_numpy()
    Y = frame['name'].astype(str).str.endswith('GSM').astype(int).to_numpy()
    D = (frame['Edema'] == 'Y').astype(int).to_numpy()
    if X.shape != (183, 12) or frame['name'].nunique() != len(Y):
        raise ValueError('Expected the original 183 distinct patients and 12 features')
    patient_hash = hashlib.sha256(frame['name'].to_json().encode()).hexdigest()
    return X, Y, D, patient_hash


def boot_indices(b, n):
    return np.random.RandomState(b).choice(n, size=n, replace=True)


def boot_valid(idx, D):
    return all(len(np.unique(idx[D[idx] == c])) >= FOLDS for c in (0, 1))


def digest(indices):
    return hashlib.md5(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:12]


def splits(X, D, seed, groups=None):
    cls = StratifiedKFold if groups is None else StratifiedGroupKFold
    folds = list(cls(n_splits=FOLDS, shuffle=True, random_state=seed).split(X, D, groups))
    if groups is not None:
        for train, test in folds:
            if np.intersect1d(groups[train], groups[test]).size:
                raise ValueError('A patient appears on both sides of a fold')
    if not np.array_equal(np.sort(np.concatenate([test for _, test in folds])), np.arange(len(D))):
        raise ValueError('Every real observation must be held out exactly once')
    return folds


def fold_labels(folds, n):
    labels = np.empty(n, dtype=np.int64)
    for k, (_, test) in enumerate(folds):
        labels[test] = k
    return labels


def check_reference_folds(X, D):
    reference = hashlib.md5()
    for b in range(3):
        idx = boot_indices(b, len(D))
        folds = splits(X[idx], D[idx], b, idx)
        reference.update(fold_labels(folds, len(D)).tobytes())
    if reference.hexdigest()[:12] != REFERENCE_FOLD_HASH:
        raise ValueError('Bootstrap folds do not match the historical QNN reference')
