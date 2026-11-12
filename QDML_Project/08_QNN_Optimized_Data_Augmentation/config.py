import os
from pathlib import Path

HOME = Path(__file__).resolve().parent
PROJECT = HOME.parent
ORIGINAL_DIR = '05_QNN_Optimized_Paired_Bootstrap'
ORIGINAL_NAME = f'{ORIGINAL_DIR}.py'
RATIOS = (25, 50, 75, 100, 150, 200)
SELECTION_SEED = 42
BOOTSTRAPS = 1000
FOLDS = 5
REFERENCE_FOLD_HASH = 'a56f4caa2b1e'
BASELINE = ('baseline', 0)
METHODS = ('SEQ', 'BN')
BN_COVARIANCE_SHRINKAGE = 0.2


def resolve_input(variable, candidates):
    override = os.environ.get(variable)
    paths = (Path(override).expanduser(),) if override else candidates
    for path in paths:
        if path.is_file():
            return path.resolve()
    attempted = ', '.join(map(str, paths))
    raise FileNotFoundError(f'Input not found. Tried {attempted}. Set {variable} to its file path.')


def original_source(root=PROJECT):
    nested = root / 'Projet_qdml_bootstrap'
    return resolve_input('QNN_AUG_ORIGINAL', (
        nested / ORIGINAL_DIR / ORIGINAL_NAME,
        root / ORIGINAL_DIR / ORIGINAL_NAME,
        root / ORIGINAL_NAME,
        nested / ORIGINAL_NAME,
    ))


def workbook_path(root=PROJECT):
    relative = Path('data') / 'GBM_GBS.xlsx'
    return resolve_input('QNN_AUG_WORKBOOK', (root / relative,
                                            root / 'Projet_qdml_bootstrap' / relative))


def run_directory(name):
    if not name or not name.isascii() or not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Run name must use ASCII letters, digits, underscores or hyphens')
    path = HOME / 'results' / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def synthetic_count(real_count, percent):
    if type(percent) is not int or percent < 0:
        raise ValueError('Augmentation percentage must be a nonnegative integer')
    return (real_count * percent + 50) // 100


def planned_cases(ratios=RATIOS):
    return [BASELINE, *((method, ratio) for method in METHODS for ratio in ratios)]
