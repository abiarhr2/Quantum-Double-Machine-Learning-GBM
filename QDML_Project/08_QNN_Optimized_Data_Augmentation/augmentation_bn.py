"""Fixed conditional-Gaussian BN: D -> Y; (D,Y) -> X, with dense X ordering."""
import numpy as np

from config import BN_COVARIANCE_SHRINKAGE


def generate(X, Y, D, count, seed):
    """Sample P(D) P(Y|D) P(X|D,Y), using empirical discrete probabilities."""
    if not np.isfinite(X).all() or not np.isin(Y, [0, 1]).all() or not np.isin(D, [0, 1]).all():
        raise ValueError('BN needs finite, imputed predictors and binary labels')
    rng = np.random.default_rng(seed)
    groups = 2 * D + Y
    frequencies = np.bincount(groups, minlength=4)
    draws = rng.choice(4, size=count, p=frequencies / len(groups))

    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1
    Z = (X - center) / scale
    means = np.zeros((4, X.shape[1]))
    for group in np.flatnonzero(frequencies):
        means[group] = Z[groups == group].mean(axis=0)
    residual = Z - means[groups]
    covariance = residual.T @ residual / max(1, len(X) - np.count_nonzero(frequencies))
    shrink = BN_COVARIANCE_SHRINKAGE
    covariance = (1 - shrink) * covariance + shrink * np.diag(np.diag(covariance))
    raw = (rng.multivariate_normal(np.zeros(X.shape[1]), covariance, size=count,
                                   check_valid='raise') + means[draws]) * scale + center
    lo, hi = X.min(axis=0), X.max(axis=0)
    synthetic = np.clip(raw, lo, hi)
    return synthetic, draws % 2, draws // 2, float(np.mean(raw != synthetic))


def diagnostics(X, Y, D, synthetic, sy, sd, clipped):
    if (not np.isfinite(synthetic).all() or not np.isin(sy, [0, 1]).all()
            or not np.isin(sd, [0, 1]).all()
            or np.any(synthetic < X.min(axis=0)) or np.any(synthetic > X.max(axis=0))):
        raise ValueError('Invalid synthetic data')
    rows = np.column_stack((synthetic, sy, sd))
    real = set(map(tuple, np.column_stack((X, Y, D))))
    scale = X.std(axis=0)
    scale[scale == 0] = 1

    def correlation(a):
        z = a - a.mean(axis=0)
        lengths = np.linalg.norm(z, axis=0)
        lengths[lengths == 0] = 1
        z = z / lengths
        return z.T @ z

    return {
        'real_joint_counts': np.bincount(2 * D + Y, minlength=4).tolist(),
        'synthetic_joint_counts': np.bincount(2 * sd + sy, minlength=4).tolist(),
        'synthetic_duplicates': len(rows) - len(np.unique(rows, axis=0)),
        'exact_real_matches': sum(tuple(row) in real for row in rows),
        'clipped_cell_fraction': clipped,
        'max_standardized_mean_shift': float(np.max(np.abs(synthetic.mean(axis=0) - X.mean(axis=0)) / scale)),
        'mean_absolute_joint_correlation_shift': float(np.mean(np.abs(
            correlation(rows) - correlation(np.column_stack((X, Y, D)))))),
    }
