"""Training and PLR score from the experiment 05 optimized QNN."""
import time

import numpy as np
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler

from augmentation_bn import diagnostics, generate
from config import synthetic_count


def quantum_nn_fit_predict(Xtr, ytr, Xte, seed):
    # Inputs have already been transformed by the original fold-local pipeline.
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device('lightning.qubit', wires=12)

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(12), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(12))
        return qml.expval(qml.PauliZ(0))

    def predict(weights, X):
        return (1 - pnp.array([circuit(weights, x) for x in X])) / 2

    rng = np.random.RandomState(seed)
    weights = pnp.array(0.1 * rng.randn(3, 12, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.05)
    target = pnp.array(ytr, dtype=float)

    def cost(w):
        p = pnp.clip(predict(w, Xtr), 1e-6, 1 - 1e-6)
        return -pnp.mean(target * pnp.log(p) + (1 - target) * pnp.log(1 - p))

    for _ in range(120):
        weights, _ = opt.step_and_cost(cost, weights)
    predictions = np.array(predict(weights, Xte), dtype=float)
    if not np.isfinite(predictions).all():
        raise FloatingPointError('Nonfinite QNN predictions')
    return predictions


def losses(Y, D, g, m):
    return dict(g_logloss=float(log_loss(Y, g, labels=[0, 1])),
                m_logloss=float(log_loss(D, m, labels=[0, 1])))


def fit_fold(Xtr, Ytr, Dtr, Xte, method, percent, seed, fold):
    imputer = SimpleImputer(strategy='median').fit(Xtr)
    real = imputer.transform(Xtr)
    if real.shape[1] != 12:
        raise ValueError('The optimized QNN requires 12 nonempty training columns')
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(real)
    count = synthetic_count(len(real), percent)
    audit = None
    start = time.perf_counter()
    if method == 'BN' and count:
        # A separate random stream keeps circuit initialization identical.
        generator_seed = [seed, fold, 1]
        synthetic, sy, sd, clipped = generate(real, Ytr, Dtr, count, generator_seed)
        audit = diagnostics(real, Ytr, Dtr, synthetic, sy, sd, clipped)
        Xtr = np.concatenate((real, synthetic))
        Ytr = np.concatenate((Ytr, sy))
        Dtr = np.concatenate((Dtr, sd))
    elif method == 'baseline' and percent == 0:
        Xtr = real
    else:
        raise ValueError(f'Unavailable or invalid augmentation: {method}-{percent}')

    train = scaler.transform(Xtr)
    test = scaler.transform(imputer.transform(Xte))
    fit_start = time.perf_counter()
    g = quantum_nn_fit_predict(train, Ytr, test, seed)
    m = quantum_nn_fit_predict(train, Dtr, test, seed)
    return g, m, dict(n_synthetic=count, synthetic_diagnostics=audit,
                      training_s=time.perf_counter() - fit_start,
                      elapsed_s=time.perf_counter() - start)


def dml(Y, D, g, m):
    """The original score, applied to real observations and real OOF predictions."""
    if not np.isfinite([g, m]).all():
        raise FloatingPointError('Nonfinite nuisance predictions')
    Y_res = Y - g
    D_res = D - np.clip(m, 0.01, 0.99)
    denom = np.sum(D_res ** 2)
    if not np.isfinite(denom) or denom <= 1e-12:
        raise FloatingPointError('Invalid DML denominator')
    theta = np.sum(D_res * Y_res) / denom
    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / J ** 2 / len(Y))
    z = float(theta / se) if se else None
    return dict(theta=float(theta), se=float(se), z=z,
                pval=float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None,
                ci=[float(theta - 1.96 * se), float(theta + 1.96 * se)])
