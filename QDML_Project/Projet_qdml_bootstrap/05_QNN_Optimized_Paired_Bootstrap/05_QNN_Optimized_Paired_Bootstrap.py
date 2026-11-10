"""
DML QUANTIQUE OPTIMISÉ (QNN_OPT) - 12 variables radiomiques
BOOTSTRAP COMPLET, APPARIÉ PAR ITÉRATION b AVEC LES RÉSULTATS MLP EXISTANTS

Dérivé de QDML_Bootstrap_paired.py. Seul le QNN optimisé est entraîné :
  12 qubits, 3 couches, 120 époques, lr = 0.05 (meilleure config de Test_Choix_Hyperparametres_QNN.py)
Le MLP n'est JAMAIS réentraîné : ses theta* sont relus dans
resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized/ par compare_qdml_results.py.

L'appariement est valide car, pour une itération b, l'échantillon (graine BOOT_SEED + b), les plis
(StratifiedGroupKFold, graine BOOT_SEED + b, groupes = patients) et la règle de validité
ne dépendent que des données, pas du modèle.

Utilisation :
  python QDML_Bootstrap_With_Optimized_QNN_and_MLP.py --check                      # vérifications, aucun entraînement
  python QDML_Bootstrap_With_Optimized_QNN_and_MLP.py --start 0 --end 50           # estimation originale + itérations 0..49
  python QDML_Bootstrap_With_Optimized_QNN_and_MLP.py --start 50 --end 100 --skip-original

"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import pennylane as qml
from pennylane import numpy as pnp

# ===========================
# Paramètres (identiques à QDML_Bootstrap_paired.py)
# ===========================
RANDOM_STATE = 42
N_FOLDS = 5
PROPENSITY_CLIP = 0.01
BOOT_SEED = 0              # itération b -> graine BOOT_SEED + b (identique à l'ancien bootstrap)
DATA_PATH = "data/GBM_GBS.xlsx"
PREV_DIR = "resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized"   # résultats MLP/QNN existants (lecture seule)
MODEL = "QNN_OPT"                                                  # nom des colonnes de sortie

# QNN optimisé
N_LAYERS = 3
QNN_EPOCHS = 120
QNN_LR = 0.05
TRAIN_CLIP = 1e-6          # clip des probabilités dans la perte (inchangé)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="première itération (incluse)")
parser.add_argument("--end", type=int, default=1000, help="dernière itération (exclue)")
parser.add_argument("--skip-original", action="store_true",
                    help="ne pas calculer l'estimation sur les données originales")
parser.add_argument("--check", action="store_true",
                    help="vérifie données, appariement et sorties, sans entraînement")
parser.add_argument("--out-dir", default="resultats_bootstrap_QNN_OPT",
                    help="dossier de sortie (jamais un dossier de résultats existant d'un autre modèle)")
args = parser.parse_args()

OUT_DIR = args.out_dir
# ne jamais écrire dans les anciens dossiers de résultats
for protected in (PREV_DIR, "resultats_bootstrap"):
    if os.path.abspath(OUT_DIR) == os.path.abspath(protected):
        raise SystemExit(f"--out-dir ne peut pas être {protected} (résultats existants protégés)")
if not 0 <= args.start < args.end:
    raise SystemExit("Il faut 0 <= --start < --end")

# ===========================
# Les 12 variables radiomiques
# ===========================
SELECTED_RADIO = [
    "original_shape_MajorAxis", "original_shape_Sphericity",
    "original_glcm_JointEntropy", "original_glcm_MaximumProbability",
    "log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformity",
    "log-sigma-5-0-mm-3D_firstorder_RootMeanSquared",
    "wavelet-LLL_firstorder_Kurtosis", "wavelet-LHH_firstorder_Variance",
    "wavelet-HHL_firstorder_Skewness", "wavelet-HHL_gldm_LowGrayLevelEmphasis",
    "squareroot_glrlm_HighGrayLevelRunEmphasis", "squareroot_glrlm_RunVariance",
]
assert len(SELECTED_RADIO) == 12, "SELECTED_RADIO doit contenir exactement 12 variables"

# ===========================
# 1. Données (inchangé)
# ===========================
df = pd.read_excel(DATA_PATH, header=3)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)

df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

missing = [c for c in SELECTED_RADIO if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du fichier : {missing}")

X = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce")   # NaN imputés dans chaque pli
Y = df["Y"].values
D = df["D"].values
Xs = X.values
n = len(Y)

print(f"n = {n} | p = {X.shape[1]}")
print(f"traitement : {D.sum()} avec oedème / {n - D.sum()} sans oedème\n")

# ===========================
# 2. Modèle de nuisance quantique optimisé
# ===========================
N_QUBITS = X.shape[1]      # 12 qubits (un par variable)

_dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(_dev, diff_method="adjoint")
def _quantum_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def _quantum_predict_proba(weights, X_):
    raw = pnp.array([_quantum_circuit(weights, x) for x in X_])
    return (1 - raw) / 2

def quantum_nn_fit_predict(Xtr, ytr, Xte, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(Xtr)
    Xtr_q = scaler.transform(Xtr)
    Xte_q = scaler.transform(Xte)

    weights = pnp.array(0.1 * rng.randn(N_LAYERS, N_QUBITS, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=QNN_LR)
    ytr_arr = pnp.array(ytr, dtype=float)

    def cost(w):
        p_hat = _quantum_predict_proba(w, Xtr_q)
        p_hat = pnp.clip(p_hat, TRAIN_CLIP, 1 - TRAIN_CLIP)
        return -pnp.mean(ytr_arr * pnp.log(p_hat) + (1 - ytr_arr) * pnp.log(1 - p_hat))

    for _ in range(QNN_EPOCHS):
        weights, _ = opt.step_and_cost(cost, weights)

    preds = np.array(_quantum_predict_proba(weights, Xte_q), dtype=float)
    if not np.all(np.isfinite(preds)):
        raise FloatingPointError("prédictions QNN non finies")
    return preds

# ===========================
# 3. DML générique
# ===========================
def run_dml(nuisance_fit_predict, Xs, Y, D, n_folds=N_FOLDS, SEED=RANDOM_STATE,
            clip=PROPENSITY_CLIP, label="", groups=None, verbose=True):
    n = len(Y)

    if groups is None:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = skf.split(Xs, D)
    else:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = skf.split(Xs, D, groups=groups)

    g_hat = np.zeros(n)   # P(Y=1 | X)
    m_hat = np.zeros(n)   # score de propension P(D=1 | X)
    covered = np.zeros(n, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if verbose:
            print(f"[{label}] pli {fold}/{n_folds}...", flush=True)
        if groups is not None and np.intersect1d(groups[train_idx], groups[test_idx]).size:
            raise RuntimeError("fuite : un patient est à la fois en apprentissage et en test")
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        g_hat[test_idx] = nuisance_fit_predict(Xtr, Y[train_idx], Xte, seed=SEED)
        m_hat[test_idx] = nuisance_fit_predict(Xtr, D[train_idx], Xte, seed=SEED)
        covered[test_idx] += 1

    if not np.all(covered == 1):
        raise RuntimeError("chaque observation doit être prédite exactement une fois")
    if not (np.all(np.isfinite(g_hat)) and np.all(np.isfinite(m_hat))):
        raise FloatingPointError("prédictions de nuisance non finies")

    m_hat = np.clip(m_hat, clip, 1 - clip)

    Y_res = Y - g_hat
    D_res = D - m_hat
    denom = np.sum(D_res ** 2)
    if not np.isfinite(denom) or denom <= 1e-12:
        raise FloatingPointError(f"dénominateur DML invalide : {denom}")
    theta = np.sum(D_res * Y_res) / denom

    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / (J ** 2) / n)
    z = theta / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    ci = (theta - 1.96 * se, theta + 1.96 * se)

    return {"theta": theta, "se": se, "z": z, "pval": pval, "ci": ci}

# ===========================
# Échantillon bootstrap b (identique à l'ancien script) et règle de validité
# ===========================
def boot_indices(b, n=n, seed=BOOT_SEED):
    return np.random.RandomState(seed + b).choice(n, size=n, replace=True)

def boot_valid(idx, D=D, n_folds=N_FOLDS):
    Db = D[idx]
    return (len(np.unique(idx[Db == 0])) >= n_folds and
            len(np.unique(idx[Db == 1])) >= n_folds)

def idx_hash(idx):
    return hashlib.md5(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]

# Empreinte des plis StratifiedGroupKFold des itérations b = 0, 1, 2.
# REF_FOLD_HASH a été obtenue avec scikit-learn >= 1.8 et reproduit exactement (12 décimales) les theta*
# du MLP existant. scikit-learn <= 1.7 mélange les groupes autrement (bug corrigé en 1.8) :
# les plis, donc l'appariement avec le MLP, seraient différents.
REF_FOLD_HASH = "a56f4caa2b1e"

def fold_fingerprint(n_iter=3):
    h = hashlib.md5()
    for b in range(n_iter):
        idx = boot_indices(b)
        lab = np.empty(n, dtype=np.int64)
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=BOOT_SEED + b)
        for k, (_, te) in enumerate(skf.split(np.zeros(n), D[idx], groups=idx)):
            lab[te] = k
        h.update(lab.tobytes())
    return h.hexdigest()[:12]

FOLD_HASH = fold_fingerprint()
PAIRED = FOLD_HASH == REF_FOLD_HASH

def save_csv(rows, path):
    """Écriture atomique : un job tué ne laisse jamais un fichier tronqué."""
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)

CONFIG = {"model": MODEL, "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "epochs": QNN_EPOCHS,
          "lr": QNN_LR, "train_clip": TRAIN_CLIP, "n_folds": N_FOLDS, "propensity_clip": PROPENSITY_CLIP,
          "boot_seed": BOOT_SEED, "random_state": RANDOM_STATE, "n": int(n),
          "sklearn": sklearn.__version__, "pennylane": qml.__version__, "fold_hash": FOLD_HASH}
UNPAIRED_MSG = (f"plis bootstrap différents de ceux du MLP existant (empreinte {FOLD_HASH} ≠ {REF_FOLD_HASH}) : "
                f"scikit-learn {sklearn.__version__}, il faut scikit-learn >= 1.8 "
                "(pip install 'scikit-learn>=1.8')")

# ===========================
# --check : vérifications sans entraînement
# ===========================
if args.check:
    problems = []
    print("=== Vérifications ===")
    print("config :", json.dumps(CONFIG))
    w = pnp.array(0.1 * np.random.RandomState(0).randn(N_LAYERS, N_QUBITS, 3))
    p = float(_quantum_predict_proba(w, np.zeros((1, N_QUBITS)))[0])
    print(f"circuit : 1 évaluation OK (p = {p:.4f})")
    if not 0 <= p <= 1:
        problems.append("probabilité du circuit hors [0, 1]")
    if PAIRED:
        print(f"plis bootstrap identiques à ceux du MLP existant (empreinte {FOLD_HASH}) : OK")
    else:
        problems.append(UNPAIRED_MSG)

    prev = sorted(glob.glob(os.path.join(PREV_DIR, "boot_*_*.csv")))
    if not prev:
        problems.append(f"aucun fichier boot_*.csv dans {PREV_DIR}")
    else:
        old = pd.concat([pd.read_csv(f) for f in prev]).drop_duplicates("b", keep="last")
        valid_old = set(old.dropna(subset=["NN"])["b"].astype(int))
        print(f"MLP existant : {len(old)} itérations, {len(valid_old)} valides")
        # la règle de validité recalculée doit reproduire exactement le motif des NaN du MLP
        mism = [b for b in old["b"].astype(int) if boot_valid(boot_indices(b)) != (b in valid_old)]
        if mism:
            problems.append(f"validité recalculée ≠ MLP existant pour {len(mism)} itérations (ex. {mism[:5]})")
        else:
            print("validité des échantillons recalculée = motif du MLP existant : OK")

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_path = os.path.join(OUT_DIR, "config_QNN_OPT.json")
    if os.path.exists(cfg_path):
        old_cfg = json.load(open(cfg_path))
        diffs = {k: (old_cfg.get(k), v) for k, v in CONFIG.items() if old_cfg.get(k) != v}
        if diffs:
            problems.append(f"config différente de celle déjà enregistrée : {diffs}")
    done = [f for f in glob.glob(os.path.join(OUT_DIR, "boot_*_*.csv"))]
    if done:
        cur = pd.concat([pd.read_csv(f) for f in done]).drop_duplicates("b", keep="last")
        bad = [int(r.b) for r in cur.itertuples() if r.idx_hash != idx_hash(boot_indices(int(r.b)))]
        print(f"{MODEL} déjà fait : {len(cur)} itérations ({cur[MODEL].notna().sum()} valides)")
        if bad:
            problems.append(f"hash d'échantillon incohérent pour b = {bad[:5]}")
    print("\n" + ("PROBLÈMES :\n  - " + "\n  - ".join(problems) if problems else "Tout est OK."))
    raise SystemExit(1 if problems else 0)

if not PAIRED:   # refuse de produire des theta* non appariables avec le MLP
    raise SystemExit("ARRÊT : " + UNPAIRED_MSG)

os.makedirs(OUT_DIR, exist_ok=True)
cfg_path = os.path.join(OUT_DIR, "config_QNN_OPT.json")
if not os.path.exists(cfg_path):   # écriture atomique (plusieurs tâches SLURM en parallèle)
    tmp = f"{cfg_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(CONFIG, f, indent=2)
    os.replace(tmp, cfg_path)
elif json.load(open(cfg_path)).get("fold_hash") != FOLD_HASH:
    raise SystemExit(f"ARRÊT : {cfg_path} a été produit avec d'autres plis ; utiliser un autre --out-dir")

# ===========================
# 4. Estimation sur les données originales (QNN_OPT seulement, jamais écrasée)
# ===========================
ORIG_PATH = os.path.join(OUT_DIR, "resultats_originaux_QNN_OPT.csv")

if not args.skip_original:
    if os.path.exists(ORIG_PATH):
        print(f"{ORIG_PATH} existe déjà : estimation originale non recalculée")
    else:
        print(f"\n=== DML sur données originales : {MODEL} ===")
        t0 = time.time()
        r = run_dml(quantum_nn_fit_predict, Xs, Y, D, label=MODEL)
        dt = time.time() - t0
        print(f" => terminé en {dt:.1f} s | theta = {r['theta']:.4f} | SE = {r['se']:.4f} "
              f"| p = {r['pval']:.4f} | IC = [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        save_csv([{"Modèle": MODEL, "theta": r["theta"], "SE": r["se"], "p-value": r["pval"],
                   "IC_low": r["ci"][0], "IC_high": r["ci"][1], "temps_s": dt}], ORIG_PATH)

# ===========================
# 5. Bootstrap (mêmes échantillons et plis que l'ancien bootstrap apparié)
# ===========================
def qnn_bootstrap(Xs, Y, D, start, end, seed=BOOT_SEED, n_folds=N_FOLDS):
    """Itérations start..end-1, sauvegarde à chaque itération, reprise automatique."""
    path = os.path.join(OUT_DIR, f"boot_{start}_{end}.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {int(r["b"]) for r in rows}
    todo = [b for b in range(start, end) if b not in done]
    if done:
        print(f"Reprise : {len(done)} itérations déjà faites, {len(todo)} restantes")
    t0 = time.time()

    for k, b in enumerate(todo, start=1):
        idx = boot_indices(b, n=len(Y), seed=seed)
        row = {"b": b, MODEL: np.nan, MODEL + "_se": np.nan, "idx_hash": idx_hash(idx),
               "temps_s": np.nan, "erreur": ""}
        tb = time.time()
        if not boot_valid(idx, D=D, n_folds=n_folds):
            row["erreur"] = "échantillon invalide (trop peu de patients distincts par groupe)"
        else:
            try:
                res = run_dml(quantum_nn_fit_predict, Xs[idx], Y[idx], D[idx], n_folds=n_folds,
                              SEED=seed + b, groups=idx, verbose=False)
                row[MODEL] = res["theta"]
                row[MODEL + "_se"] = res["se"]
            except Exception as e:
                row["erreur"] = f"{type(e).__name__}: {e}"[:200]
                print(f"  itération {b} écartée : {row['erreur']}", flush=True)
        row["temps_s"] = time.time() - tb

        rows.append(row)
        save_csv(rows, path)   # sauvegarde à chaque itération
        print(f"[bootstrap] b={b} | {k}/{len(todo)} | theta={row[MODEL]:.4f} | "
              f"{row['temps_s']:.0f}s (total {time.time() - t0:.0f}s)", flush=True)

print(f"\n=== Bootstrap {MODEL} : itérations {args.start} à {args.end - 1} ===")
qnn_bootstrap(Xs, Y, D, args.start, args.end)
print("Terminé. Analyse : python compare_qdml_results.py")
