"""
DML CLASSIQUE (MLP) VS DML QUANTIQUE (QNN) - 12 variables radiomiques
BOOTSTRAP COMPLET APPARIÉ (approche A)

À chaque itération b du bootstrap :
  1. on rééchantillonne les patients (avec remise) -> MÊME échantillon pour NN et QNN
  2. cross-fitting : on réentraîne le MLP, puis le QNN, sur cet échantillon
  3. on calcule les résidus et theta (formule DML identique pour les deux)
On répète B fois (B = 100 pour commencer, 1000 ensuite) -> distribution des theta*.

Utilisation :
  python QDML_bootstrap_paired.py                       # estimation initiale + itérations 0..99 + analyse
  python QDML_bootstrap_paired.py --start 0 --end 50 --skip-original      # un "worker"
  python QDML_bootstrap_paired.py --start 50 --end 100 --skip-original    # un autre worker en parallèle
  python QDML_bootstrap_paired.py --analyse-only        # fusionne les fichiers et fait l'analyse

Légende : [AJOUTÉ] [MODIFIÉ] [SUPPRIMÉ] 
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline          # [AJOUTÉ] standardisation à l'intérieur des plis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib
matplotlib.use("Agg")                      # figures enregistrées, pas de fenêtre
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

# ===========================
# Paramètres
# ===========================
RANDOM_STATE = 42
N_FOLDS = 5
PROPENSITY_CLIP = 0.01
BOOT_SEED = 0              # [AJOUTÉ] graine des rééchantillonnages (itération b -> graine BOOT_SEED + b)
# [AJOUTÉ] configuration du MLP, choisie sur le log-loss hors-échantillon (30 graines, comparaison appariée)
MLP_HIDDEN = 32            # [MODIFIÉ] avant : 16
MLP_ALPHA = 10.0           # [MODIFIÉ] avant : 0.01
MLP_MAX_ITER = 500         # [MODIFIÉ] avant : 2000 (valeur utilisée dans le test de sélection)
OUT_DIR = "resultats_bootstrap"   # [AJOUTÉ] dossier de sortie
os.makedirs(OUT_DIR, exist_ok=True)

# [AJOUTÉ] arguments de ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="première itération (incluse)")
parser.add_argument("--end", type=int, default=100, help="dernière itération (exclue)")
parser.add_argument("--skip-original", action="store_true",
                    help="ne pas recalculer l'estimation sur les données originales")
parser.add_argument("--analyse-only", action="store_true",
                    help="ne lance aucun calcul, fusionne les fichiers et analyse")
args = parser.parse_args()

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
# 1. Données
# ===========================
df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)

df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

missing = [c for c in SELECTED_RADIO if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du fichier : {missing}")

# [SUPPRIMÉ] Age et variables cliniques (Sex, Localization, Necrosis) : seulement les 12 radiomiques
# [MODIFIÉ] NaN -> médiane (avant : 0)
X = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

Y = df["Y"].values
D = df["D"].values
# [MODIFIÉ] plus de standardisation globale : elle est faite dans chaque pli (Pipeline du MLP).
# Le QNN a son propre MinMaxScaler ajusté sur Xtr, insensible à une standardisation préalable.
Xs = X.values
n = len(Y)

print(f"n = {n} | p = {X.shape[1]}")
print(f"traitement : {D.sum()} avec oedème / {n - D.sum()} sans oedème\n")

# ===========================
# 2. Modèle de nuisance classique : MLP
# ===========================
def classical_nn_fit_predict(Xtr, ytr, Xte, seed=RANDOM_STATE):
    # [MODIFIÉ] Pipeline : le StandardScaler est ajusté sur Xtr seulement (pas de fuite),
    #           puis appliqué à Xte. early_stopping désactivé (instable avec 22 patients sans oedème).
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(MLP_HIDDEN,),
            activation="relu",
            alpha=MLP_ALPHA,           # régularisation L2 forte
            max_iter=MLP_MAX_ITER,
            early_stopping=False,      # [MODIFIÉ] avant : True
            random_state=seed,
        ),
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

# ===========================
# 3. Modèle de nuisance quantique : QNN
# ===========================
N_QUBITS = X.shape[1]      # 12 qubits (un par variable)
N_LAYERS = 1
QNN_EPOCHS = 40
QNN_LR = 0.2

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
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(Xtr)
    Xtr_q = scaler.transform(Xtr)
    Xte_q = scaler.transform(Xte)

    weights = pnp.array(0.1 * rng.randn(N_LAYERS, N_QUBITS, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=QNN_LR)
    ytr_arr = pnp.array(ytr, dtype=float)

    def cost(w):
        p_hat = _quantum_predict_proba(w, Xtr_q)
        p_hat = pnp.clip(p_hat, 1e-6, 1 - 1e-6)
        return -pnp.mean(ytr_arr * pnp.log(p_hat) + (1 - ytr_arr) * pnp.log(1 - p_hat))

    for _ in range(QNN_EPOCHS):
        weights, _ = opt.step_and_cost(cost, weights)

    return np.array(_quantum_predict_proba(weights, Xte_q))

# ===========================
# 4. DML générique
# ===========================
# [MODIFIÉ] paramètre `groups` : dans un échantillon bootstrap, les copies d'un même patient
#           restent dans le même pli (évite la fuite de données)
# [MODIFIÉ] n = len(Y) local ; print corrigé ; paramètre verbose
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

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if verbose:
            print(f"[{label}] pli {fold}/{n_folds}...", flush=True)
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        g_hat[test_idx] = nuisance_fit_predict(Xtr, Y[train_idx], Xte, seed=SEED)
        m_hat[test_idx] = nuisance_fit_predict(Xtr, D[train_idx], Xte, seed=SEED)

    m_hat = np.clip(m_hat, clip, 1 - clip)

    Y_res = Y - g_hat
    D_res = D - m_hat
    theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)

    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / (J ** 2) / n)
    z = theta / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    ci = (theta - 1.96 * se, theta + 1.96 * se)

    return {"theta": theta, "se": se, "z": z, "pval": pval, "ci": ci}

MODELS = {"NN": classical_nn_fit_predict, "QNN": quantum_nn_fit_predict}   # [AJOUTÉ]

# ===========================
# 5. [AJOUTÉ] Estimation sur les données originales (point de référence)
# ===========================
ORIG_PATH = os.path.join(OUT_DIR, "resultats_originaux.csv")

if not args.analyse_only and not args.skip_original:
    rows = []
    for name, fp in MODELS.items():
        print(f"\n=== DML sur données originales : {name} ===")
        t0 = time.time()
        r = run_dml(fp, Xs, Y, D, label=name)
        dt = time.time() - t0
        print(f" => terminé en {dt:.1f} s | theta = {r['theta']:.4f} | SE = {r['se']:.4f} "
              f"| p = {r['pval']:.4f} | IC = [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        rows.append({"Modèle": name, "theta": r["theta"], "SE": r["se"], "p-value": r["pval"],
                     "IC_low": r["ci"][0], "IC_high": r["ci"][1], "temps_s": dt})
    pd.DataFrame(rows).set_index("Modèle").round(4).to_csv(ORIG_PATH)

# ===========================
# 6. [AJOUTÉ] Bootstrap complet apparié (NN et QNN sur les mêmes échantillons)
# ===========================
def paired_bootstrap(models, Xs, Y, D, start, end, seed=BOOT_SEED, n_folds=N_FOLDS):
    """Itérations start..end-1. Reprend automatiquement après une interruption."""
    path = os.path.join(OUT_DIR, f"boot_{start}_{end}.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    first = (int(max(r["b"] for r in rows)) + 1) if rows else start
    if first > start:
        print(f"Reprise à l'itération {first} (déjà faites : {first - start})")
    n = len(Y)
    t0 = time.time()

    for b in range(first, end):
        # même graine -> même échantillon pour NN et QNN, et reproductible à la reprise
        idx = np.random.RandomState(seed + b).choice(n, size=n, replace=True)
        Db = D[idx]
        row = {"b": b}

        # assez de patients DISTINCTS sans/avec oedème pour former les plis
        ok = (len(np.unique(idx[Db == 0])) >= n_folds and
              len(np.unique(idx[Db == 1])) >= n_folds)
        if ok:
            try:
                for name, fp in models.items():
                    res = run_dml(fp, Xs[idx], Y[idx], Db, n_folds=n_folds,
                                  SEED=seed + b, groups=idx, verbose=False)
                    row[name] = res["theta"]
                    row[name + "_se"] = res["se"]
            except Exception as e:
                print(f"  itération {b} écartée : {e}", flush=True)
                row = {"b": b}                     # on écarte pour les DEUX modèles (garde l'appariement)

        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)   # sauvegarde à chaque itération
        print(f"[bootstrap] {b + 1 - start}/{end - start} | {time.time() - t0:.0f}s", flush=True)

if not args.analyse_only:
    print(f"\n=== Bootstrap complet apparié : itérations {args.start} à {args.end - 1} ===")
    paired_bootstrap(MODELS, Xs, Y, D, args.start, args.end)

# ===========================
# 7. [AJOUTÉ] Analyse : fusion des fichiers, SE, IC percentile, p-value, histogrammes
# ===========================
def summarize(thetas):
    thetas = np.asarray(thetas)
    p_low = (np.sum(thetas <= 0) + 1) / (len(thetas) + 1)
    p_high = (np.sum(thetas >= 0) + 1) / (len(thetas) + 1)
    return {"moyenne_boot": thetas.mean(),
            "mediane_boot": np.median(thetas),
            "se_boot": thetas.std(ddof=1),
            "ic_low": np.percentile(thetas, 2.5),
            "ic_high": np.percentile(thetas, 97.5),
            "p_boot": min(1.0, 2 * min(p_low, p_high)),
            "B_valides": len(thetas)}

files = sorted(glob.glob(os.path.join(OUT_DIR, "boot_*_*.csv")))
if not files:
    raise SystemExit("Aucun fichier boot_*.csv trouvé : rien à analyser.")

allb = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
allb = allb.drop_duplicates(subset="b", keep="last").sort_values("b")
n_total = len(allb)
valid = allb.dropna(subset=["NN", "QNN"])       # itérations valides pour les DEUX modèles
print(f"\nItérations : {n_total} | valides (appariées) : {len(valid)} | écartées : {n_total - len(valid)}")

if len(valid) < 2:
    raise SystemExit("Pas assez d'itérations valides pour l'analyse.")

diff = valid["QNN"] - valid["NN"]
summary = pd.DataFrame({
    "NN": summarize(valid["NN"]),
    "QNN": summarize(valid["QNN"]),
    "QNN - NN": summarize(diff),
}).T

print("\n=== Résumé du bootstrap ===")
print(summary.round(4).to_string())

if os.path.exists(ORIG_PATH):
    orig = pd.read_csv(ORIG_PATH, index_col=0)
    print("\n=== Estimation sur données originales (SE analytique) ===")
    print(orig.to_string())

summary.round(4).to_csv(os.path.join(OUT_DIR, "resume_bootstrap.csv"))
valid.to_csv(os.path.join(OUT_DIR, "thetas_bootstrap_fusionnes.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(valid["NN"], bins=20, alpha=0.6, label="MLP")
axes[0].hist(valid["QNN"], bins=20, alpha=0.6, label="QNN")
axes[0].axvline(0, color="k", ls="--")
axes[0].set_xlabel("θ*")
axes[0].set_ylabel("fréquence")
axes[0].set_title("Distribution bootstrap de θ*")
axes[0].legend()

axes[1].hist(diff, bins=20, color="gray")
axes[1].axvline(0, color="k", ls="--")
axes[1].set_xlabel("θ*_QNN − θ*_NN")
axes[1].set_title("Différence appariée")
plt.savefig(os.path.join(OUT_DIR, "bootstrap_histogrammes.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFichiers dans {OUT_DIR}/ : resume_bootstrap.csv, thetas_bootstrap_fusionnes.csv, "
      f"bootstrap_histogrammes.png, resultats_originaux.csv")

# [SUPPRIMÉ] section "Diagnostics des modèles de nuisance" (AUC, log-loss, J, référence prévalence)
# [SUPPRIMÉ] forest plot et ancien tableau de comparaison (remplacés par l'analyse bootstrap)