"""
01_Selection_Initiale_MLP


CHOIX DU NOMBRE DE NEURONES CACHÉS DU MLP (12 variables radiomiques)

Pour chaque taille de couche cachée, on lance le DML sur les données ORIGINALES
avec plusieurs graines de cross-fitting, puis on regarde :

  1. CRITÈRE DE CHOIX (neutre) : la qualité de prédiction des nuisances g(X) et m(X)
     = log-loss hors-échantillon (obtenue par cross-fitting). Plus petit = meilleur.
  2. ANALYSE DE SENSIBILITÉ : theta, SE, p-value. Ils servent à vérifier que le
     résultat ne dépend pas trop de la taille, PAS à choisir la taille.

Lancement :  python choix_neurones_mlp.py
Durée : quelques secondes à quelques minutes (MLP seulement, pas de QNN).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

# ===========================
# Paramètres
# ===========================
N_FOLDS = 5
PROPENSITY_CLIP = 0.01
HIDDEN_SIZES = [4, 8, 12, 16, 32]      # tailles de la couche cachée à tester
SEEDS = list(range(10))                # 10 découpages différents des plis (stabilité)
ALPHA = 1e-2

SELECTED_RADIO = [
    "original_shape_MajorAxis", "original_shape_Sphericity",
    "original_glcm_JointEntropy", "original_glcm_MaximumProbability",
    "log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformity",
    "log-sigma-5-0-mm-3D_firstorder_RootMeanSquared",
    "wavelet-LLL_firstorder_Kurtosis", "wavelet-LHH_firstorder_Variance",
    "wavelet-HHL_firstorder_Skewness", "wavelet-HHL_gldm_LowGrayLevelEmphasis",
    "squareroot_glrlm_HighGrayLevelRunEmphasis", "squareroot_glrlm_RunVariance",
]
assert len(SELECTED_RADIO) == 12

# ===========================
# Données (identiques au script principal)
# ===========================
df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)

df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

missing = [c for c in SELECTED_RADIO if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du fichier : {missing}")

X = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
Y = df["Y"].values
D = df["D"].values
#Xs = StandardScaler().fit_transform(X.values)
Xs = X.values
n = len(Y)
print(f"n = {n} | p = {X.shape[1]} | avec oedème = {D.sum()} | sans oedème = {n - D.sum()}\n")

# ===========================
# MLP avec taille de couche cachée variable
# ===========================
def make_fit_predict(hidden):
    def fit_predict(Xtr, ytr, Xte, seed=0):
        clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", alpha=ALPHA,
                            max_iter=2000, early_stopping=True, random_state=seed)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1] 
    

# ===========================
# DML (même formule que le script principal)
# ===========================
def run_dml(fit_predict, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    g_hat = np.zeros(n)
    m_hat = np.zeros(n)
    for tr, te in skf.split(Xs, D):
        g_hat[te] = fit_predict(Xs[tr], Y[tr], Xs[te], seed=seed)
        m_hat[te] = fit_predict(Xs[tr], D[tr], Xs[te], seed=seed)
    m_hat = np.clip(m_hat, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)

    Y_res, D_res = Y - g_hat, D - m_hat
    theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / J ** 2 / n)
    pval = 2 * (1 - norm.cdf(abs(theta / se)))

    return {"theta": theta, "se": se, "pval": pval,
            "ll_g": log_loss(Y, np.clip(g_hat, 1e-6, 1 - 1e-6)),
            "ll_m": log_loss(D, m_hat)}

# ===========================
# Boucle sur les tailles et les graines
# ===========================
all_rows = []
for h in HIDDEN_SIZES:
    fp = make_fit_predict(h)
    for s in SEEDS:
        r = run_dml(fp, seed=s)
        r["h"], r["seed"] = h, s
        all_rows.append(r)
    print(f"taille {h:>2d} terminée")

res = pd.DataFrame(all_rows)
res["ll_total"] = res["ll_g"] + res["ll_m"]      # critère combiné

summary = res.groupby("h").agg(
    ll_g=("ll_g", "mean"),
    ll_m=("ll_m", "mean"),
    ll_total=("ll_total", "mean"),
    ll_total_sd=("ll_total", "std"),
    theta_moy=("theta", "mean"),
    theta_sd_graines=("theta", "std"),
    se_moy=("se", "mean"),
    pval_moy=("pval", "mean"),
)
summary["n_params"] = [12 * h + h + h + 1 for h in summary.index]

# Référence naïve : prévalence seule (aucun apprentissage)
ll_g_ref = log_loss(Y, np.full(n, Y.mean()))
ll_m_ref = log_loss(D, np.full(n, D.mean()))

print("\n" + "=" * 78)
print("RÉSULTATS PAR TAILLE DE COUCHE CACHÉE (moyenne sur", len(SEEDS), "graines)")
print("=" * 78)
print(summary.round(4).to_string())
print(f"\nRéférence prévalence seule : ll_g = {ll_g_ref:.4f} | ll_m = {ll_m_ref:.4f} "
      f"| total = {ll_g_ref + ll_m_ref:.4f}")

# ===========================
# Choix : le plus petit réseau dont le log-loss est proche du meilleur
# ===========================
best_h = summary["ll_total"].idxmin()
tol = summary.loc[best_h, "ll_total_sd"]
candidates = summary[summary["ll_total"] <= summary.loc[best_h, "ll_total"] + tol]
chosen = candidates.index.min()

print("\n" + "=" * 78)
print(f"Meilleur log-loss total : {best_h} neurones")
print(f"Règle de parcimonie (le plus petit réseau à moins d'un écart-type du meilleur) "
      f"-> {chosen} neurones")
print("=" * 78)

spread = summary["theta_moy"].max() - summary["theta_moy"].min()
print(f"\nSensibilité : theta moyen varie de {summary['theta_moy'].min():.4f} "
      f"à {summary['theta_moy'].max():.4f} (écart {spread:.4f}) selon la taille.")
print(f"SE moyen typique : {summary['se_moy'].mean():.4f}")
if spread < summary["se_moy"].mean():
    print("-> L'écart est plus petit que le SE : le résultat dépend peu de la taille du réseau.")
else:
    print("-> L'écart est plus grand que le SE : le résultat est sensible à la taille, à signaler.")

summary.round(4).to_csv("choix_neurones_resume.csv")
res.round(5).to_csv("choix_neurones_detail.csv", index=False)
print("\nFichiers : choix_neurones_resume.csv, choix_neurones_detail.csv")