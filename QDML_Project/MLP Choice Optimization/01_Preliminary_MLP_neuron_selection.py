"""

02_Optimisation_Alpha_MLP

CHOIX DE LA CONFIGURATION DU MODÈLE DE NUISANCE (12 variables radiomiques)

Compare, sur les données ORIGINALES, avec N_SEEDS découpages différents des plis :
  - MLP : taille de la couche cachée x alpha (régularisation L2) x early_stopping
  - Régression logistique L2 (plusieurs valeurs de C)
  - Référence "prévalence seule" (même probabilité pour tous)

CRITÈRE DE CHOIX (neutre, défini avant de regarder theta) :
  log-loss hors-échantillon de g(X) = P(Y|X) et m(X) = P(D|X), obtenu par cross-fitting.
  ll_total = ll_g + ll_m. Plus petit = meilleur.

COMPARAISON APPARIÉE : chaque configuration est comparée à la meilleure graine par graine
(même découpage des plis), ce qui retire une grande partie du bruit.

theta, SE et p-value sont affichés pour la SENSIBILITÉ seulement, jamais pour choisir.

Lancement : python choix_config_nuisance.py
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ===========================
# Paramètres
# ===========================
N_SEEDS = 30                       # nombre de découpages différents des plis
N_FOLDS = 5
PROPENSITY_CLIP = 0.01
N_JOBS = -1                        # tous les cœurs ; mets 1 pour désactiver le parallélisme
#HIDDEN_SIZES = [4, 8, 12, 16, 32]
HIDDEN_SIZES = [8, 16, 32]

#ALPHAS = [0.01, 0.1, 1.0, 10.0]
ALPHAS = [10.0, 30.0, 100.0]

#EARLY_STOPPING = [True, False]
EARLY_STOPPING = [False]

LOGISTIC_C = [0.01, 0.1, 1.0, 10.0]

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

# ===========================
# Configurations à comparer
# ===========================
def build_configs():
    cfgs = []
    for h, a, es in itertools.product(HIDDEN_SIZES, ALPHAS, EARLY_STOPPING):
        cfgs.append({"label": f"MLP h={h} a={a} es={'oui' if es else 'non'}",
                     "kind": "mlp", "h": h, "alpha": a, "es": es,
                     "n_params": 12 * h + h + h + 1})
    for c in LOGISTIC_C:
        cfgs.append({"label": f"Logistique C={c}", "kind": "logit", "C": c, "n_params": 13})
    cfgs.append({"label": "Prévalence seule", "kind": "prev", "n_params": 1})
    return cfgs

def fit_predict(cfg, Xtr, ytr, Xte, seed):
    if cfg["kind"] == "prev":
        return np.full(len(Xte), ytr.mean())
    if cfg["kind"] == "logit":
        clf = LogisticRegression(C=cfg["C"], max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(cfg["h"],), activation="relu",
                            alpha=cfg["alpha"],
                            max_iter=2000 if cfg["es"] else 500,
                            early_stopping=cfg["es"], random_state=seed)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

# ===========================
# DML sur les données originales (même formule que le script principal)
# ===========================
def run_task(cfg, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    g_hat = np.zeros(n)
    m_hat = np.zeros(n)
    for tr, te in skf.split(Xs, D):
        g_hat[te] = fit_predict(cfg, Xs[tr], Y[tr], Xs[te], seed)
        m_hat[te] = fit_predict(cfg, Xs[tr], D[tr], Xs[te], seed)
    g_hat = np.clip(g_hat, 1e-6, 1 - 1e-6)
    m_hat = np.clip(m_hat, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)

    Y_res, D_res = Y - g_hat, D - m_hat
    theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / J ** 2 / n)
    pval = 2 * (1 - norm.cdf(abs(theta / se)))
    ll_g, ll_m = log_loss(Y, g_hat), log_loss(D, m_hat)
    return {"label": cfg["label"], "seed": seed, "theta": theta, "se": se, "pval": pval,
            "ll_g": ll_g, "ll_m": ll_m, "ll_total": ll_g + ll_m}

# ===========================
# Exécution
# ===========================
if __name__ == "__main__":
    print(f"n = {n} | p = {X.shape[1]} | avec oedème = {D.sum()} | sans oedème = {n - D.sum()}")
    configs = build_configs()
    tasks = [(c, s) for c in configs for s in range(N_SEEDS)]
    print(f"{len(configs)} configurations x {N_SEEDS} graines = {len(tasks)} DML à lancer...")

    t0 = time.time()
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_task)(c, s) for c, s in tasks)
    print(f"Terminé en {time.time() - t0:.0f} s\n")

    detail = pd.DataFrame(rows)
    detail.round(5).to_csv("config_detail_alpha_extended.csv", index=False)

    npar = {c["label"]: c["n_params"] for c in configs}
    summary = detail.groupby("label").agg(
        ll_g=("ll_g", "mean"), ll_m=("ll_m", "mean"),
        ll_total=("ll_total", "mean"), ll_total_sd=("ll_total", "std"),
        theta_moy=("theta", "mean"), theta_sd_graines=("theta", "std"),
        se_moy=("se", "mean"), pval_moy=("pval", "mean"))
    summary["n_params"] = summary.index.map(npar)
    summary = summary.sort_values("ll_total")

    # ---- comparaison appariée à la meilleure configuration MLP ----
    piv = detail.pivot(index="seed", columns="label", values="ll_total")
    mlp_labels = [c["label"] for c in configs if c["kind"] == "mlp"]
    best_mlp = summary.loc[summary.index.isin(mlp_labels)].index[0]

    paired = []
    for lab in mlp_labels:
        d = piv[lab] - piv[best_mlp]                     # >0 : moins bon que le meilleur
        se_d = d.std(ddof=1) / np.sqrt(len(d))
        paired.append({"label": lab, "ecart_vs_meilleur": d.mean(), "SE_ecart": se_d,
                       "indiscernable": (d.mean() - 1.96 * se_d) <= 0})
    paired = pd.DataFrame(paired).set_index("label")
    summary = summary.join(paired[["ecart_vs_meilleur", "SE_ecart", "indiscernable"]])
    summary.round(4).to_csv("config_resume_alpha_extended.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("=" * 100)
    print("TOP 15 (trié par ll_total, moyenne sur", N_SEEDS, "graines)")
    print("=" * 100)
    print(summary.head(15).round(4).to_string())

    # ---- références ----
    prev = summary.loc["Prévalence seule"]
    best_logit = summary.loc[[l for l in summary.index if l.startswith("Logistique")]].iloc[0]
    print("\n" + "=" * 100)
    print("RÉFÉRENCES")
    print("=" * 100)
    print(f"Prévalence seule : ll_g = {prev['ll_g']:.4f} | ll_m = {prev['ll_m']:.4f} "
          f"| total = {prev['ll_total']:.4f}")
    print(f"Meilleure logistique ({best_logit.name}) : ll_g = {best_logit['ll_g']:.4f} "
          f"| ll_m = {best_logit['ll_m']:.4f} | total = {best_logit['ll_total']:.4f}")
    b = summary.loc[best_mlp]
    print(f"Meilleur MLP ({best_mlp}) : ll_g = {b['ll_g']:.4f} | ll_m = {b['ll_m']:.4f} "
          f"| total = {b['ll_total']:.4f}")
    print("-> Le meilleur MLP bat la prévalence seule pour g(X) ?", b["ll_g"] < prev["ll_g"])
    print("-> Le meilleur MLP bat la prévalence seule pour m(X) ?", b["ll_m"] < prev["ll_m"])
    print("-> Le meilleur MLP bat la meilleure logistique ?", b["ll_total"] < best_logit["ll_total"])

    # ---- choix : règle de parcimonie sur les configurations indiscernables du meilleur ----
    cands = summary.loc[summary.index.isin(mlp_labels) & (summary["indiscernable"] == True)]
    chosen = cands.sort_values(["n_params", "ll_total"]).index[0]
    c = summary.loc[chosen]
    print("\n" + "=" * 100)
    print(f"Meilleur MLP : {best_mlp}")
    print(f"Configurations MLP indiscernables du meilleur (écart apparié, IC 95 %) : {len(cands)}")
    print(f"CHOIX (le MLP le plus petit parmi elles) : {chosen}  ({int(c['n_params'])} paramètres)")
    print("=" * 100)

    print("\nSENSIBILITÉ de theta parmi les configurations indiscernables :")
    print(f"  theta de {cands['theta_moy'].min():.4f} à {cands['theta_moy'].max():.4f} "
          f"| SE moyen = {cands['se_moy'].mean():.4f}")

    print("\nFichiers : config_resume_alpha_extended.csv, config_detail_alpha_extended.csv")