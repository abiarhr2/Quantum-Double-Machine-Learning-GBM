"""
DOUBLE MACHINE LEARNING sur 12 variables 
================================================================================
Ce script applique un modèle DML partiellement linéaire (Chernozhukov et al.,
2018) à un jeu de 12 variables radiomiques + variables cliniques de confusion.
La liste des 12 variables est un simple paramètre modifiable
(SELECTED_RADIO) : elles peuvent provenir de CBS, d'un autre screening, ou
être choisies manuellement — le code ne dépend d'aucune méthode de sélection
particulière.

X (confusion) = Age, Sex, Localization, Necrosis + 12 variables radiomiques
D (traitement) = Œdème (Y/N)
Y (résultat)   = type de tumeur (1 = GSM, 0 = GBM)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

RANDOM_STATE = 42
N_FOLDS = 5

#Est simplement la probabilité qu’un patient ait reçu le traitement \(D=1\), selon ses caractéristiques \(X\).
PROPENSITY_CLIP = 0.01   # clip m_hat away from {0,1} for numerical stability
#Je force les probabilités prédites à rester comprises entre 1 % et 99 %.

# --------------------------------------------------------------------------- 
# 0. >>> Les 12 variables 
# --------------------------------------------------------------------------- 
SELECTED_RADIO = [
    "original_shape_MajorAxis",
    "original_shape_Sphericity",
    "original_glcm_JointEntropy",
    "original_glcm_MaximumProbability",
    "log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformity",
    "log-sigma-5-0-mm-3D_firstorder_RootMeanSquared",
    "wavelet-LLL_firstorder_Kurtosis",
    "wavelet-LHH_firstorder_Variance",
    "wavelet-HHL_firstorder_Skewness",
    "wavelet-HHL_gldm_LowGrayLevelEmphasis",
    "squareroot_glrlm_HighGrayLevelRunEmphasis",
    "squareroot_glrlm_RunVariance",
]
assert len(SELECTED_RADIO) == 12, "SELECTED_RADIO doit contenir exactement 12 variables"

# --------------------------------------------------------------------------- 
# 1. Données 
# --------------------------------------------------------------------------- 
df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df.columns = df.columns.str.strip()                      # évite les pièges "Age " vs "Age"
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)

df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

missing = [c for c in SELECTED_RADIO if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes absentes du fichier : {missing}")

clinical_cat = ["Sex", "Localization", "Necrosis"]
X_clin = pd.get_dummies(df[clinical_cat], drop_first=True)
X_age = df[["Age"]]
X_radio = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce")

X = pd.concat([X_age, X_clin, X_radio], axis=1).fillna(0)
Y = df["Y"].values
D = df["D"].values
Xs = StandardScaler().fit_transform(X.values)
n = len(Y)

print(f"n = {n} | p = {X.shape[1]} (dont {len(SELECTED_RADIO)} variables radiomiques choisies)")
print(f"traitement : {D.sum()} avec œdème / {n - D.sum()} sans œdème")


# --------------------------------------------------------------------------- 
# 2. Fonction DML réutilisable (partially linear regression model)
# --------------------------------------------------------------------------- 
def fit_dml_plr(Xs, Y, D, n_folds=N_FOLDS, seed=RANDOM_STATE, clip=PROPENSITY_CLIP):
    """Cross-fitting + estimation orthogonalisée pour un modèle DML-PLR.

    Retourne theta, se, z, pval, ci, ainsi que g_hat/m_hat pour diagnostic.
    """
    n_ = len(Y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    g_hat = np.zeros(n_)
    m_hat = np.zeros(n_)

    # on stratifie sur D : c'est la variable la plus sensible à un découpage
    # déséquilibré (propension), et on réutilise les mêmes folds pour g_hat
    for train_idx, test_idx in skf.split(Xs, D):
        Xtr, Xte = Xs[train_idx], Xs[test_idx]

        model_y = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", Cs=10, cv=3, max_iter=5000,
            scoring="neg_log_loss", random_state=seed,
        )
        model_y.fit(Xtr, Y[train_idx])
        g_hat[test_idx] = model_y.predict_proba(Xte)[:, 1]

        model_d = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", Cs=10, cv=3, max_iter=5000,
            scoring="neg_log_loss", random_state=seed,
        )
        model_d.fit(Xtr, D[train_idx])
        m_hat[test_idx] = model_d.predict_proba(Xte)[:, 1]

    m_hat = np.clip(m_hat, clip, 1 - clip)

    Y_res = Y - g_hat
    D_res = D - m_hat

    theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
    psi = (Y_res - theta * D_res) * D_res
    J = np.mean(D_res ** 2)
    sigma2 = np.mean(psi ** 2) / (J ** 2)
    se = np.sqrt(sigma2 / n_)
    z = theta / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    ci = (theta - 1.96 * se, theta + 1.96 * se)

    return {
        "theta": theta, "se": se, "z": z, "pval": pval, "ci": ci,
        "g_hat": g_hat, "m_hat": m_hat,
    }


# --------------------------------------------------------------------------- 
# 3. Estimation principale 
# --------------------------------------------------------------------------- 
res = fit_dml_plr(Xs, Y, D)

print("\n=== DML (partially linear model) sur 12 variables ===")
print(f"theta   : {res['theta']:+.4f}")
print(f"SE      : {res['se']:.4f}")
print(f"z       : {res['z']:.3f}")
print(f"p-value : {res['pval']:.4f}")
print(f"IC 95%  : [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
print(f"m_hat range : {res['m_hat'].min():.3f}-{res['m_hat'].max():.3f}")
print(f"g_hat range : {res['g_hat'].min():.3f}-{res['g_hat'].max():.3f}")

# --------------------------------------------------------------------------- 
# 4. Contrôle de stabilité : plusieurs graines de découpage en folds 
# --------------------------------------------------------------------------- 
print("\n=== Stabilité sur plusieurs graines (cross-fitting) ===")
for rs in [1, 42, 123]:
    r = fit_dml_plr(Xs, Y, D, seed=rs)
    print(f"seed={rs:3d} : theta={r['theta']:.4f}  se={r['se']:.4f}  p={r['pval']:.4f}")