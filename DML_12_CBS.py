"""
MODÈLE 3 — DOUBLE MACHINE LEARNING sur variables présélectionnées (CBS)
================================================================================
Contrairement aux Modèles 1 (Lasso) et 2 (Ridge), qui utilisaient les 1312
variables radiomiques brutes, ce modèle utilise seulement 12 variables
radiomiques présélectionnées indépendamment par Causal Ball Screening (CBS)
avec bootstrap et suppression des variables fortement corrélées.

X (confusion) = Age, Sex, Localization, Necrosis + 12 variables radiomiques CBS
                (p = 16 au total, contre 1312 dans les Modèles 1 et 2)
D (traitement) = Œdème (Y/N)
Y (résultat)   = type de tumeur (1 = GSM, 0 = GBM)

Avec p << n (16 vs 183), le besoin de régularisation forte disparaît : les
modèles de nuisance g(X) et m(X) utilisent une pénalité L2 légère, choisie
par validation croisée, essentiellement pour la stabilité numérique plutôt
que pour la sélection de variables. On s'attend donc à une estimation de
l'effet beaucoup plus stable qu'avec les 1312 variables brutes.
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

# ---------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------
df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)
df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

clinical_cat = ["Sex", "Localization", "Necrosis"]
X_clin = pd.get_dummies(df[clinical_cat], drop_first=True)
X_age = df[["Age "]].rename(columns={"Age ": "Age"})

# Les 12 variables radiomiques retenues par CBS (bootstrap + décorrélation)
selected_radio = [
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
X_radio = df[selected_radio].apply(pd.to_numeric, errors="coerce")

X = pd.concat([X_age, X_clin, X_radio], axis=1).fillna(0)
Y = df["Y"].values
D = df["D"].values
Xs = StandardScaler().fit_transform(X.values)
n = len(Y)

print(f"n = {n} | p = {X.shape[1]} (contre 1312 dans les Modèles 1 et 2)")
print(f"traitement : {D.sum()} avec œdème / {n - D.sum()} sans œdème")

# ---------------------------------------------------------------------------
# 2. Cross-fitting (L2 léger, car p << n : pas besoin de forte régularisation)
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
g_hat = np.zeros(n)
m_hat = np.zeros(n)

for train_idx, test_idx in skf.split(Xs, Y):
    Xtr, Xte = Xs[train_idx], Xs[test_idx]

    model_y = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=10,
                                    cv=3, max_iter=5000, random_state=RANDOM_STATE)
    model_y.fit(Xtr, Y[train_idx])
    g_hat[test_idx] = model_y.predict_proba(Xte)[:, 1]

    model_d = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=10,
                                    cv=3, max_iter=5000, random_state=RANDOM_STATE)
    model_d.fit(Xtr, D[train_idx])
    m_hat[test_idx] = model_d.predict_proba(Xte)[:, 1]

# ---------------------------------------------------------------------------
# 3-4. Orthogonalisation + estimation finale
# ---------------------------------------------------------------------------
Y_res = Y - g_hat
D_res = D - m_hat

theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
psi = (Y_res - theta * D_res) * D_res
J = np.mean(D_res ** 2)
sigma2 = np.mean(psi ** 2) / (J ** 2)
se = np.sqrt(sigma2 / n)
z = theta / se
pval = 2 * (1 - norm.cdf(abs(z)))
ci = (theta - 1.96 * se, theta + 1.96 * se)

print("\n=== Modèle 3 : DML sur 12 variables CBS ===")
print(f"theta   : {theta:+.4f}")
print(f"SE      : {se:.4f}")
print(f"z       : {z:.3f}")
print(f"p-value : {pval:.4f}")
print(f"IC 95%  : [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"m_hat range : {m_hat.min():.3f}-{m_hat.max():.3f}")
print(f"g_hat range : {g_hat.min():.3f}-{g_hat.max():.3f}")

# ---------------------------------------------------------------------------
# 5. Contrôle de stabilité : plusieurs graines de découpage en folds
#    (à comparer avec l'instabilité observée dans les Modèles 1 et 2)
# ---------------------------------------------------------------------------
print("\n=== Stabilité sur plusieurs graines (cross-fitting) ===")
for rs in [1, 42, 123]:
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rs)
    g2 = np.zeros(n); m2 = np.zeros(n)
    for tr, te in skf2.split(Xs, Y):
        my = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=10, cv=3,
                                   max_iter=5000, random_state=RANDOM_STATE)
        my.fit(Xs[tr], Y[tr]); g2[te] = my.predict_proba(Xs[te])[:, 1]
        md = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=10, cv=3,
                                   max_iter=5000, random_state=RANDOM_STATE)
        md.fit(Xs[tr], D[tr]); m2[te] = md.predict_proba(Xs[te])[:, 1]
    yr, dr = Y - g2, D - m2
    th = np.sum(dr * yr) / np.sum(dr ** 2)
    ps = (yr - th * dr) * dr
    j2 = np.mean(dr ** 2)
    se2 = np.sqrt(np.mean(ps ** 2) / (j2 ** 2) / n)
    p2 = 2 * (1 - norm.cdf(abs(th / se2)))
    print(f"seed={rs:3d} : theta={th:.4f}  p={p2:.4f}")