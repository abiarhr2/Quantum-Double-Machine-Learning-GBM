"""
MODÈLE 2 — DOUBLE MACHINE LEARNING avec pénalité RIDGE (L2)
================================================================================
Mêmes variables que le Modèle 1 (Lasso) :
  X (confusion) = Age, Sexe, Localization, Necrosis + 1308 variables radiomiques
  D (traitement) = Œdème (Y/N)
  Y (résultat)   = type de tumeur (1 = GSM, 0 = GBM)

Seule différence avec le Modèle 1 : penalty="l1" (Lasso) → penalty="l2" (Ridge)
dans les deux modèles de nuisance g(X) et m(X).

Ridge conserve toutes les variables (aucun coefficient à zéro) et réduit leur
amplitude, au lieu de sélectionner un sous-ensemble comme le Lasso. Plus stable
en présence de variables radiomiques corrélées entre elles, mais moins
parcimonieux et moins interprétable.
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
# 1. Données (identique au Modèle 1)
# ---------------------------------------------------------------------------
df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)
df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

#Varibales Qualitatives
clinical_cat = ["Sex", "Localization", "Necrosis"]
X_clin = pd.get_dummies(df[clinical_cat], drop_first=True)
X_age = df[["Age "]].rename(columns={"Age ": "Age"})
radiomics_cols = [c for c in df.columns
                   if c not in ["name", "Sex", "Age ", "Localization",
                                "Necrosis", "Edema", "Y", "D"]]
X_radio = df[radiomics_cols].apply(pd.to_numeric, errors="coerce")
X = pd.concat([X_age, X_clin, X_radio], axis=1).fillna(0)
Y = df["Y"].values
D = df["D"].values
Xs = StandardScaler().fit_transform(X.values)
n = len(Y)

print(f"n = {n} | p = {X.shape[1]} | traitement : {D.sum()} avec œdème / {n - D.sum()} sans")

# ---------------------------------------------------------------------------
# 2. Cross-fitting avec nuisance RIDGE (L2) — au lieu du Lasso (L1)

# Le cross-fitting est utilisé pour éviter le surapprentissage des modèles
# de nuisance g(X) et m(X).
#
# Principe :
# Les données sont divisées en plusieurs folds (ici 5). À chaque itération,
# les modèles sont entraînés sur 4 folds et testés sur le fold restant.
# Ainsi, chaque patient reçoit une prédiction provenant d'un modèle qui n'a
# jamais été entraîné sur lui, ce qui limite le risque de mémorisation des
# données et améliore la validité de l'estimation.
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
g_hat = np.zeros(n)
m_hat = np.zeros(n)

for train_idx, test_idx in skf.split(Xs, Y):
    Xtr, Xte = Xs[train_idx], Xs[test_idx]

    model_y = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=8, # "Teste automatiquement 8 niveaux de pénalisation différents."
                                    cv=3, max_iter=5000, random_state=RANDOM_STATE)
    model_y.fit(Xtr, Y[train_idx])
    g_hat[test_idx] = model_y.predict_proba(Xte)[:, 1]

    model_d = LogisticRegressionCV(penalty="l2", solver="lbfgs", Cs=8,
                                    cv=3, max_iter=5000, random_state=RANDOM_STATE)
    model_d.fit(Xtr, D[train_idx])
    m_hat[test_idx] = model_d.predict_proba(Xte)[:, 1]

#

# ---------------------------------------------------------------------------
# 3-4. Orthogonalisation + estimation finale (identique au Modèle 1)
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

print("\n=== Modèle 2 : DML + Ridge ===")
print(f"theta   : {theta:+.4f}")
print(f"SE      : {se:.4f}")
print(f"z       : {z:.3f}")
print(f"p-value : {pval:.4f}")
print(f"IC 95%  : [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"m_hat range : {m_hat.min():.3f}-{m_hat.max():.3f}")
print(f"g_hat range : {g_hat.min():.3f}-{g_hat.max():.3f}")