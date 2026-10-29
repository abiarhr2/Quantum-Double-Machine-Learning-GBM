"""
DOUBLE MACHINE LEARNING (DML) + LASSO
=====================================

Effet de l'œdème (D) sur le type tumoral (Y)

Y :
    1 = Gliosarcome (GSM)
    0 = Glioblastome (GBM)

D :
    1 = Œdème présent
    0 = Pas d'œdème

X :
     toutes les variables radiomiques

Méthode :
    Double Machine Learning (Partialling-Out)
    Cross-fitting 5 folds
    Logistic Regression Lasso (L1)


Comme j’ai beaucoup plus de variables que de patients, j’utilise le Lasso.
Le Lasso réduit certains coefficients jusqu’à zéro et effectue donc une sélection automatique des variables utiles aux modèles de nuisance.

    
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm

# PARAMETRES

RANDOM_STATE = 42
N_FOLDS = 5

# 1. CHARGEMENT DES DONNEES

df = pd.read_excel("data/GBM_GBS.xlsx", header=3)

df = (
    df
    .dropna(subset=["name"])
    .dropna(axis=1, how="all")
    .reset_index(drop=True)
)

# Outcome Y
df["Y"] = ( #si gsm = 1, si gbm = 0
    df["name"]
    .astype(str)
    .str.endswith("GSM")
    .astype(int)
)

# Treatment D
df["D"] = (
    df["Edema"] == "Y"
).astype(int)

# ------------------------------------------------------------------
# 2. CONSTRUCTION DES COVARIABLES
# ------------------------------------------------------------------

clinical_cat = ["Sex", "Localization", "Necrosis"]

X_clin = pd.get_dummies( #transforme les catégories en colonnes binaires 0/1
    df[clinical_cat],
    drop_first=True
)

X_age = df[["Age "]].rename( #recupère la colonne Age
    columns={"Age ": "Age"}
)

radiomics_cols = [ #On parcours toutes les colonnes du dataset et on enlève celles qui ne sont pas considérées commes des features radiomiques
    c for c in df.columns
    if c not in [
        "name",
        "Sex",
        "Age ",
        "Localization",
        "Necrosis",
        "Edema",
        "Y",
        "D"
    ]
]

X_radio = ( # conversion des features radiomiques en nombres
    df[radiomics_cols]
    .apply(pd.to_numeric, errors="coerce") # une valeur impossible à convertir devient NaN
)

X = pd.concat( # On rassemble horizontalement : X = âge + données cliniques + données radiomiques
    [X_age, X_clin, X_radio],
    # on les sépare uniquementt prq ils n ont pas le meme format
    axis=1 # on ajoute les blocs comme colonnes, pas comme nouvelles lignes
).fillna(0) # remplace toutes les valeurs manquantes par 0

Y = df["Y"].values # récuperation du resulats
D = df["D"].values # récupèration du traitement

# ------------------------------------------------------------------
# 3. STANDARDISATION
# ------------------------------------------------------------------

Xs = StandardScaler().fit_transform(X) #calcule moyenne et écart type de chaque colonne et utilise ces valeurs pour standardiser le X

n = len(Y) # nombre de patients
p = X.shape[1] # nombre de colonnes, de covariables

print("=" * 60)
print("DESCRIPTION DES DONNEES")
print("=" * 60)
print(f"Patients : {n}")
print(f"Covariables : {p}")
print(f"GSM : {Y.sum()}")
print(f"GBM : {(Y == 0).sum()}")
print(f"Oedème oui : {D.sum()}")
print(f"Oedème non : {(D == 0).sum()}")

# ------------------------------------------------------------------
# 4. CROSS-FITTING: séparer plusieurs fois les patients en un groupe d'entrainement et un groupe de test
# ------------------------------------------------------------------

g_hat = np.zeros(n) # la prédiction du modèle associé à Y
m_hat = np.zeros(n) # la prédiction du modèle associé à D

selected_y = [] # selected_y → nombre de features sélectionnées pour prédire Y à chaque fold
selected_d = [] # selected_d → nombre de features sélectionnées pour prédire D à chaque fold

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for train_idx, test_idx in skf.split(Xs, Y):

    # train_idx → indices des patients pour l'entraînement
    # test_idx  → indices des patients pour le test
    

    Xtr = Xs[train_idx] #données pour entrainement
    Xte = Xs[test_idx]  #données pour test

    # --------------------------------------------------------------
    # g(X) = E[Y|X]
    # --------------------------------------------------------------

    model_y = LogisticRegressionCV(
        penalty="l1", #LASSO
        solver="liblinear",
        cv=3,
        Cs=10,
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    model_y.fit(Xtr, Y[train_idx]) # Apprends la relation entre les caractéristiques X des patients d'entraînement et leur valeur Y

    selected_y.append(
        np.sum(model_y.coef_.ravel() != 0)
    )

    g_hat[test_idx] = (
        model_y.predict_proba(Xte)[:, 1] #prend les patients qu'il n'a pas utilisés pour son entraînement et calcule leur probabilité d'appartenir à la classe 1
    )

    # --------------------------------------------------------------
    # m(X) = E[D|X] : Modèle cherche à prédire D (le tratement) à partir de X
    # --------------------------------------------------------------

    model_d = LogisticRegressionCV(
        penalty="l1", #LASSO
        solver="liblinear",
        cv=3,
        Cs=8,  
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    model_d.fit(Xtr, D[train_idx])

    selected_d.append( # Nombre de variable retenues pour prédire Y
        np.sum(model_d.coef_.ravel() != 0)
    )

    m_hat[test_idx] = ( # Prediction de D 
        model_d.predict_proba(Xte)[:, 1] #prend les patients qu'il n'a pas utilisés pour son entraînement et calcule leur probabilité d'avoir l oedeme
    )

# ------------------------------------------------------------------
# 5. ORTHOGONALISATION: Retirer de Y et de D ce qui est expliqué par X, puis mesurer la relation entre ce qu'il reste
# ------------------------------------------------------------------

Y_res = Y - g_hat
D_res = D - m_hat

# ------------------------------------------------------------------
# 6. ESTIMATION DML: 
# ------------------------------------------------------------------

theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2) #^ mesure l'effet de D sur Y après avoir retiré l'effet des variables de contrôle.

#fonction d'influence de chaque observation : 
# 	la contribution de chaque patient à l'estimation finale.
# à l'estimation de θ. Elle indique si l'observation pousse θ vers le
# haut (ψ_i > 0) ou vers le bas (ψ_i < 0).
psi = (Y_res - theta * D_res) * D_res

J = np.mean(D_res ** 2)


#Variance asymptotique
## 	la variabilité globale des contributions des patients.
# Plus les ψ_i sont dispersés, plus l'estimation est incertaine.
sigma2 = np.mean(psi ** 2) / (J ** 2)

#Erreur standard: mesure l incertitude autour de l'estimateur theta
# une petite valeur indique une estimation plus précise.
se = np.sqrt(sigma2 / n)

# Statistique de test z pour H0 : θ = 0 : On suppose au départ que l'effet de D sur Y est nul.
# Elle mesure combien d'erreurs standards séparent θ̂de 0.
z = theta / se

#p-value associée au test bilatéral H0 : θ = 0.
# Elle indique si l'effet estimé est statistiquement significatif.
pval = 2 * (1 - norm.cdf(abs(z)))

ci_low = theta - 1.96 * se
ci_high = theta + 1.96 * se

# ------------------------------------------------------------------
# 7. RESULTATS
# ------------------------------------------------------------------

print("\n")
print("=" * 60)
print("RESULTATS DML + LASSO")
print("=" * 60)

print(f"Theta            : {theta:.4f}")
print(f"Erreur standard  : {se:.4f}")
print(f"z-score          : {z:.4f}")
print(f"p-value          : {pval:.4f}")
print(f"IC95%            : [{ci_low:.4f} ; {ci_high:.4f}]")

print("\nSELECTION LASSO")
print("-" * 60)

print(
    f"Variables retenues pour g(X)=E[Y|X] : "
    f"{np.mean(selected_y):.1f}"
)

print(
    f"Variables retenues pour m(X)=E[D|X] : "
    f"{np.mean(selected_d):.1f}"
)

print("\nINTERPRETATION")

print(
    f"La présence d'un œdème est associée à "
    f"{theta*100:+.1f} points de pourcentage "
    f"de probabilité d'être classé comme "
    f"gliosarcome plutôt que glioblastome, "
    f"après ajustement sur l'ensemble des "
    f"covariables cliniques et radiomiques."
)