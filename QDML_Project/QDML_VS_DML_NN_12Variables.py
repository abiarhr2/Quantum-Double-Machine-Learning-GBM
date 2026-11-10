"""
NOO

DML QUANTIQUE (QNN) VS QML CLASIQUE (NN) - sur les memes variables

================================================================================
Ce script compare deux versions du même modèle DML partiellement linéaire :
 
  - Version « classique »  : g_hat(X) et m_hat(X) sont estimés par un réseau
                              de neurones classique (MLPClassifier, sklearn).
  - Version « quantique »  : g_hat(X) et m_hat(X) sont estimés par un réseau
                              de neurones quantique (circuit variationnel,
                              simulé classiquement avec PennyLane).

Dans les deux cas, EXACTEMENT la même formule DML (cross-fitting,
orthogonalisation, theta, SE, p-value, IC) est utilisée. Seule la boîte noire
qui prédit g_hat/m_hat change. Cela permet une comparaison propre : la
différence observée vient du choix du modèle de nuisance, et non de la
méthode causale elle-même.    

"""


import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
import time

import pennylane as qml
from pennylane import numpy as pnp

RANDOM_STATE = 42
N_FOLDS = 5
PROPENSITY_CLIP = 0.01

#===========================

# les 12 variables

#===========================


SELECTED_RADIO = [
    "original_shape_MajorAxis", "original_shape_Sphericity", "original_glcm_JointEntropy", "original_glcm_MaximumProbability", 
               "log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformity", "log-sigma-5-0-mm-3D_firstorder_RootMeanSquared", "wavelet-LLL_firstorder_Kurtosis",
               "wavelet-LHH_firstorder_Variance", "wavelet-HHL_firstorder_Skewness", "wavelet-HHL_gldm_LowGrayLevelEmphasis", "squareroot_glrlm_HighGrayLevelRunEmphasis", "squareroot_glrlm_RunVariance"
]


assert len(SELECTED_RADIO) == 12, "SELECTED_RADIO doit contenir exactement 12 variables"

#=======================

#1. Données

#=======================

df = pd.read_excel("data/GBM_GBS.xlsx", header=3)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True) #supprime les lignes où la colonnes name contient une valeur manquante, et supprime les colonnes dont toutes les valeurs sont manquantes et réinitialise les indices dees ligens à partir de 0



df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
df["D"] = (df["Edema"] == "Y").astype(int)

missing = [c for c in SELECTED_RADIO if c not in df.columns]

if missing:
    raise ValueError(f"Colonnes absentes du fichier : {missing}")

clinical_cat = ["Sex", "Localization", "Necrosis"]
X_clin = pd.get_dummies(df[clinical_cat], drop_first=True) # tranforme les variables cliniques categorielles en varibales numériques
X_age = df[["Age"]]
X_radio = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce") #remplace les valeurs impossibles à convertir par NaN


X = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())          # médiane plutôt que 0

Y = df["Y"].values
D = df["D"].values
Xs = StandardScaler().fit_transform(X.values)
n = len(Y)

print(f"n = {n} | p = {X.shape[1]}")
print(f"traitemnet : {D.sum()} avec oedème / {n - D.sum()} sans oedème\n")


#==================================

#2.Modèle de nuisance classique : réseau de neurones (MLP, sklearn) Multilayer Perceptron

#==================================

def classical_nn_fit_predict(Xtr, ytr ,Xte, seed = RANDOM_STATE):
    """
    J ai choisi un réseau de neurones peu profond afin de limiter le risque de surapprentissage lié à la petite tailles de l'échantillon
    tout en permettant au modèle de capturer des relations non linéaires
    """


    clf = MLPClassifier( # multi layer perceptron
        hidden_layer_sizes=(16,), #une seule couche cachée contenant 16 neurones
        activation= "relu", #fonction d'actication qui permet d'apprendre des relations non linéaires
        #Si le résultat est positif, elle le conserve.
        #Si le résultat est négatif, elle le remplace par zéro.

        alpha=1e-2, #Régularisation L2 (0,01) pour limiter le surapprentissage
        max_iter=2000, #Autorise au maximum 2 000 itérations d'entraînement
        early_stopping=True, #Arrête l'entraînement si le modèle ne s'améliore plus sur les données de validation
        random_state=seed, #Rend les opérations aléatoires reproductibles.
    )

    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:,1] # Retourne la probabilité prédite de la classe 1 (GSM pour Y, œdème présent pour D).

#===================================

#3. Modèle de nuisance quantique : circuit variationnel (QNN, Pennylane)

#===================================

"""
Un qubit par variable d'entrée (pas de compression PCA) : chaque colonne de
X est encodée sur son propre qubit. Avec p = X.shape[1] variables (12
radiomiques + confusion clinique)

J ai pas utilisé le PCA içi, une méthode de réduction de dimension qui transforme une grand nombre de varibales en un nombre plus petit de nouvelles varibales appelées composantes prinicpales,
parce que je voulais conserver toutes les varibales radiomiques originales dans le modèle
"""

N_QUBITS = X.shape[1] # une variable par qubit, aucune compression
N_LAYERS = 1 #Une couvhe quantique entrainable
QNN_EPOCHS = 40 # Nombre de passages complets sur les données pour entraîner le circuit quantique.
QNN_LR = 0.2 # Taux d'apprentissage : contrôle la taille des mises à jour des paramètres quantiques par Adam.


_dev = qml.device("lightning.qubit", wires=N_QUBITS) #creation du simulateur quantique sur lequel le QNN va fonctionner


#variational quantum circuit
@qml.qnode(_dev, diff_method="adjoint")

def _quantum_circuit(weights, x): 
    #encode les données classiques dans les angles de rotation des qubits
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation = "Y") # porte RY

    #Couche entrainables du QNN: rotations parametrées par weights et portes d'intrication
    #couches entrainables ( l'equivalent quantique des couches d'un MLP)
    #les rotationes et l'intrication sont controlées par les paramètres weights

    #Cette ligne demande au circuit quantique de faire deux opérations :
    #1. Rotations : chaque qubit tourne selon trois angles contenus dans weights. Ces angles sont ajustés pendant l'entraînement pour améliorer les prédictions.
    #2. Intrication : le circuit applique des portes quantiques qui relient les qubits pour traiter leurs informations ensemble.
    qml.StronglyEntanglingLayers(weights, wires= range(N_QUBITS))

    #Mesure le qubit 0 et retourne une valeur entre -1 et 1
    #la mesure transforme l'état quantique final en une valeur numérique exploitable. Dans mon code, cette valeur est ensuite convertie en probabilité pour prédire Y ou D.
    return qml.expval(qml.PauliZ(0))  #Calculer la valeur moyenne de la mesure Pauli-Z sur le qubit numéro 0.
    #Cette valeur joue le même rôle que la sortie d'un neurone dans un réseau classique.
    #Grâce à l'intrication, le qubit 0 contient indirectement de l'information provenant des autres qubits.

def _quantum_predict_proba(weights, X):
    raw = pnp.array([_quantum_circuit(weights, x) for x in X])
    return (1 - raw) / 2

def quantum_nn_fit_predict(Xtr, ytr, Xte, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(Xtr)
    Xtr_q = scaler.transform(Xtr)
    Xte_q = scaler.transform(Xte)

    #Initialiser les parametres entrainables
    weights = pnp.array(
        #initialisation des angles aleatoirement
        0.1*rng.randn(N_LAYERS, N_QUBITS,3), #PennyLane définit StronglyEntanglingLayers avec trois paramètres de rotation par qubit et par couche.
        requires_grad= True
    )
    opt = qml.AdamOptimizer(stepsize=QNN_LR)
    ytr_arr = pnp.array(ytr, dtype=float)
 
    def cost(w):
        p_hat = _quantum_predict_proba(w, Xtr_q)
        p_hat = pnp.clip(p_hat, 1e-6, 1 - 1e-6)
        return -pnp.mean(ytr_arr * pnp.log(p_hat) + (1 - ytr_arr) * pnp.log(1 - p_hat))
 
    for _ in range(QNN_EPOCHS):
        weights, _ = opt.step_and_cost(cost, weights)
 
    return np.array(_quantum_predict_proba(weights, Xte_q))


#================================================================

#4. Modèle DML générique

#================================================================

def run_dml(nuisance_fit_predict, Xs, Y, D, n_folds=N_FOLDS, SEED=RANDOM_STATE, clip=PROPENSITY_CLIP, label=""):
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED ) #StratifiedKFold essaie de conserver approximativement la même proportion de patients avec et sans œdème dans chaque fold. 

    #Deux tableaux
    # g_hat : probabilité prédite de GSM (Y=1)
    # m_hat : probabilité prédite d'œdème (D=1)

    g_hat = np.zeros(n)
    m_hat = np.zeros(n)


    #Cross-Fitting
    fold = 0
    for train_idx, test_idx in skf.split(Xs, D): # La stratification sur D permet de mieux répartir ces deux groupes dans les cinq folds, notamment pour entraîner m̂(X).
        fold +=1
        print(f"[{label}] pli {fold}/{n_folds}...")

        #On obtient les caractéristiques des patients d'entrainement et de test
        Xtr, Xte = Xs[train_idx], Xs[test_idx]

        #Prédire le type tumoral
        g_hat[test_idx] = nuisance_fit_predict(Xtr, Y[train_idx], Xte, seed=SEED)

        #Prédire l'oedème
        m_hat[test_idx] = nuisance_fit_predict(Xtr, D[train_idx], Xte, seed=SEED) # score de propension: Le score de propension est la probabilité estimée qu'un patient présente un œdème, compte tenu de ses caractéristiques radiomiques X.

    m_hat = np.clip(m_hat, clip, 1-clip) # empeche les score de propension d'etre prohe de 0 ou de 1 pour limiter les prédictions extremes

    #Partialling-Out
    Y_res = Y - g_hat
    D_res = D - m_hat

    # Theta représente l'effet étudié de D sur Y après ajustement sur les caractéristiques X.
    theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)

    #Calcule le score psi pour chaque patient. Il intervient dans l'estimation de l'incertitude
    # statistique associée à theta.
    #La contribution de chaque patient à l'incertiture
    psi = (Y_res - theta * D_res) * D_res

    #La variation résiduele de D
    J = np.mean(D_res ** 2)

    # Estime la variance asymptotique
    sigma2 = np.mean(psi ** 2) / (J ** 2)

    # Mesure l'incertitude statistique de theta, Plus elle est petite, plus l'estimation est généralement précise.
    se = np.sqrt(sigma2 / n)

    # Mesure la distance entre theta et zéro, exprimée en nombre d'erreurs standards.
    #Un z-score proche de zéro indique que l'estimation n'est pas très éloignée de l'hypothèse nulle relativement à son incertitude.
    z = theta / se

    

    # Test bilatéral de l'hypothèse H0 : theta = 0. Utilise l'approximation normale asymptotique.
    pval = 2 * (1 - norm.cdf(abs(z)))

    #Si p value > 0.05: Le résultat n'est pas statistiquement significatif au seuil de 5 %. Cela ne prouve pas que l'association est exactement nulle. Cela signifie que tes données ne fournissent pas suffisamment de preuves pour rejeter l'hypothèse θ = 0.

    #Calcule les bornes inférieure et supérieure de l'intervalle de confiance de theta.
    #sous une approximation normale, environ 95 % de la distribution se trouve entre −1,96 et +1,96 erreurs standards autour de sa moyenne.
    ci = (theta - 1.96 * se, theta + 1.96 * se)

    return {"theta": theta, "se": se, "z": z, "pval": pval, "ci": ci,
            "g_hat": g_hat, "m_hat": m_hat}



#=============================================================

#5. Les deux DML

#=============================================================

print("===DML classique (réseau de neurons MLP) ===")
t0 = time.time()
res_classical = run_dml(classical_nn_fit_predict, Xs, Y, D, label="NN classique")
time_classical = time.time() - t0
print(f" => terminé en {time_classical:.1f} s")

print("\n=== DML quantique (réseau de neurones quantique / QNN) ===")
t0 = time.time()
res_quantum = run_dml(quantum_nn_fit_predict, Xs, Y, D, label="QNN")
time_quantum = time.time() - t0
print(f"  -> terminé en {time_quantum:.1f} s")


#===============================================================

#6. Comparaison finales

#===============================================================

print(f"{'':15s}{'theta':>10s}{'SE':>10s}{'p-value':>10s}{'IC 95 %':>22s}{'temps':>12s}")
for name, r, t in [("Classique (NN)", res_classical, time_classical),
                    ("Quantique (QNN)", res_quantum, time_quantum)]:
    ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
    print(f"{name:15s}{r['theta']:10.4f}{r['se']:10.4f}{r['pval']:10.4f}{ci_str:>22s}{t:11.1f}s")
 


 