# ====================================================

 #7. Diagnostic des modèles de nuisance + Forest Plot

 # =====================================================


import matplotlib

matploblibuse("Agg")  #enregistre les figures sans ouvrir de fenetre
import matplotlibpyplot as plt
from sklearnmatrics import roc_auc_score, log_loss

# Tableau de diagnostics

def diagnostic(res, name):
    #Récupération des prédictions des deux modèles:
    g, m = res["g_hat"], res["m_hat"]
    return {
        "Modèle" : name,
        "AUC g(x)" : roc_auc_score(Y, g), #Capacité à distinguer GSM et GBM
        "LogLoss g(x)" : log_loss(Y, g), #Qualité des probabilités prédites pour GSM
        "AUC m(x)" : roc_auc_score(D, m), #Capacité à distingues les patients avec et sans oedème
        "LogLoss m(x)" : log_loss(D, m), #Qualité des probabbilités prédites pour l oedème
        "J = mean(D_res²)": np.mean((D - m) ** 2), #Quantité des variation de l'oedeme restant après ajustement sur X
        
        # J mesure ce que le modèle n'arrive pas à expliquer concernant la présence d'œdème à partir des caractéristiques radiomiques X.
        # Pour estimer l'effet de l'œdème sur le type de tumeur (θ), le DML utilise la variation de l'œdème qui n'est pas expliquée par X.
        # - J très faible : il reste peu d'information pour distinguer l'effet de l'œdème de celui des caractéristiques radiomiques.
        # - J plus élevé : il reste davantage de variation exploitable pour estimer cet effet.
    }

# Référence naive: un modele qui ne cherche pas à apprendre à partir des caractéristiques des patients mais regarde seulement les proportions dans dataset et donne les memes prédictions à tout le monde

# sert à comparer les modèles NN et QNN avec un modèle qui ne fait aucun apprentissage, juste les proportiones du dataset
ref = {
    "Modèle": "Référence (prévalence seule)",

    # AUC g(X) = 0.5
    # Exemple :
    # Patient A : GSM -> le modèle prédit 45,4 %.
    # Patient B : GBM -> le modèle prédit 45,4 %.
    # Les deux ont le même score.
    # AUC = 0.5 car le modèle ne peut pas les distinguer.
    "AUC g(X)": 0.5, "LogLoss g(X)": log_loss(Y, np.full(n, Y.mean())),

    # AUC m(X) = 0.5
    # Exemple :
    # Patient A : avec œdème -> prédiction 88 %.
    # Patient B : sans œdème -> prédiction 88 %.
    # Le modèle ne distingue pas les deux patients.
    # Son AUC est donc 0.5.
    "AUC m(X)": 0.5, "LogLoss m(X)": log_loss(D, np.full(n, D.mean())),
    "J = mean(D_res²)": np.mean((D - D.mean()) ** 2),
}


diag = pd.DataFrame([
    diagnostics(res_classical, "NN classique"),
    diagnostics(res_quantum, "QNN quantique"),
    ref,
]).set_index("Modèle")

print("\n" + "=" * 70)
print("DIAGNOSTICS DES MODÈLES DE NUISANCE")
print("=" * 70)
print(diag.round(3).to_string())
diag.round(4).to_csv("diagnostics_nuisance.csv")
