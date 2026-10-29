import pandas as pd

# Approche naïve

df = pd.DataFrame({
    "age": [25, 30, 35, 40, 28, 32, 45, 38],
    "experience": [2, 5, 10, 15, 3, 7, 20, 12],
    "formation": [1, 1, 1, 1, 0, 0, 0, 0],
    "performance": [80, 85, 90, 95, 70, 75, 78, 82]
})

print(df)

treated = df[df["formation"] == 1]["performance"].mean()
control = df[df["formation"] == 0]["performance"].mean()

ate = treated - control

print("Effet moyen =", ate)

# Içi La formation semble augmenter la performance de 11.25 points, les personnes formées ont peut-être plus d'expérience.


from sklearn.ensemble import RandomForestRegressor

# T-learner

#on sépare les données selon la formation

treated_df = df[df["formation"] == 1]
control_df = df[df["formation"] == 0]

# Içi on cherche comment Pour les personnes formées, comment l'âge et l'expérience influencent-ils la performance ?

X_t = treated_df[["age", "experience"]]
y_t = treated_df["performance"]

# Pour les personnes non formées, comment l'âge et l'expérience influencent-ils la performance ?

X_c = control_df[["age", "experience"]]
y_c = control_df["performance"]

model_t = RandomForestRegressor()
model_c = RandomForestRegressor()

model_t.fit(X_t, y_t)
model_c.fit(X_c, y_c)


sarah = [[33, 8]] # 33 ans et 8 ans d'expérience

y1 = model_t.predict(sarah)[0]
y0 = model_c.predict(sarah)[0]

effect = y1 - y0

print("Y(1) =", y1)
print("Y(0) =", y0)
print("Effet causal =", effect)

#Y(1) = 88.9
#Y(0) = 76.27
#Effet causal = 13.63

#Pour Sarah, la formation augmenterait la performance d'environ 13 points.

from sklearn.ensemble import RandomForestRegressor

#S-learner

X = df[["age", "experience", "formation"]] # le traitement devient simplement une variable comme les autres
y = df["performance"]

model = RandomForestRegressor()
model.fit(X, y)

sarah_treated = [[33, 8, 1]]
sarah_control = [[33, 8, 0]]

y1 = model.predict(sarah_treated)[0]
y0 = model.predict(sarah_control)[0]

effect = y1 - y0

print(effect)

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

#econML

X = df[["age", "experience"]] #confounder
T = df["formation"] # traitement
Y = df["performance"] # resultat 

# on modelise directement x => y et x => t

model = CausalForestDML(
    model_y=RandomForestRegressor(), #Quel niveau de performance est attendu uniquement de l'âge et de l'expérience ?
    model_t=RandomForestRegressor(), #Quelle est la probabilité qu'une personne reçoive la formation ?
    random_state=42
)

#  Ce modèle sert à détecter le biais de sélection.
# il permet de retirer ce qui est expliqué par les confounders

model.fit(Y, T, X=X)

# Etapes du modele econml
# 1. Apprendre l'effet des confondeurs sur la performance : X => Y
# 2. Apprendre l 'effet des confounders sur le traitement: X => T
# 3.(double machine learning) Retirer ces effets de la performance(resultat) et aussi de la formation (traitement)
# et on travaille ensuite sur le reste (résidu-Y) et (résidu-T)
# puis econml étudie l effet du résidu-T sur le résidu-Y pour isoler l'effet causal
# 4. Construire une foret causale pour prédire l'effet causal
# 5. Estimer Effet causal individuel = Y(1)-Y(0) 



# Étapes du modèle EconML (CausalForestDML)
# 1. Apprendre la relation entre les variables de confusion (X) et le résultat (Y).
# X → Y
# Exemple :
# âge, expérience → performance
# Le modèle estime quelle partie de la performance est expliquée par les confondeurs.

# 2. Apprendre la relation entre les variables de confusion (X) et le traitement (T).
# X → T
# Exemple :
# âge, expérience → probabilité de recevoir une formation
# Le modèle estime quelles caractéristiques influencent l'attribution du traitement.

# 3. Double Machine Learning (DML)
# Le modèle retire l'influence des confondeurs à la fois sur :
# - le résultat (Y)
# - le traitement (T)
# On obtient alors :
# résidu_Y = Y - E[Y|X]
# résidu_T = T - E[T|X]
# où :
# E[Y|X] = performance attendue selon les confondeurs
# E[T|X] = probabilité attendue de recevoir le traitement
# Ces résidus représentent la partie de Y et de T qui n'est pas expliquée par les variables de confusion.

# 4. Estimation de l'effet causal
# Le modèle étudie la relation :
# résidu_T → résidu_Y
# Cette étape permet d'isoler l'effet causal du traitement en réduisant le biais dû aux variables de confusion.

# 5. Construction d'une forêt causale (Causal Forest)
# Plusieurs arbres sont construits afin d'estimer l'effet du traitement pour différents profils d'individus.
# Exemple :
# Employés juniors      → effet = +5
# Employés expérimentés → effet = +15
# Contrairement à un Random Forest classique qui prédit Y,
# la Causal Forest cherche à prédire l'effet causal du traitement.

# 6. Estimation des effets causaux individuels
# Pour un individu de caractéristiques X, le modèle estime :
# τ(X) = E[Y(1) - Y(0) | X]
# où :
# Y(1) = résultat potentiel avec traitement
# Y(0) = résultat potentiel sans traitement
# La différence τ(X) représente l'effet causal estimé pour ce profil d'individu.
# Exemple :
# X = [33 ans, 8 ans d'expérience]
# Y(1) = 88
# Y(0) = 76
# τ(X) = 88 - 76 = 12
# Interprétation :
# Pour une personne de 33 ans avec 8 ans d'expérience,
# la formation augmente la performance d'environ 12 points.