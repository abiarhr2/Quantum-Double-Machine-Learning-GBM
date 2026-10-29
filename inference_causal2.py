import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Création DB fictif

df = pd.DataFrame({
    "experience": [20, 18, 22, 15, 2, 3, 4, 5], #Variable confondante
    "formation":  [1,  1,  1,  1, 0, 0, 0, 0], #traitement
    "performance":[95, 92, 98, 88, 70, 72, 74, 75] #résultat
})

print("="*50)
print("DONNEES")
print("="*50)
print(df)

# 2. Approche naïve ( estimation biasée )

treated_mean = df[df["formation"] == 1]["performance"].mean()
control_mean = df[df["formation"] == 0]["performance"].mean()

naive_effect = treated_mean - control_mean

print("\n" + "="*50)
print("APPROCHE NAIVE")
print("="*50)

print(f"Moyenne groupe traitement : {treated_mean:.2f}")
print(f"Moyenne groupe contrôle  : {control_mean:.2f}")
print(f"Effet estimé             : {naive_effect:.2f}")

# 3. Vérification du confondeur

print("\n" + "="*50)
print("VARIABLE CONFONDANTE")
print("="*50)

avg_exp_treated = df[df["formation"] == 1]["experience"].mean() 
avg_exp_control = df[df["formation"] == 0]["experience"].mean()

print(f"Expérience moyenne (traitement) : {avg_exp_treated:.2f}")
print(f"Expérience moyenne (contrôle)   : {avg_exp_control:.2f}")

print("\nObservation :")
print("Les employés formés ont beaucoup plus d'expérience.")
print("L'expérience influence probablement la performance.")
print("Elle est donc une variable confondante.")

# 4. Ajustement par régression (inclure les variables de confusion dans le modèle)

X = df[["formation", "experience"]]
y = df["performance"]

model = LinearRegression()
model.fit(X, y)

# le modele apprend une équation Performance = a × Formation + b × Experience + c

formation_coef = model.coef_[0] # coefficient de la formation
experience_coef = model.coef_[1] # coefficient de l experience

print("\n" + "="*50)
print("MODELE AJUSTE")
print("="*50)

print(f"Effet de la formation estimé  : {formation_coef:.2f}")
print(f"Effet de l'expérience estimé : {experience_coef:.2f}")

# 5. Comparaison contrefactuelle 

# Que se passerait-il pour la même personne si elle suivait la formation, puis si elle ne la suivait pas ?

print("\n" + "="*50)
print("ANALYSE CONTREFACTUELLE")
print("="*50)

experience_employee = 10

# Scénario sans formation
y0 = model.predict([[0, experience_employee]])[0]

# Scénario avec formation
y1 = model.predict([[1, experience_employee]])[0]

causal_effect = y1 - y0

print(f"Employé avec {experience_employee} ans d'expérience")

print(f"\nSans formation : {y0:.2f}")
print(f"Avec formation : {y1:.2f}")

print(f"\nEffet causal estimé : {causal_effect:.2f}")

# 

# 6. Conclusion

print("\n" + "="*50)
print("CONCLUSION")
print("="*50)

print(f"Effet naïf      : {naive_effect:.2f}")
print(f"Effet ajusté    : {causal_effect:.2f}")

print("""
L'approche naïve compare uniquement les moyennes.

Elle attribue toute la différence observée à la formation.

L'approche causale ajuste sur l'expérience
(variable confondante).

On compare alors deux scénarios identiques
(même expérience), où seule la formation change.

La différence obtenue est une estimation
de l'effet causal de la formation.
""")