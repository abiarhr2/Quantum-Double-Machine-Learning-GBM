# Validation effectuée

- Compilation des quatre sources Python en mémoire : réussie (sans créer de bytecode).
- Vérification des directives des deux jobs : 1 CPU, 4 Go, 24 h, activation de venv_qdml, logs stderr distincts, fins de ligne Unix.
- Aucun bootstrap de production exécuté et aucun job soumis.
- Toutes les modifications effectuées concernent exclusivement ce nouveau dossier ; les sources et résultats 04/05 n'ont pas été édités.
- Revue du prétraitement : tous les fit sont dans les plis d'entraînement ; entraînements g et m distincts ; groupes des copies bootstrap contrôlés.

Limite de validation : le Python accessible ici (`C:/msys64/ucrt64/bin/python.exe`) ne possède pas scikit-learn. Le test numérique court, la reproductibilité numérique et la reprise après interruption réelle n'ont donc pas été exécutés. Les commandes correspondantes sont documentées dans README.md ; ces vérifications restent à effectuer dans l'environnement DICO avant les calculs de production.

Limite fonctionnelle : les archives 04/05 ne permettent pas de prouver l'appariement complet demandé. Le script exporte les résultats descriptifs et un audit, mais bloque explicitement les intervalles des différences appariées en l'absence des manifestes historiques. Aucune équivalence de graines n'est présentée comme une preuve.
