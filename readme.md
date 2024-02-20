
PROJET MACHINE LEARNING STREAMLIT
==========
Utilisation de Python et Streamlit pour proposer une application de machine learning : Data Management et application d'un modèle ScikitLearn.

Objectif : 
-------------------------------------------
Concevoir une application permettant d’entrainer et (potentiellement appliquer) des modèles de machine learning en fonction du jeu de donnée.

Entrée : 
-------------------------------------------
2 jeux de données -> 1 jeu de donnée de classification et 1 jeu de donnée pour la régression

Framework et langage: 
-------------------------------------------
Streamlit et Python

Etapes :
-------------------------------------------

Intégration des datasets sous un format attendu (csv)
Intégration des méthodes de nettoyage (fillna, correlation, sélection de features…)
Proposition de l’algorithme en fonction de la target détecté
Proposition éventuelle d’hyperparamètres potentiels
Proposition éventuelle d’un GridSearchCV
Proposition de graphique d’analyse des métrics après entrainement
Proposition de sauvegarde du modèle
 
Idée générale : 
-------------------------------------------
Recevoir un jeu de donnée, effectuer le plus de nettoyage possible, entrainer un algorithme et afficher les résultats de l’évaluation (metrics et ou graphique)
