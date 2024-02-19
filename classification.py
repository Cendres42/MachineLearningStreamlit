import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from style import styled_write, styled_write2



def modelLogistic():
	model = LogisticRegression()
	# Définir la grille des hyperparamètres à rechercher
	param_grid = {
		'penalty': ['l1', 'l2'],
		'C': [0.001, 0.01, 0.1, 1, 10, 100],
		'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
		'max_iter': [50, 100, 200],
		'fit_intercept': [True, False],
		'class_weight': [None, 'balanced']
	}
	# Initialiser la recherche sur grille avec une validation croisée (5 plis dans cet exemple)
	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
	return grid_search

def modelRandomForest():
	model = RandomForestClassifier()
	param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
	}
	# Initialiser GridSearchCV avec le modèle et les hyperparamètres
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	return grid_search


def modelKNeighbors():
	param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
	}
	# Initialiser le modèle KNeighborsClassifier
	model = KNeighborsClassifier()
	# Initialiser GridSearchCV avec le modèle et les hyperparamètres
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	return grid_search


def classificationChoice(df,selected_model):
		# Séparez les features et la target
	y = df['target']
	X = df.drop('target', axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	if selected_model =="LogisticRegression":
		grid_search=modelLogistic()
	elif selected_model == "RandomForestClassifier":
		grid_search=modelRandomForest()
	elif selected_model == "KNeighborsClassifier":
		grid_search=modelKNeighbors()
	else:
		return ("NoData")

	#entrainement du modèle
	styled_write("Les hyperparamètres optimaux ont été appliqués grâce à GridSearchCV")
	grid_search.fit(X_train, y_train)
	# Prédictions
	y_pred = grid_search.predict(X_test)
	styled_write(f"Un extrait des prédictions de classification du vin")
	st.table(y_pred[0:7])
	y_prob = grid_search.predict_proba(X_test)
	styled_write(f"Un extrait des probabilités de classification du vin")
	st.table(y_prob[0:7])
	
	h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 2 - Evaluation du modèle - Metrics </h2>'
	st.markdown(h2_title, unsafe_allow_html=True)
	accuracy = accuracy_score(y_test, y_pred)
	styled_write(f"Précision sur l'ensemble de test : {accuracy}")
	styled_write(f"<p style= 'font-weight:bold;'>Matrice de confusion</p>")
	cm = confusion_matrix(y_test, y_pred)
	st.table(cm)
	styled_write(f"<p style= 'font-weight:bold;'>Aire sous la courbe ROC</p>")
	roc=roc_auc_score(y_test, grid_search.predict_proba(X_test),multi_class='ovr',average='macro')
	styled_write(f"Probabilité que le modèle attribue un score plus élevé à une instance positive par rapport à une instance négative choisie au hasard : {roc}")
	styled_write("Un modèle parfait aurait une ROC AUC égale à 1, tandis qu'un modèle aléatoire aurait une ROC AUC de 0.5.")
	cr = classification_report(y_test, y_pred)
	st.text("Rapport de Classification :\n\n{}".format(cr))




	
	
