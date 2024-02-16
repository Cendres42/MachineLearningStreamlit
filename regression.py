import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np


def styled_write(content):
    st.write(
        f"""
        <div style="background-color: #FFFFFF; padding: 10px; border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def findAlphaforRidge(X_train,y_train):
	param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
	# Utilisez GridSearchCV pour trouver la meilleure valeur d'alpha
	model = Ridge()
	grid_search = GridSearchCV(model, param_grid, cv=5)
	grid_search.fit(X_train, y_train)
	# Obtenez la meilleure valeur d'alpha
	best_alpha = grid_search.best_params_['alpha']
	return best_alpha

def modelRidge(best_alpha):
	styled_write("La valeur d'alpha est optimisée grâce à GridSearchCV mais vous pouvez la modifier")
	alpha = st.slider("Alpha (Lasso)", min_value=0.0, max_value=1.0, value=best_alpha)
	model = Ridge(alpha=best_alpha)
	# Obtenez la meilleure valeur d'alpha
	return model

def modelLinear():
	model = LinearRegression()
	return model

def modelLasso():
	alpha = st.slider("Alpha (Lasso)", min_value=0.0, max_value=1.0, value=0.1)
	model = Lasso(alpha=alpha)
	return model

def validationCroisee(df,model):
	styled_write("Voici les résultats de la validation croisée du KFold")
	X=df.drop('target', axis=1)
	y=df['target']
	# Définir le nombre de folds pour la validation croisée (par exemple, k=5)
	k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
	# Initialiser une liste pour stocker les scores de performance (par exemple, MSE) pour chaque fold
	moyennes=[]
	scores = []
	# Effectuer la validation croisée
	for train_index, test_index in k_fold.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		# Entraîner le modèle
		model.fit(X_train, y_train)
		# Faire des prédictions sur l'ensemble de test
		y_pred = model.predict(X_test)
		# Calculer le score (MSE dans cet exemple)
		mse = round(mean_squared_error(y_test, y_pred),2)
		moyennes.append(mse)
		Rscore=round(r2_score(y_test,y_pred),2)
		scores.append(Rscore)

	# Afficher les scores pour chaque fold
	for fold, myne in enumerate(moyennes, start=1):
		styled_write(f"Fold {fold}: MSE = {myne}")
	for fold, score in enumerate(scores, start=1):
		styled_write(f"Fold {fold}: Rscore={score}")
	# Afficher la moyenne des scores
	styled_write(f"Moyenne des MSE sur tous les folds : {sum(scores) / len(scores)}")
	

def regressionChoice(df,selected_model):
		# Séparez les features et la target
	y = df['target']
	X = df.drop('target', axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Créez un menu déroulant pour choisir le modèle
	if selected_model =="Linear Regression":
		#X=[['age','sex','bmi','bp']]
		model=modelLinear()
	elif selected_model == "Ridge Regression":
		best_alpha=findAlphaforRidge(X_train,y_train)
		model=modelRidge(best_alpha)
	elif selected_model == "Lasso Regression":
		model=modelLasso()
	else:
		return ("NoData")
	# Divisez les données en ensembles d'entraînement et de test
	#entrainement du modèle
	model.fit(X_train, y_train)
	# Prédictions
	y_pred = model.predict(X_test)
	styled_write(f"Un extrait des prédictions d'évolution du diabète sur un an")
	st.table(y_pred[0:7])
	# Évaluez le modèle
	MSE = round(mean_squared_error(y_test, y_pred),2)
	styled_write(f"Mean Squared Error : {MSE}")
	styled_write("Plus le MSE est proche de zéro, meilleure est la performance du modèle.")
	MAE=round(mean_absolute_error(y_test, y_pred),2)
	styled_write(f"Mean absolute Error : {MAE}")
	styled_write("Comme pour le MSE, une valeur de MAE plus proche de zéro indique une meilleure performance du modèle. \n Il est également plus robuste aux valeurs aberrantes que le MSE")
	Rscore=round(r2_score(y_test,y_pred),2)
	styled_write(f"Coefficient de détermination : {Rscore}")
	styled_write("Une valeur de R2 proche de 1 signifie que le modèle explique une grande partie de la variance des données")
	# Obtenez les coefficients du modèle
	coefficients = model.coef_
	# Obtenez l'intercept
	intercept = model.intercept_,
	# Affichez les coefficients et l'intercept
	styled_write(f"Coefficients : {coefficients}")
	styled_write(f"Intercept : {intercept}")
	styled_write(f"L'intercept a une valeur de {intercept}, ce qui est la valeur attendue de la variable cible lorsque toutes les caractéristiques sont à zéro.")
	styled_write(f"Rappel des colonnes : {df.columns}")
	# Interprétation des coefficients
	for i, coef in enumerate(coefficients):
		styled_write(f"La caractéristique {i+1} a un impact de {coef} sur la variable cible.")
	
	validationCroisee(df,model)

