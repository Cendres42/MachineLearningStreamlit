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
from style import styled_write
import joblib

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
	alpha = st.slider("Alpha (Ridge)", min_value=0.0, max_value=1.0, value=best_alpha)
	model = Ridge(alpha)
	return model

def modelLinear():
	model = LinearRegression()
	return model

def modelLasso():
	alpha = st.slider("Alpha (Lasso)", min_value=0.0, max_value=1.0, value=0.1)
	model = Lasso(alpha=alpha)
	return model

def validationCroisee(df,selected_columns):
	h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 3 - Validation croisée </h2>'
	st.markdown(h2_title, unsafe_allow_html=True)
	styled_write("Voici les résultats de la validation croisée du KFold")
	FEATURES = selected_columns
	TARGET = 'target'
	equation = " + ".join(FEATURES)
	kf = KFold(n_splits=5,shuffle=True,random_state=2021)
	split = list(kf.split(df[FEATURES],df[TARGET]))
	test = []

	for i,(train_index, test_index) in enumerate(split):
		styled_write(f' ---------------- Fold {i+1} ------------')
		styled_write(f" -------------- Training on {len(train_index)} samples-------------- ")
		styled_write(f" -------------- Validation on {len(test_index)} samples-------------- ")
		data_train = df.loc[train_index] 
		data_test = df.loc[test_index] 
		# create a fitted model
		lm = smf.ols(formula=f'target ~ {equation}',data=data_train).fit()
		y_hat = lm.predict(data_test[FEATURES])
		test.append({
			'R2': r2_score(data_test[TARGET].values,y_hat),
			'model': lm
		})
		styled_write(f" Fold {i+1} :  MSE {round(mean_squared_error(data_test[TARGET].values,y_hat),4)}")
		styled_write(f" Fold {i+1} :  R2 {round(r2_score(data_test[TARGET].values,y_hat),4)}")
		fig=plt.figure()
		plt.scatter(data_test[TARGET].values,y_hat)
		plt.plot(np.arange(0,50))
		st.pyplot(fig)


def regressionChoice(df,selected_model,selected_columns):
	# Séparez les features et la target
	y = df['target']
	X = df[selected_columns]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Créez un menu déroulant pour choisir le modèle
	if selected_model =="Linear Regression":
		#X=[['age','sex','bmi','bp']]
		model=modelLinear()
	elif selected_model == "Ridge Regression":
		checkbox_value=st.checkbox("Cocher cette case si vous souhaitez utiliser GridSearchCV")
		if checkbox_value:
			best_alpha=findAlphaforRidge(X_train,y_train)
			model=modelRidge(best_alpha)
			styled_write("La valeur d'alpha est optimisée grâce à GridSearchCV mais vous pouvez la modifier")

		else:
			model=modelRidge(1.0)
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
	h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 2 - Evaluation du modèle - Metrics </h2>'
	st.markdown(h2_title, unsafe_allow_html=True)
	# Évaluez le modèle
	MSE = round(mean_squared_error(y_test, y_pred),2)
	styled_write(f"<p style= 'font-weight:bold;'>Mean Squared Error : {MSE}</p>")
	styled_write("Plus le MSE est proche de zéro, meilleure est la performance du modèle.")
	MAE=round(mean_absolute_error(y_test, y_pred),2)
	styled_write(f"<p style= 'font-weight:bold;'>Mean absolute Error : {MAE}</p>")
	styled_write("Comme pour le MSE, une valeur de MAE plus proche de zéro indique une meilleure performance du modèle. \n Il est également plus robuste aux valeurs aberrantes que le MSE")
	Rscore=round(r2_score(y_test,y_pred),2)
	styled_write(f"<p style= 'font-weight:bold;'>Coefficient de détermination : {Rscore}</p>")
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
	
	checkbox_value2=st.checkbox("Cocher cette case si vous souhaitez effectuer une validation croisée")
	if checkbox_value2:
		validationCroisee(df,selected_columns)

	st.title("Sauvegarder le modèle de ML")
	checkbox_value3=st.button("Cliquez ici si vous souhaitez sauvegarder votre modèle")
	# Proposer à l'utilisateur de sauvegarder le modèle
	if checkbox_value3:
		# Sauvegarder le modèle avec joblib
		filename = "model.bin"
		joblib.dump(model, filename)
		st.success(f"Modèle sauvegardé sous le nom : {filename}")
		with open(filename,'rb') as f:
   			st.download_button('Download Model', f,file_name='model.bin',mime='binary/octet-stream')  # Defaults to 'text/plain'

