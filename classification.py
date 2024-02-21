import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from style import styled_write
from regression import validationCroisee
import joblib


#
# @brief application d'un gridSearchCV à un modèle de régression logistique
# @return les hyperparamètres optimaux
#
def modelLogistic():
	model = LogisticRegression()
	# Définir la grille des hyperparamètres à rechercher
	param_grid = {
		#'penalty': ['l1', 'l2'],
		'C': [0.001, 0.01, 0.1, 1, 10, 100],
		#'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
		'max_iter': [50, 100, 200],
		'fit_intercept': [True, False],
		#'class_weight': [None, 'balanced']
	}
	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
	print(grid_search)
	return grid_search
#
# @brief application d'un gridSearchCV à un modèle de random forest
# @return les hyperparamètres optimaux
#
def modelRandomForest():
	model = RandomForestClassifier()
	param_grid = {
    'n_estimators': [50, 100, 200],
    #'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
	}
	# Initialiser GridSearchCV avec le modèle et les hyperparamètres
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	return grid_search

#
# @brief application d'un gridSearchCV à un modèle de KNeighbors
# @return les hyperparamètres optimaux
#
def modelKNeighbors():
	param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
	}
	# Initialiser le modèle KNeighborsClassifier
	model = KNeighborsClassifier()
	# Initialiser GridSearchCV avec le modèle et les hyperparamètres
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	return grid_search

#
# @param le df nettoye, le modele selectionne et les colonnes selectionnees
# @brief fonction qui effectue le split, l'entrainement et calcule les prédictions et leurs proba
#
def classificationChoice(df,selected_model,selected_columns):
		# Séparez les features et la target
	y = df['target']
	X = df[selected_columns]
	#découpage entre données d'entrainement et de test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	if selected_model =="LogisticRegression":
		checkbox_value=st.checkbox("Cocher cette case si vous souhaitez utiliser GridSearchCV")
		if checkbox_value:
			grid_search=modelLogistic()
			styled_write("La valeur des hyperparamètres est optimisée grâce à GridSearchCV")
		else:
			grid_search=LogisticRegression()
	elif selected_model == "RandomForestClassifier":
		checkbox_value2=st.checkbox("Cocher cette case si vous souhaitez utiliser GridSearchCV")
		if checkbox_value2:
			grid_search=modelRandomForest()
			styled_write("La valeur des hyperparamètres est optimisée grâce à GridSearchCV")
		else:
			grid_search=RandomForestClassifier()
	elif selected_model == "KNeighborsClassifier":
		checkbox_value3=st.checkbox("Cocher cette case si vous souhaitez utiliser GridSearchCV")
		if checkbox_value3:
			grid_search=modelKNeighbors()
			styled_write("La valeur des hyperparamètres est optimisée grâce à GridSearchCV")
		else:
			grid_search=KNeighborsClassifier()
	else:
		return ("NoData")
	# Entrainement du modèle
	grid_search.fit(X_train, y_train)
	# Prédictions 
	y_pred = grid_search.predict(X_test)
	styled_write(f"Un extrait des prédictions de classification du vin")
	st.table(y_pred[0:7])
	# Probabilités
	y_prob = grid_search.predict_proba(X_test)
	styled_write(f"Un extrait des probabilités de classification du vin")
	st.table(y_prob[0:7])
	h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 2 - Evaluation du modèle - Metrics </h2>'
	st.markdown(h2_title, unsafe_allow_html=True)
	# Évaluation du modèle - métrics
	accuracy = accuracy_score(y_test, y_pred)
	styled_write(f"Précision sur l'ensemble de test : {accuracy}")
	styled_write(f"<p style= 'font-weight:bold;'>Matrice de confusion</p>")
	cm = confusion_matrix(y_test, y_pred)
	st.table(cm)

	styled_write(f"<p style= 'font-weight:bold;'>Score ROC</p>")
	roc=roc_auc_score(y_test, grid_search.predict_proba(X_test),multi_class='ovr',average='macro')
	styled_write(f"Probabilité que le modèle attribue un score plus élevé à une instance positive par rapport à une instance négative choisie au hasard : {roc}")
	styled_write("Un modèle parfait aurait une ROC AUC égale à 1, tandis qu'un modèle aléatoire aurait une ROC AUC de 0.5.")

	cr = classification_report(y_test, y_pred)
	st.text("Rapport de Classification :\n\n{}".format(cr))
	
	# Validation croisée
	checkbox_value2=st.checkbox("Cocher cette case si vous souhaitez effectuer une validation croisée")
	if checkbox_value2:
		validationCroisee(df,selected_columns,5)
	
	# Sauvegarde du modèle
	st.title("Sauvegarder le modèle de ML")
	checkbox_value3=st.button("Cliquez ici si vous souhaitez sauvegarder votre modèle")
	if checkbox_value3:
		filename = "model.bin"
		joblib.dump(grid_search, filename)
		st.success(f"Modèle sauvegardé sous le nom : {filename}")
		with open(filename,'rb') as f:
   			st.download_button('Download Model', f,file_name='model.bin',mime='binary/octet-stream')  



	
	
