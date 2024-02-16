# votre code ici
import pandas as pd 

#df=pd.read_csv("C:/Users/Gwen/Desktop/Projet3/diabete.csv",sep=",")

def donnees(df):
	affichageColonnes=df.columns
	taille=df.shape
	MoyMedMinMax=df.describe(include='all')
	columnsWithNa = df.columns[df.isna().any()].tolist()
	return affichageColonnes,taille,MoyMedMinMax,columnsWithNa

def verifSandard(df):
	#exclusion de la target de la standardisaton:
	toexclude=df.drop('target',axis=1)
	#vérification que le standardisation a été appliquée
	colstandard = toexclude.apply(lambda col: all(-1 <= x <= 1 for x in col))
	# Vérifiez si au moins une valeur est en dehors de la plage -1 à +1 pour chaque colonne
	colnonstandard = toexclude.apply(lambda col: any(x < -1 or x > 1 for x in col))
	# Liste des colonnes pour lesquelles toutes les valeurs sont comprises entre -1 et +1
	listeColstandard = list(colstandard[colstandard].index)
	# Liste des colonnes pour lesquelles au moins une valeur est en dehors de la plage -1 à +1
	listeNonColstandard = list(colnonstandard[colnonstandard].index)
	if listeNonColstandard:
		tostandardValue(df,listeNonColstandard)
	return listeColstandard,listeNonColstandard

def tostandardValue(df,listeNonColstandard):
	from sklearn.preprocessing import StandardScaler
	# Identifiez les colonnes dont les valeurs ne sont pas comprises entre -1 et +1
	# Créez un objet StandardScaler
	scaler = StandardScaler()
	# Standardisez seulement les colonnes nécessaires
	df[listeNonColstandard] = scaler.fit_transform(df[listeNonColstandard])
	# Affichez le DataFrame après standardisation





