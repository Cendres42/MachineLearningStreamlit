# Import des bibliothèques nécessaires
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from nettoyage import *
from regression import *
from classification import *



# Charger une image pour le fond


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('C:/Users/Gwen/Desktop/Projet3K/monitor2.jpg')

def styled_write(content):
    st.write(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 10px; border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


styled_write("<h1>Application pour modèles de machine learning</h1>")
styled_write("<p>Sélectionnez un fichier sur lequel appliquer le machine learning</p>")

# choix du jeu de données
uploadFile=st.selectbox("Menu déroulant",[" ","diabete.csv", "vin.csv","co2.csv"])


#
# @param le fichier uploade
# @brief chargement du fichier dans un dataframe
# return le dataframe 
#
def loading(uploadFile):
    if uploadFile is not None and uploadFile!=" ":
        styled_write(f"<p>Voici les 5 premières lignes de  {uploadFile}</p>")
        if uploadFile=="c02.csv":
            df= pd.read_csv("C:/Users/Gwen/Desktop/Projet3K/"+uploadFile,sep=";")
        else:
            df= pd.read_csv("C:/Users/Gwen/Desktop/Projet3K/"+uploadFile,sep=",")
        st.dataframe(df.head())
        return df
    else:
        styled_write("<p>Vous n'avez sélectionné aucun jeu de données: </p>")


#
# @brief affichage et nettoyage du jeu de donnees selectionne
# return le dataframe nettoye
#
def nettoyage():
    styled_write("<h2>Voici un aperçu de votre jeu de donnée</h2>")
    df=loading(uploadFile)
    if df is not None:
        print(df.iloc[0,1])
        affichageColonnes,taille,MoyMedMinMax,columnsWithNa=donnees(df)
        styled_write(f"<p>Voici les différentes colonnes de votre dataframe :  </p>")
        styled_write(affichageColonnes)
        styled_write(f"<p>Voici le nombre de lignes et de colonnes de votre dataframe :  {taille}</p>")
        styled_write(f"<pre>Voici quelques donnees statistiques sur votre dataframe : </pre>")
        st.table(MoyMedMinMax)
        if not columnsWithNa :
            styled_write("<p>Votre jeu de données ne contient pas de valeurs nulles donc pas de lignes à supprimer. </p>")
        else:
            # df['nomCol'].fillna(df['nomcol'].mean(), inplace=True)
            styled_write(f"<p>Ces colonnes contiennent des valeurs nulles :  {columnsWithNa}</p>")
        if df.columns[0]=="Unnamed: 0":
            df.drop('Unnamed: 0',axis=1, inplace=True)
            styled_write(f"Votre jeu de données contenait une colonne unnamed faisant doublon avec l'index, elle a été supprimée")
        else:
            styled_write(f"Aucune colonne de votre dataframe n'a été supprimée")
            j=0
        
        styled_write("<p>Voici la représentation graphique de votre target</p>")
        frequences = df['target'].value_counts(normalize=True)
        effectifs = df['target'].value_counts()
        tableau_distribution = pd.DataFrame({'Modalités': effectifs.index, 'Effectifs': effectifs.values, 'Fréquences': frequences.values})
        fig=plt.figure()

        plt.bar(tableau_distribution['Modalités'], tableau_distribution['Fréquences'], color='skyblue')
        plt.xlabel('Modalités')
        plt.ylabel('Fréquences')
        st.pyplot(fig)

        styled_write("<h2>Voici un aperçu des relations entre les principales variables de votre jeu de donnée</h2>")
        toplot=df[['age','sex','bmi','bp','target']]
        fig = sns.pairplot(toplot)
        st.pyplot(fig)

        mask = np.triu(df.select_dtypes("number").corr())
        fig2, ax = plt.subplots(figsize=(10, 10))
        cmap = sns.diverging_palette(15, 160, n=11, s=100)
        sns.heatmap(
            df.select_dtypes("number").corr(),
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
            )
        st.pyplot(fig2)
        
        # Test pour trouver d'éventuelles colonnes non numériques à recoder
        non_numeric_columns = df.select_dtypes(exclude='number').columns
        if non_numeric_columns>0:
            styled_write(f"Colonnes non numériques : {non_numeric_columns}")
        else:
            styled_write("Toutes vos colonnes sont numériques, aucun encodage de variables catégorielles n'est nécessaire")
            listeColstandard,listeNonColstandard=verifSandard(df)
            styled_write(f"Colonnes déjà standardisées à l'import : {listeColstandard}")
            styled_write(f"Colonnes ayant fait l'objet d'un standardisation : {listeNonColstandard}")
        styled_write(f"<p>Voici les 5 premières lignes de votre dataframe après nettoyage :  </p>")
        st.dataframe(df.head())
    return df


df_clean=nettoyage()



# On considère pour l'exercice qu'à une target catégorielle est appliqué 
# un modèle de classification et à une target continue est appliqué
# un modèle de regression
def gotoML(df_clean):
    if df_clean is not None :
        styled_write("Pour obtenir des prédictions, choisissez un modèle de ML")
        if pd.to_numeric(df_clean['target'], errors='coerce').notna().all():
            selected_model = st.selectbox("Choisissez un modèle de régression", ["Aucun modèle selectionné","Linear Regression", "Ridge Regression", "Lasso Regression"])
        else: 
            selected_model = st.selectbox("Choisissez un modèle de classification", ["Aucun modèle selectionné","", "", ""])
        return selected_model
    
def gotoModel(selected_model):
    if selected_model is not None:
        if pd.to_numeric(df_clean['target'], errors='coerce').notna().all():
            if selected_model != "Aucun modèle selectionné":
                styled_write(f"<h2>Voici le résultat de votre {selected_model}</h2>")
                regressionChoice(df_clean,selected_model)
        else: 
            if selected_model != "Aucun modèle selectionné":
                st.title("Classification avec Streamlit")
                st.write(f"Modèle sélectionné : {selected_model}")
                classificationChoice(df_clean,selected_model)

  
selected_model=gotoML(df_clean) 
gotoModel(selected_model)  

# df=df.dropna(subset=['hc'])
# df['nomcol']=pd.to_numeric(df['nomCol'].str.replace(',','.'),errors='coerce')