# Import des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nettoyage import *
from regression import *
from classification import *
from style import *


#
# @param le fichier choisi
# @brief chargement du fichier dans un dataframe
# return le dataframe 
#
def loading(uploadFile):
    if uploadFile is not None and uploadFile!=" ":
        h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 1 - Analyse et nettoyage du jeu de données</h2>'
        st.markdown(h2_title, unsafe_allow_html=True)
        styled_write2(f"<p  style= 'font-weight:bold;color:rgb(15, 76, 116);'>Voici un aperçu de votre jeu de donnée</p>")
        styled_write2(f"<p  style= 'font-weight:bold;'>Voici les 5 premières lignes de  {uploadFile}</p>")
        df= pd.read_csv(uploadFile,sep=",")
        st.dataframe(df.head())
        return df
    else:
        styled_write("<p>Vous n'avez sélectionné aucun jeu de données: </p>")


#
# @param le fichier choisi
# @brief affichage et nettoyage du jeu de donnees selectionne
# return le dataframe nettoye et les colonnes selectionnees
#
def nettoyage(uploadFile):
    selected_columns=[]
    df=loading(uploadFile)
    categ=False
    if df is not None :
        # affichage des donnees descriptives du df
        affichageColonnes,taille,MoyMedMinMax,columnsWithNa=donnees(df)
        styled_write2(f"<p  style= 'font-weight:bold;'>Voici les différentes colonnes de votre dataframe :  </p>")
        styled_write(affichageColonnes)
        styled_write(f"<p  style= 'font-weight:bold;'>Voici le nombre de lignes et de colonnes de votre dataframe :  {taille}</p>")
        styled_write2(f"<pre  style= 'font-weight:bold;'>Voici quelques donnees statistiques sur votre dataframe : </pre>")
        st.table(MoyMedMinMax)
        #test presence de valeurs nulles
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
        # test presence de doublons
        doublons = df[df.duplicated()]
        if doublons.empty:
            styled_write(f"Votre dataframe ne contient aucun doublon ni valeur aberrante")
        # Test pour trouver d'éventuelles colonnes non numériques à recoder
        h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 2 - Recodage et standardisation</h2>'
        st.markdown(h2_title, unsafe_allow_html=True)
        non_numeric_columns = df.select_dtypes(exclude='number').columns
        if len(non_numeric_columns)>0:
            categ=True
            styled_write(f"Colonnes non numériques : {non_numeric_columns}")
            df['target'].replace({'Vin equilibre' : 0,'Vin amer' : 1, 'Vin sucré' : 2},inplace=True)
            styled_write("La variable cible catégorielle a été recodée de la façon suivante: Vin equilibre : 0, Vin amer : 1,  Vin sucré : 2")
        else:
            categ=False
            styled_write("Toutes vos colonnes sont numériques, aucun encodage de variables catégorielles n'est nécessaire")
        # standardisation éventuelle des donnees avec la méthode StandardScaler
        listeColstandard,listeNonColstandard=verifSandard(df)
        styled_write(f"Colonnes déjà standardisées à l'import : {listeColstandard}")
        styled_write(f"Colonnes ayant fait l'objet d'un standardisation : {listeNonColstandard}")
        styled_write2(f"<p  style= 'font-weight:bold;'>Voici les 5 premières lignes de votre dataframe après nettoyage :  </p>")
        st.dataframe(df.head())
        styled_write2("<p  style= 'font-weight:bold;color:rgb(15, 76, 116);'>Le modèle de ML sera appliqué sur les colonnes que vous allez sélectionner</p>")
        selected_columns=st.multiselect("Choisissez vos features :", df.columns)
        display_columns = selected_columns.copy()
        display_columns.append('target')
        h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 3 - Représentation graphique </h2>'
        st.markdown(h2_title, unsafe_allow_html=True)
        
        # représentation graphique Target
        styled_write2("<p  style= 'font-weight:bold;color:rgb(15, 76, 116);'>Voici la représentation graphique de votre target</p>")
        frequences = df['target'].value_counts(normalize=True)
        effectifs = df['target'].value_counts()
        tableau_distribution = pd.DataFrame({'Modalités': effectifs.index, 'Effectifs': effectifs.values, 'Fréquences': frequences.values})
        fig=plt.figure()
        plt.bar(tableau_distribution['Modalités'], tableau_distribution['Fréquences'], color='skyblue')
        plt.xlabel('Modalités')
        plt.ylabel('Fréquences')
        st.pyplot(fig)
        
        # représentation graphique des corrélations (nuage de points)
        styled_write2("<p  style= 'font-weight:bold;color:rgb(15, 76, 116);'>Voici un aperçu des relations entre les principales variables de votre jeu de donnée</p>")
        if selected_columns!=[]:
            toplot=df[display_columns]
            fig = sns.pairplot(toplot)
            st.pyplot(fig)

        # représentation graphique des corrélations (palettes de couleurs)
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

    return df, selected_columns,categ

#
# @brief sélection du jeu de donnees et enregistrement du df nettoye et des colonne selectionne dans la session
#  
def page1():
    h1_title = '<h1 style=" color:rgb(15, 76, 116); background-color: rgba(255, 255, 255, 0.7);font-size: 25px;">Data Management: nettoyage et features engineering</h1>'
    st.markdown(h1_title, unsafe_allow_html=True)
    # choix du jeu de données
    uploadFile=st.selectbox("Sélectionnez un fichier sur lequel appliquer le machine learning",[" ","diabete.csv", "vin.csv"])
    df_clean,selected_columns,categ=nettoyage(uploadFile)
    st.session_state['df']= df_clean
    st.session_state['selected_columns']=selected_columns
    st.session_state['categ']=categ

#image de fond
set_background('monitor2.jpg')

#
# @param le df nettoye
# @brief proposition de modèles de ML en fonction du type de Target
# On considère pour l'exercice qu'à une target catégorielle est appliqué 
# un modèle de classification et à une target continue est appliqué
# un modèle de regression
# 
# @return le modèle selectionné
# 
def gotoML(df_clean,categ):
    if df_clean is not None :
        styled_write("Pour obtenir des prédictions, choisissez un modèle de ML")
        if categ==False:
            selected_model = st.selectbox("Choisissez un modèle de régression", ["Aucun modèle selectionné","Linear Regression", "Ridge Regression", "Lasso Regression"])
        else: 
            selected_model = st.selectbox("Choisissez un modèle de classification", ["Aucun modèle selectionné","LogisticRegression","RandomForestClassifier","KNeighborsClassifier"])
        return selected_model,categ

#
# @param le df nettoye, le modele selectionne et les colonnes selectionnees
# @brief proposition de modèles de ML en fonction du type de Target
#     
def gotoModel(selected_model,df_clean,selected_columns,categ):
    if selected_model is not None:
        if categ==False:
            if selected_model != "Aucun modèle selectionné":
                h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 1 - Split et train</h2>'
                st.markdown(h2_title, unsafe_allow_html=True)
                styled_write("Le jeu de données à été partagé entre le train_set (80%) et le test_set(20%) puis le modèle a été entrainé sur le train_set")
                regressionChoice(df_clean,selected_model,selected_columns)
        else: 
            if selected_model != "Aucun modèle selectionné":
                h2_title = '<h2 style=" color:darkred; background-color: rgba(255, 255, 255, 0.7);font-size: 20px;">Partie 1 - Split et train</h2>'
                st.markdown(h2_title, unsafe_allow_html=True)
                styled_write("Le jeu de données à été partagé entre le train_set (80%) et le test_set(20%) puis le modèle a été entrainé sur le train_set")
                classificationChoice(df_clean,selected_model,selected_columns)
#
# @brief sélection du jeu de donnees et enregistrement du df nettoye et des colonne selectionne dans la session
# @param  le modele selectionne et les colonnes selectionnees
#
def page2(df_clean,selected_columns,categ):
    st.title("Machine learning")
    selected_model,categ=gotoML(df_clean,categ) 
    gotoModel(selected_model,df_clean,selected_columns,categ)  

#
# @brief page d'accueil
#
def home():
    h1_title = '<h1 style=" background-color: rgba(255, 255, 255, 0.7);color:rgb(15, 76, 116); font-size: 25px;">Bienvenue sur cette application Streamlit de machine learning</h1>'
    st.markdown(h1_title, unsafe_allow_html=True)
    st.write("Pour commencer cliquer sur Data Management")
    
# Barre de navigation
pages = {
    "Accueil": home,
    "Data Management": page1,
    "Machine learning": page2,
}


# Utilisation de la variable de session pour suivre l'état de la page
current_page = st.session_state.get("current_page", "home")

#application d'un background aux tableaux
styl = '''
    <style>
        table{
            background-color: rgba(255, 255, 255, 0.7);
       }
        
    </style>
    '''
st.markdown(styl, unsafe_allow_html=True)


# Barre de navigation avec des boutons
home_button = st.sidebar.button("Accueil", key="home")
page1_button = st.sidebar.button("Data Management", key="page1")
page2_button = st.sidebar.button("Machine learning", key="page2")

if home_button:
    current_page = "home"
elif page1_button:
    current_page = "page1"
elif page2_button:
    current_page = "page2"

# Affichage de la page correspondante
if current_page == "home":
    home()
elif current_page == "page1":
   page1()
elif current_page == "page2":
    page2(st.session_state['df'],st.session_state['selected_columns'],st.session_state['categ'])

# Mise à jour de la variable de session
st.session_state["current_page"] = current_page
