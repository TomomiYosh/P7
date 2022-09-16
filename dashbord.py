# http://127.0.0.1:5000/
# Dashboard

import pandas as pd
import streamlit as st  # 🎈 data web app development
import requests
import json
from zipfile import ZipFile

url = 'https://p7-credit-scoring-ty.herokuapp.com'

# Création de page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🎈",
    layout="wide",
)

# Titre de dashboard
st.title("Informations client")


# Obtenir id, score et résultat
def load_prediction(client_id):
    client_data = requests.get(f'{url}/predict/{client_id}') # f : quand il y a une variable python dans la chaine
    return json.loads(client_data.content)


# Selection d'un ID crédit
st.sidebar.header("**Credit scoring dashboard**")

# Selectbox
client_id = st.sidebar.text_input("Entrez un ID crédit")

# Explication des features
st.sidebar.subheader("**Description d'une feature**")
description = pd.read_csv("HomeCredit_columns_description.csv", sep=",")
description = description.query('Row != "SK_ID_CURR"')
choix = st.sidebar.selectbox('Choisissez une feature que vous voulez avoir une explication', description['Row'])
explication = description.loc[choix]
st.sidebar.write("Description : ", explication['Description'])

# Information du score et résultat des études
if client_id:
    client_id = int(client_id)
    st.subheader("**Information de score du client** {:.0f}".format(client_id))
    client_prediction = load_prediction(client_id)
    st.write("Résultat des études dossier : ", client_prediction["Résultat des études"])

    # Gauge
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = client_prediction["Score de crédit"],
        mode = "gauge+number+delta",
        title = {'text': "Score de crédit", 'font': {'size': 32}},
        delta = {'reference': 0.12},
        gauge = {'axis': {'range': [None, 1], 'tickwidth': 0.6, 'tickcolor': "darkblue"},
                 'bar': {'color': "darkblue"},
                 'bgcolor': "white",
                 'borderwidth': 2,
                 'bordercolor': "gray",
                 'steps' : [
                     {'range': [0, 0.12], 'color': "cyan"},
                     {'range': [0.12, 1], 'color': "red"}],
                 'threshold' : {'line': {'color': "blue", 'width': 5}, 'thickness': 1, 'value': 0.12}}))
    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    st.plotly_chart(fig)

    # Load le modèle entrainé
    import pickle
    pickle_in = open('model_final', 'rb')
    clf = pickle.load(pickle_in)

    # Feature importance locale
    # Données X standardisées
    z = ZipFile("data.zip")
    main_data = pd.read_csv(z.open('X.csv'), index_col='SK_ID_CURR', sep=",", encoding='utf-8')
    main_data = main_data.drop({'Unnamed: 0'}, axis=1)

    import shap
    if st.checkbox("Afficher feature importance de ce client ?"):
        shap.initjs()
        X = main_data.loc[[client_id]]
        st.subheader("**Feature importance locale**")
        fig, ax = plt.subplots(figsize=(10, 20))
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type="bar", max_display=20, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
        X = False

    # Informations détaillées d'un client
    st.subheader("**Informations détailées du client**")

    if st.checkbox("Plus d'informations clients ?"):

    ### Des information supplémentaires ###
    # Obtenir des infos de bases
        def load_infobase(client_id) :
            client_info = requests.get(f'{url}/client/{client_id}')
            return json.loads(client_info.content)

        client_infos = load_infobase(client_id)
        gender = client_infos["CODE_GENDER"]
        if gender == 0.0:
            st.write("Gender : Male")
        if gender == 1.0:
            st.write("Gender : Female")
        st.write("Age : {:.0f} ans".format(int(client_infos["DAYS_BIRTH"]/-365.25)))
        st.write("Revenu annuel : {:.0f}".format(client_infos["AMT_INCOME_TOTAL"])) # AMT_INCOME_TOTAL
        st.write("Montant de crédit : {:.0f}".format(client_infos["AMT_CREDIT"])) # AMT_CREDIT
        st.write("Nombre d'années employé : {:.0f} ans".format(int(client_infos["DAYS_EMPLOYED"]/-365.25)))
        st.write("Numbre d'enfants : {:.0f}".format(client_infos["CNT_CHILDREN"]))

    # Infos sur les autres clients
    # La feature importance globale
    st.header("**Informations des autres clients**")
    shap.initjs()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("**Feature importance globale**")
        fig, ax = plt.subplots(figsize=(12, 12))
        explainer = shap.TreeExplainer(clf)
        shap_values_m = explainer.shap_values(main_data)
        shap.summary_plot(shap_values_m[0], features=main_data, feature_names=main_data.columns, plot_type='bar')
        st.pyplot(fig)

    with col2:
        # Distribution de features sélectionnées
        st.subheader("**Distribution de la feature sélectionnée**")

        # Données non standardisées
        data = pd.read_csv(z.open("data.csv"), index_col='SK_ID_CURR', sep=",")
        data = data.drop({'Unnamed: 0'}, axis=1)
        data2 = data.drop({'TARGET'}, axis=1)
        data_X = data.loc[client_id]
        z.close()

        # Selectbox et graphique box plot
        import seaborn as sns
        option = st.selectbox('Choisissez une feature que vous souhaitez étudier', data2.columns)
        st.write("Le point rouge concerne le client du crédit concerné.")
        if option:
            g = sns.catplot(x="TARGET", y=option, kind="box", data=data)

            # Ajouter un point rouge du client concerné
            g.ax.scatter(x=data_X["TARGET"], y=data_X[option],c="red",zorder=10)
            sns.set(rc={'figure.figsize':(6,4)})
            st.pyplot(g)

    col4, col5 = st.columns(2)

    with col4:
        # Un graphique d’analyse bi-variée entre les deux features sélectionnées
        st.subheader("**Analyse bi-variée entre les deux features sélectionnées**")
        st.write("Le point rouge concerne le client du crédit concerné. Si le couleur est plus foncé, le score est plus élevé.")
        option2 = st.selectbox('Choisissez la première feature (abscisse) que vous souhaitez étudier', data2.columns)
        option3 = st.selectbox('Choisissez la deuxième feature (ordonnée) que vous souhaitez étudier', data2.columns)
        score = clf.predict_proba(main_data)[:, 1]
        if option2 and option3 :
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(x=data[option2], y=data[option3], c=score, cmap='YlGnBu')
            plt.scatter(x=data_X[option2], y=data_X[option3], c="red", zorder=10)
            plt.xlabel(option2)
            plt.ylabel(option3)
            st.pyplot(fig)



