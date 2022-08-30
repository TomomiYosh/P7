# http://127.0.0.1:5000/
# Streamlit
# Le tableau de bord intégrera au minimum :
# N° client, crédit accepté ou non, score détaillé sous forme de jauge : permet de juger s’il est loin du seuil ou non.
# Sa feature importance locale sous forme de graphique, qui permet au chargé d’étude de comprendre quelles sont les données du client qui ont le plus influencé le calcul de son score
# Le tableau de bord présentera également d’autres graphiques sur les autres clients :
# Deux graphiques de features sélectionnées dans une liste déroulante, présentant la distribution de la feature selon les classes, ainsi que le positionnement de la valeur du client
# Un graphique d’analyse bi-variée entre les deux features sélectionnées, avec un dégradé de couleur selon le score des clients, et le positionnement du client
# La feature importance globale
# D’autres graphiques complémentaires
# https://blog.streamlit.io/how-to-build-a-real-time-live-dashboard-with-streamlit/

# dans pd.read_csv le paramètre "nrows" limite le nombre de lignes chargée en mémoire,
# a supprimer avant de rendre le projet.

import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import requests
import json

#app = st(__name__)

# Créatuib de page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🎈",
    layout="wide",
)

# dashboard title
st.title("Informations client")


# Obtenir id, score et résultat
def load_prediction(client_id):
    client_data = requests.get(f'http://127.0.0.1:5000/predict/{client_id}')
    return json.loads(client_data.content)

# text input
#Customer ID selection
st.sidebar.header("**Credit scoring dashboard**")
#Loading selectbox
client_id = st.sidebar.text_input("Mettez un ID crédit")
if client_id:
    client_id = int(client_id)
    # st.write("Client ID : ", client_id)
    st.subheader("**Information de score du client** {:.0f}".format(client_id))
    # {:.0f} ans".format(int(client_infos["DAYS_BIRTH"]/-365.25))
    client_prediction = load_prediction(client_id)
    st.write("Résultat des études dossier : ", client_prediction["Résultat des études"])


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

    import pickle
    # loading the trained model
    pickle_in = open('model_final', 'rb')
    clf = pickle.load(pickle_in)

    # Feature importance / description
    # Données X standardisées
    main_data = pd.read_csv("X.csv", index_col='SK_ID_CURR', sep=",", nrows=100)
    main_data = main_data.drop({'Unnamed: 0'}, axis=1)


    import shap
    if st.checkbox("Afficher feature importance de ce client ?"):
        shap.initjs()
        X = main_data.loc[[client_id]]
        st.subheader("**Feature importance locale**")
        fig, ax = plt.subplots(figsize=(10, 20))
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        # shap.summary_plot(shap_values, features=X, feature_names=X.columns)
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, color_bar=False, plot_size=(5, 5))
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X[0], feature_names=X.columns)
        st.pyplot(fig)
        X = False

    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.subheader("**Informations détailées du client**")

    if st.checkbox("Plus d'informations clients ?"):

    ### Des information supplémentaires ###
    # Obtenir des infos de bases
        def load_infobase(client_id) :
            client_info = requests.get(f'http://127.0.0.1:5000/client/{client_id}')
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

    # Sur les autres clients
    # La feature importance globale
    st.header("**Informations des autres clients**")
    shap.initjs()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("**Feature importance globale**")
        fig, ax = plt.subplots(figsize=(12, 12))
        explainer = shap.TreeExplainer(clf)
        shap_values_m = explainer.shap_values(main_data)
        shap.summary_plot(shap_values_m, features=main_data, feature_names=main_data.columns, plot_type='bar')
        # shap.summary_plot(shap_values_m, main_data, plot_type="bar", max_display=20, color_bar=False, plot_size=(5, 5))
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X[0], feature_names=X.columns)
        st.pyplot(fig)

    with col2:
        # Distribution de features sélectionnées
        st.subheader("**Distribution de la feature sélectionnée**")

        # Données non stardadisées
        data = pd.read_csv("data.csv", index_col='SK_ID_CURR', sep=",", nrows=100)
        #data_1 = data.query('TARGET == 1.0')
        #data_0 = data.query('TARGET == 0.0')
        data = data.drop({'Unnamed: 0'}, axis=1)
        data2 = data.drop({'TARGET'}, axis=1)
        data_X = data.loc[client_id]

        # Add selectbox in streamlit
        import seaborn as sns
        option = st.selectbox('Choisissez une feature que vous souhaitez étudier', data2.columns)
        st.write("Le point rouge concerne le client du crédit concerné.")
        if option:
            #plt.title(f"Distribution de la classe 0 (crédit accepté) de la feature {option}")
            g = sns.catplot(x="TARGET", y=option, kind="box", data=data)

            # adding data points
            g.ax.scatter(x=data_X["TARGET"], y=data_X[option],c="red",zorder=10)
            sns.set(rc={'figure.figsize':(6,4)})
            st.pyplot(g)

    col4, col5 = st.columns(2)

    with col4:
        # Un graphique d’analyse bi-variée entre les deux features sélectionnées, avec un dégradé de couleur selon le score des clients, et le positionnement du client
        st.subheader("**Analyse bi-variée entre les deux features sélectionnées**")
        st.write("Le point rouge concerne le client du crédit concerné. Si le couleur est plus foncé, le score est plus élevé.")
        option2 = st.selectbox('Choisissez la première feature (abscisse) que vous souhaitez étudier', data2.columns)
        option3 = st.selectbox('Choisissez la deuxième feature (ordonnée) que vous souhaitez étudier', data2.columns)
        score = clf.predict_proba(main_data)[:, 1]
        if option2 and option3 :
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(x=data[option2], y=data[option3], c=score, cmap='YlGnBu')
            plt.scatter(x=data_X[option2], y=data_X[option3], c="red", zorder=10)
            # sns.set(rc={'figure.figsize': (8, 6)})
            st.pyplot(fig)



