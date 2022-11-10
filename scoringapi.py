# http://127.0.0.1:5000/
# API

import pandas as pd
from flask import Flask, request, jsonify
import pickle
from zipfile import ZipFile

# Création d'application Flask app: obtenir l'API server
app = Flask(__name__)

# Load le modèle entrainé
pickle_in = open('model_final', 'rb')
clf = pickle.load(pickle_in)

# Charger les données X standardisées
z = ZipFile("data.zip")
X = pd.read_csv(z.open('X.csv'), index_col='SK_ID_CURR', sep=",", encoding='utf-8')
X = X.drop({'Unnamed: 0'}, axis=1)

# Données non stardadisées
data = pd.read_csv(z.open('data.csv'), index_col='SK_ID_CURR', sep=",", encoding='utf-8', usecols=[
    "SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT"
])
z.close()


@app.route('/') #Créer Accueil
def home():
    return 'Home'


# API qui envoie le score, résultat du client choisi
@app.route('/predict/<int:client_id>', methods=['GET'])
def query_score(client_id):
    score = False
    resultat = False

    if client_id in X.index :
        score = clf.predict_proba(X.loc[[client_id]])[:, 1][0]
        if score < 0.12 :
            resultat = "Accepté"
        else :
            resultat = "Rejeté"

    return jsonify({
        "ID de clients": client_id,
        "Score de crédit": score,
        "Résultat des études": resultat})


# API qui envoie des infos clients selon id
@app.route('/client/<int:client_id>')
def client_info(client_id):
    if client_id in data.index:
        client = data.loc[client_id]
        return jsonify({
            "CODE_GENDER": client['CODE_GENDER'],
            "DAYS_BIRTH": client['DAYS_BIRTH'],
            "DAYS_EMPLOYED": client['DAYS_EMPLOYED'],
            "CNT_CHILDREN": client['CNT_CHILDREN'],
            "AMT_INCOME_TOTAL": client['AMT_INCOME_TOTAL'],
            "AMT_CREDIT": client['AMT_CREDIT']})

# A partir d'ici est utilisé pour le développement de l'API. C'est créer un serveur temporaire
# pour vérification de la fonctionnement.On n'a plus besoin.
# if __name__ == '__main__':
    # Marcher l'app dans debug mode sur le port 5000
    # app.run(debug=True, port=5000) # Lance le server

