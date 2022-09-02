# http://127.0.0.1:5000/predict/id
# import main Flask class and request object

import pandas as pd
from flask import Flask, request, jsonify
import pickle
from zipfile import ZipFile

# create the Flask app: obtain the API server
app = Flask(__name__)

# loading the trained model
pickle_in = open('model_final', 'rb')
clf = pickle.load(pickle_in)

# Données X standardisées
z = ZipFile("data.zip")
X = pd.read_csv(z.open('X.csv'), index_col='SK_ID_CURR', sep=",", encoding='utf-8')
# X = pd.read_csv("X.csv", index_col='SK_ID_CURR', sep=",")
X = X.drop({'Unnamed: 0'}, axis=1)

# Données non stardadisées
data = pd.read_csv(z.open('data.csv'), index_col='SK_ID_CURR', sep=",", encoding='utf-8', usecols=[
    "SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT"
])
# data = pd.read_csv("data.csv", index_col='SK_ID_CURR', sep=",", usecols=[
#    "SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH",
#    "DAYS_EMPLOYED",
#    "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT"
# ])

@app.route('/') #Créer Accueil
def home():
    return 'Home'


# API qui envoie le score, résultat et shap values des clients
@app.route('/predict/<int:client_id>', methods=['GET', 'POST'])
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


# Afficher les infos clients selon id
@app.route('/client/<int:client_id>') # Url / Route / API
def client_info(client_id):  # Paramètre
    if client_id in data.index:
        #client = data.query(f'data.index == {client_id}') # Pour que la fonction comprenne que 'client_id' est une variable
        client = data.loc[client_id]
        # res = client.to_json()
        return jsonify({
            "CODE_GENDER": client['CODE_GENDER'],
            "DAYS_BIRTH": client['DAYS_BIRTH'],
            "DAYS_EMPLOYED": client['DAYS_EMPLOYED'],
            "CNT_CHILDREN": client['CNT_CHILDREN'],
            "AMT_INCOME_TOTAL": client['AMT_INCOME_TOTAL'],
            "AMT_CREDIT": client['AMT_CREDIT']})
    # res # Return doit être toujours en string

# if key doesn't exist, returns None
    #client_id = request.args.get('client_id')
    # filtre sur data lorsque id=client_id

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000) # Lance le server

