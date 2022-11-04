**Implémentation d'un modèle de scoring chez Prêt à dépeser**

Une société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées telles que les données comportementales, les données provenant d'autres institutions financières, etc.

De plus, aujourd'hui les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. De ce fait, l'entreprise veut développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

Donc ici, nous avons deux missions :

 * Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique ;
 * Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

Pour la partie de la construction du modèle, les différents traitelents ont été effectués sur deux jupyter Notebook : prétraitement et modélisation.

Ensuite pour la partie dashboard, une API et un dashboard ont été déployés sur deux cloud différents

* Pour API :
https://p7-credit-scoring-ty.herokuapp.com

    * Accès aux données : 
    * https://p7-credit-scoring-ty.herokuapp.com/predict/<int:client_id ex:100002> pour le score et le résultat

    * https://p7-credit-scoring-ty.herokuapp.com/client/<int:client_id ex:100002> pour les informations détaillées d'un client

* Pour le dashbord : 
https://tomomiyosh-p7-dashbord-2hp8ej.streamlitapp.com/
