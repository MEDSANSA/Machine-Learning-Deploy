from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Charger les fichiers nécessaires pour l'application de filtres
data = pd.read_csv('happiness_with_clusters.csv')
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

numeric_columns = [
    "Life Ladder", "Log GDP per capita", "Social support",
    "Healthy life expectancy at birth", "Freedom to make life choices",
    "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"
]

def filter_data(criteria):
    if criteria == 'happy_countries':
        result = data[(data['Life Ladder'] > 6.5) & (data['Social support'] > 0.8)].sort_values(by='Life Ladder', ascending=False).head(10)
    elif criteria == 'long_life_countries':
        result = data[(data['Healthy life expectancy at birth'] > 70) & (data['Perceptions of corruption'] < 0.3)].sort_values(by='Healthy life expectancy at birth', ascending=False).head(10)
    elif criteria == 'free_countries':
        result = data[(data['Freedom to make life choices'] > 0.8) & (data['Generosity'] > 0.2)].sort_values(by='Freedom to make life choices', ascending=False).head(10)
    elif criteria == 'secure_countries':
        data['Security Score'] = 1 - data['Perceptions of corruption']
        result = data.sort_values(by='Security Score', ascending=False).head(10)
    else:
        result = None
    
    # Supprimer les doublons en fonction du nom du pays
    if result is not None:
        result = result.drop_duplicates(subset=['Country name'])
        
    return result


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/rapport')
def dashbord():
    return render_template("dashbord.html")

# Route pour la page de prédiction avec KNN
@app.route("/predict", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        log_gdp = float(request.form['log_gdp'])
        social_support = float(request.form['social_support'])
        healthy_life_expectancy = float(request.form['healthy_life_expectancy'])
        freedom_to_make_life_choices = float(request.form['freedom_to_make_life_choices'])
        generosity = float(request.form['generosity'])
        perceptions_of_corruption = float(request.form['perceptions_of_corruption'])
        positive_affect = float(request.form['positive_affect'])
        negative_affect = float(request.form['negative_affect'])

        input_features = np.array([[log_gdp, social_support, healthy_life_expectancy,
                                    freedom_to_make_life_choices, generosity, perceptions_of_corruption,
                                    positive_affect, negative_affect]])

        prediction = knn_model.predict(input_features)[0]

    return render_template("lifelader.html", prediction=prediction)

# Route pour les filtres de pays
@app.route('/filtres', methods=["GET", "POST"])
def filtres():
    result = []  # Initialiser une liste vide pour les résultats

    if request.method == "POST":
        criteria = request.form.get('criteria')  # Récupérer le critère sélectionné
        
        if criteria:  # Vérifier que le critère est valide
            result_data = filter_data(criteria)
            if result_data is not None:
                # Convertir les résultats en dictionnaire et envoyer à la vue
                result = result_data[['Country name', 'Life Ladder', 'Social support', 'Healthy life expectancy at birth', 'Generosity', 'Perceptions of corruption']].to_dict(orient='records')

    return render_template("filtres.html", results=result)

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
