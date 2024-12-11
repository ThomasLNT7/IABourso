from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import json

# Charger le modèle
MODEL_FILE = "loan_model.joblib"
model = load(MODEL_FILE)

# Encoders pour les colonnes catégoriques (adaptés à votre dataset)
categorical_columns = ['person_education', 'person_home_ownership', 'loan_intent']

label_encoders = {
    "person_education": LabelEncoder().fit(["High School", "Associate", "Bachelor", "Master", "Other"]),
    "person_home_ownership": LabelEncoder().fit(["RENT", "OWN", "MORTGAGE", "OTHER"]),
    "loan_intent": LabelEncoder().fit(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]),
}

# Initialiser l'application Flask
app = Flask(__name__)

# Ajouter `zip` comme fonction utilisable dans les templates
app.jinja_env.globals.update(zip=zip)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/stats")
def stats():
    # Charger les métriques depuis le fichier JSON
    with open("static/results.json", "r") as f:
        results = json.load(f)

    # Générer la liste des matrices de confusion disponibles
    confusion_matrices = [
        f"confusion_matrix_seed_{result['seed']}.png" for result in results
    ]

    # Envoyer les résultats et les matrices de confusion à la page stats.html
    return render_template("stats.html", results=results, confusion_matrices=confusion_matrices)

@app.route("/predict", methods=["POST"])
def predict():
    # Récupérer les données JSON
    data = request.json

    # Vérifier si toutes les colonnes nécessaires sont présentes
    required_columns = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]
    missing_columns = [col for col in required_columns if col not in data]
    if missing_columns:
        return jsonify({"error": f"Missing columns: {', '.join(missing_columns)}"}), 400

    # Transformer les données en DataFrame
    input_data = pd.DataFrame([data])

    # Encodage des colonnes catégoriques
    for col in categorical_columns:
        if col in input_data:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Calculer le taux d'endettement
    input_data['debt_to_income_ratio'] = (
        input_data['loan_amnt'] * (input_data['loan_int_rate'] / 100)
    ) / input_data['person_income']

    # Sélectionner les colonnes dans le bon ordre pour le modèle
    features = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file', 'debt_to_income_ratio'
    ]
    X = input_data[features]

    # Prédire avec le modèle
    prediction = model.predict(X)[0]

    # Retourner le résultat
    result = {"loan_status": int(prediction)}  # 1 = accepté, 0 = refusé
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
