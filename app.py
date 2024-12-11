from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import os
import json

# Charger le modèle
MODEL_FILE = "loan_model.joblib"
model = load(MODEL_FILE)

# Encoders pour les colonnes catégoriques
categorical_columns = ['person_education', 'person_home_ownership', 'loan_intent']
label_encoders = {
    "person_education": LabelEncoder().fit(["High School", "Associate", "Bachelor", "Master", "Other"]),
    "person_home_ownership": LabelEncoder().fit(["RENT", "OWN", "MORTGAGE", "OTHER"]),
    "loan_intent": LabelEncoder().fit(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]),
}

# Initialiser l'application Flask
app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/stats")
def stats():
    with open("static/results.json", "r") as f:
        results = json.load(f)
    confusion_matrices = [f"confusion_matrix_seed_{result['seed']}.png" for result in results]
    return render_template("stats.html", results=results, confusion_matrices=confusion_matrices)

def is_debt_to_income_ratio_valid(loan_amount, income):
    """
    Vérifie si le taux d'endettement respecte les limites définies.
    """
    debt_to_income_ratio = loan_amount / income
    limits = [
        (0, 50000, 0.40),
        (50001, 250000, 0.35),
        (250001, 500000, 0.30),
        (500001, 800000, 0.25),
        (800001, 1000000, 0.22),
        (1000001, float('inf'), 0.20)
    ]
    for min_amount, max_amount, max_ratio in limits:
        if min_amount <= loan_amount <= max_amount:
            return debt_to_income_ratio <= max_ratio
    return False

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Définir les colonnes nécessaires (sans 'debt_to_income_ratio')
    features = [
        'person_age', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt',
        'loan_intent', 'loan_int_rate'
    ]

    # Vérifier les colonnes nécessaires
    missing_columns = [col for col in features if col not in data]
    if missing_columns:
        return jsonify({"error": f"Missing columns: {', '.join(missing_columns)}"}), 400

    # Calculer automatiquement 'debt_to_income_ratio'
    loan_amount = data['loan_amnt']
    interest_rate = data['loan_int_rate']
    income = data['person_income']
    if not is_debt_to_income_ratio_valid(loan_amount, income):
        return jsonify({"error": "❌ Votre prêt est refusé (taux d'endettement trop élevé)."}), 400

    # Créer un DataFrame à partir des données reçues
    input_data = pd.DataFrame([data])
    input_data['debt_to_income_ratio'] = (loan_amount * (interest_rate / 100)) / income

    # Encodage des colonnes catégoriques
    for col in categorical_columns:
        if col in input_data:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Ajouter 'debt_to_income_ratio' aux caractéristiques pour la prédiction
    features.append('debt_to_income_ratio')
    X = input_data[features]

    # Prédiction
    prediction = model.predict(X)[0]
    return jsonify({"loan_status": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
