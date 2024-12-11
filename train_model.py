import json
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Chemin du fichier CSV
CSV_FILE = "datasets/loan_dataset_2.csv"

# Hyperparamètres du modèle
HYPERPARAMETERS = {
    "test_size": 0.3,  # Fraction des données utilisée pour le test
    "n_estimators": 100,  # Nombre d'arbres dans la forêt
    "max_depth": None,  # Profondeur maximale des arbres
    "min_samples_split": 4,  # Nombre minimum d'échantillons pour diviser un nœud
    "min_samples_leaf": 2  # Nombre minimum d'échantillons dans une feuille
}

# Liste des seeds pour entraînement multi-seed
SEEDS = [42, 7, 21, 99, 1234]

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- CHARGEMENT DES DONNÉES ---
logging.info("Chargement des données...")
data = pd.read_csv(CSV_FILE)
logging.info(f"Données chargées avec succès : {data.shape[0]} lignes, {data.shape[1]} colonnes.")

# Encodage des colonnes catégoriques
logging.info("Encodage des colonnes catégoriques...")
categorical_columns = ['person_education', 'person_home_ownership',
                       'loan_intent', 'previous_loan_defaults_on_file']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
logging.info("Encodage terminé.")

# Calculer le taux d'endettement
logging.info("Calcul du taux d'endettement...")
data['debt_to_income_ratio'] = (data['loan_amnt'] * (data['loan_int_rate'] / 100)) / data['person_income']

# Sélectionner les caractéristiques et la cible
features = [
    'person_age', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
    'loan_int_rate', 'debt_to_income_ratio'  # Inclure 'debt_to_income_ratio'
]
target = 'loan_status'

X = data[features]
y = data[target]

# --- ENTRAÎNEMENT MULTI-SEED ---
results = []  # Stocker les résultats pour chaque seed

for seed in SEEDS:
    logging.info(f"Entraînement avec random_state={seed}...")

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=HYPERPARAMETERS["test_size"], random_state=seed
    )
    logging.info(f"Ensemble d'entraînement : {X_train.shape[0]} lignes. Ensemble de test : {X_test.shape[0]} lignes.")

    # Initialiser le modèle avec les hyperparamètres
    model = RandomForestClassifier(
        random_state=seed,
        n_estimators=HYPERPARAMETERS["n_estimators"],
        max_depth=HYPERPARAMETERS["max_depth"],
        min_samples_split=HYPERPARAMETERS["min_samples_split"],
        min_samples_leaf=HYPERPARAMETERS["min_samples_leaf"]
    )

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculer les métriques
    accuracy = report["accuracy"]
    precision_1 = report["1"]["precision"]
    recall_1 = report["1"]["recall"]
    f1_score_1 = report["1"]["f1-score"]
    logging.info(f"Précision pour random_state={seed}: {accuracy:.2f}")

    results.append({
        "seed": seed,
        "accuracy": accuracy,
        "precision": precision_1,
        "recall": recall_1,
        "f1_score": f1_score_1,
        "conf_matrix": conf_matrix.tolist(),  # Convertir pour JSON
    })

    # Générer la matrice de confusion
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Refusé", "Accepté"],
                yticklabels=["Refusé", "Accepté"])
    plt.title(f"Matrice de confusion (random_state={seed})")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.savefig(f"static/confusion_matrix_seed_{seed}.png")
    plt.clf()  # Nettoyer le graphique pour la prochaine seed

# Sauvegarder les métriques pour les utiliser dans Flask
with open("static/results.json", "w") as f:
    json.dump(results, f)

# --- SAUVEGARDE DU MODÈLE FINAL ---
logging.info("Sauvegarde du dernier modèle entraîné...")
MODEL_FILE = "loan_model.joblib"
dump(model, MODEL_FILE)
logging.info(f"Modèle sauvegardé sous {MODEL_FILE}.")

# --- GÉNÉRATION DE GRAPHIQUES GLOBAUX ---
logging.info("Génération des graphiques globaux...")

# Courbes de précision pour chaque seed
plt.figure(figsize=(10, 6))

# Index des seeds (pour affichage simplifié sur l'axe X)
seeds = [result["seed"] for result in results]
accuracies = [result["accuracy"] for result in results]
average_accuracy = sum(accuracies) / len(accuracies)

# Création de l'histogramme
x_indices = np.arange(len(seeds))
plt.bar(x_indices, accuracies, color='skyblue', edgecolor='black', label="Précision par seed")
for i, acc in enumerate(accuracies):
    plt.text(x=i, y=acc + 0.001, s=f"{acc:.2f}", ha='center', fontsize=10)
plt.axhline(y=average_accuracy, color='red', linestyle='--', label=f"Précision moyenne ({average_accuracy:.2f})")
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)
plt.title("Précision du modèle pour différentes seeds")
plt.xlabel("Seed (index)")
plt.ylabel("Précision")
plt.xticks(x_indices, [f"Seed {i+1}" for i in range(len(seeds))])
plt.legend()
plt.savefig("static/accuracy_per_seed.png")
plt.clf()

logging.info("Graphiques générés avec succès.")
