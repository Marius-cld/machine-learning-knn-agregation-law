"""Entraînement et évaluation d'un modèle KNN pour prédire la réussite au
concours d'agrégation de droit à partir des caractéristiques d'un candidat.

Voir README.md pour le contexte du jeu de données et les résultats obtenus.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from pipeline import MODELS_DIR, pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT_DIR / "data" / "raw" / "data_agreg.xlsx"
FIGURE_DIR = ROOT_DIR / "figures"
TABLE_DIR = ROOT_DIR / "tables"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    ### Importation des données ###
    df = pd.read_excel(DATA_XLSX)
    df["resultat"] = np.where(df["resultat"] == "non-admis", 0, 1)

    ### Préparation des données ###
    # Contrainte de l'énoncé : au maximum 3 features pour le KNeighborsClassifier.
    target = "resultat"
    features = ["Delai_soutenance-concours", "Editeur_de_la_these"]

    Y = df[target]
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=Y)
    Y_train = train_set[target].reset_index(drop=True)
    Y_test = test_set[target].reset_index(drop=True)
    X_train = train_set[features].reset_index(drop=True)
    X_test = test_set[features].reset_index(drop=True)

    X_train_final = pipeline(X_train, train=True, scaler_name="standard")
    X_test_final = pipeline(X_test, train=False)

    ### Optimisation des hyperparamètres (n_neighbors, weights) ###
    k_values = list(range(1, 20))
    best_params = {}

    for weights in ("uniform", "distance"):
        mean_scores = [
            cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights=weights),
                X_train_final,
                Y_train,
                cv=10,
                scoring="accuracy",
            ).mean()
            for k in k_values
        ]
        best_k = k_values[int(np.argmax(mean_scores))]
        best_params[weights] = {"k": best_k, "cv_accuracy": max(mean_scores)}
        print(f"[{weights}] k optimal = {best_k} | accuracy moyenne (CV) = {max(mean_scores):.4f}")

    ### Entraînement et évaluation des modèles ###
    models = {}
    test_accuracies = {}
    cv_stability = {}

    for weights, params in best_params.items():
        model = KNeighborsClassifier(n_neighbors=params["k"], weights=weights)
        model.fit(X_train_final, Y_train)
        models[weights] = model

        predictions = model.predict(X_test_final)
        test_accuracies[weights] = accuracy_score(Y_test, predictions)
        print(f"[{weights}] accuracy (jeu de test) = {test_accuracies[weights]:.4f}")

        # Matrice de confusion
        conf_matrix = confusion_matrix(Y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Matrice de confusion - KNN ({weights})")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.savefig(FIGURE_DIR / f"matrice_confusion_{weights}.png")
        plt.close()

        # Validation croisée (stabilité du modèle)
        scores_cv = cross_val_score(model, X_train_final, Y_train, cv=10, scoring="accuracy")
        cv_stability[weights] = {"mean": scores_cv.mean(), "std": scores_cv.std(), "var": scores_cv.var()}
        print(
            f"[{weights}] CV accuracy : moyenne={scores_cv.mean():.4f}, "
            f"écart-type={scores_cv.std():.4f}, variance={scores_cv.var():.4f}"
        )

        # Courbe ROC
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]
        roc_auc = roc_auc_score(Y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)

        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f"KNN {weights} (AUC = {roc_auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "r--", label="Classifieur aléatoire")
        plt.xlabel("Taux de faux positifs")
        plt.ylabel("Taux de vrais positifs")
        plt.title(f"Courbe ROC - KNN ({weights})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"roc_{weights}.png")
        plt.close()

    best_weights = max(test_accuracies, key=test_accuracies.get)
    print(f"Modèle retenu : KNN ({best_weights}, k={best_params[best_weights]['k']})")

    ### Tableau récapitulatif de la sélection de modèle ###
    summary = pd.DataFrame(
        {
            weights: {
                "k_optimal": best_params[weights]["k"],
                "cv_accuracy_mean": cv_stability[weights]["mean"],
                "cv_accuracy_std": cv_stability[weights]["std"],
                "cv_accuracy_var": cv_stability[weights]["var"],
                "test_accuracy": test_accuracies[weights],
                "modele_retenu": weights == best_weights,
            }
            for weights in best_params
        }
    ).T
    summary.index.name = "weights"
    summary.to_csv(TABLE_DIR / "model_selection_summary.csv")

    ### Entraînement du modèle final sur l'ensemble des données ###
    X_full = df[features].reset_index(drop=True)
    Y_full = df[target].reset_index(drop=True)
    X_full_final = pipeline(X_full, train=True, scaler_name="standard")

    final_model = KNeighborsClassifier(
        n_neighbors=best_params[best_weights]["k"], weights=best_weights
    )
    final_model.fit(X_full_final, Y_full)

    with open(MODELS_DIR / "knn_model_final.pkl", "wb") as f:
        pickle.dump(final_model, f)

    print("Terminé.")


if __name__ == "__main__":
    main()
