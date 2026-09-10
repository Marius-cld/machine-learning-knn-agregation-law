# Prédiction de réussite au concours d'agrégation de droit (KNN)

Projet réalisé dans le cadre d'une formation universitaire en data science.
L'objectif est de construire un modèle de classification permettant de
prédire la réussite d'un candidat à l'agrégation de droit à partir de ses
caractéristiques, en utilisant l'algorithme **K-Nearest Neighbors (KNN)**.

## Contexte

Les agrégations de droit sont des concours de l'enseignement supérieur
français permettant le recrutement de professeurs des universités. Elles
regroupent quatre concours distincts (droit privé, droit public, histoire du
droit et des institutions, science politique) et se déroulent tous les deux
ans. Seuls les docteurs en droit peuvent y concourir.

Le jeu de données (non-exhaustif, fourni dans le cadre de l'exercice)
rassemble les résultats de candidats à différents concours de droit public
et de droit privé entre 1993 et 2015, avec pour chaque candidat :

| Variable                          | Description                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| `id`                                | Identifiant unique du candidat                                                        |
| `Age`                               | Âge du candidat au moment du concours                                                 |
| `Delai_soutenance-concours`        | Nombre d'années entre la soutenance de thèse et le concours                           |
| `Distance_univ_soutenance-paris`   | Distance entre Paris et l'université de soutenance de thèse                           |
| `Genre`                             | Genre du candidat                                                                     |
| `id_DT`                             | Identifiant unique du directeur de thèse                                              |
| `Univ_soutenance`                   | Université où le candidat a soutenu sa thèse                                          |
| `Editeur_de_la_these`              | Maison d'édition de la thèse (`non_publie` si non publiée)                            |
| `Concurrence`                       | Niveau de concurrence au concours (places / candidats)                                |
| `Publication`                       | Score de publications du candidat (pondéré par rapport aux autres candidats)          |
| `Specialite`                        | Concours concerné (droit public ou droit privé)                                       |
| `Period`                            | Période à laquelle le concours a eu lieu                                              |
| `resultat`                          | Résultat au concours (`admis` / `non-admis`) — **variable cible**                     |

L'énoncé complet de l'exercice est disponible dans [`docs/enonce.pdf`](docs/enonce.pdf).

> **Note sur les données** : le fichier `data/raw/data_agreg.xlsx` n'est pas
> inclus dans ce dépôt (données fournies dans un cadre pédagogique, non
> destinées à être diffusées publiquement). Pour exécuter le projet, placez
> ce fichier dans le dossier `data/raw/` à la racine.

## Objectif et contraintes

- **Cible** : prédire `resultat` (`admis` / `non-admis`).
- **Critère d'évaluation** : accuracy.
- **Contrainte imposée** : `KNeighborsClassifier`, avec un maximum de 3
  features, en jouant uniquement sur les hyperparamètres `n_neighbors` et
  `weights`.

## Démarche

1. **Exploration des données** (`src/exploration.py`) : profil statistique des
   variables numériques et catégorielles, détection des valeurs aberrantes
   (méthode IQR pour le quantitatif, catégories rares pour le qualitatif),
   visualisation des distributions et des corrélations.

2. **Sélection des features** : au vu de l'analyse exploratoire et de la
   contrainte de 3 features maximum, deux variables ont été retenues :
   `Delai_soutenance-concours` et `Editeur_de_la_these`.

3. **Préparation des données** (`src/pipeline.py`) : un pipeline unique,
   réutilisable pour l'entraînement et pour le test, gère dans l'ordre :
   - l'imputation des valeurs manquantes (médiane pour le quantitatif, mode
     pour le qualitatif) ;
   - l'écrêtage des valeurs aberrantes (méthode IQR) ;
   - l'encodage One-Hot des variables catégorielles ;
   - la normalisation (`StandardScaler` par défaut, `MinMaxScaler`
     disponible en option).

   Les transformateurs sont ajustés sur le jeu d'entraînement puis
   sauvegardés (`data/models/*.pickle`), afin d'être appliqués tels
   quels sur le jeu de test — sans fuite d'information entre les deux jeux.

4. **Modélisation et évaluation** (`src/main.py`) :
   - séparation train/test stratifiée (80/20) ;
   - recherche du `n_neighbors` optimal par validation croisée (10 folds),
     pour les deux valeurs de `weights` (`uniform` et `distance`) ;
   - évaluation sur le jeu de test : accuracy, matrice de confusion, courbe
     ROC / AUC ;
   - sélection du meilleur modèle, puis ré-entraînement sur l'intégralité
     des données pour obtenir le modèle final (`data/models/knn_model_final.pkl`).

## Résultats

Lors de l'entraînement sur le jeu de données de l'exercice, le modèle
`weights="distance"` (k=12) s'est montré légèrement plus performant que
`weights="uniform"` (k=7), avec une accuracy d'environ **70 %** sur le jeu de
test (contre ~50 % pour un classifieur aléatoire sur un problème binaire
déséquilibré). Les graphiques générés (matrices de confusion, courbes ROC,
distributions, corrélations) sont sauvegardés dans `figure/`, et le tableau
récapitulatif de la sélection de modèle dans `table/model_selection_summary.csv`,
à l'exécution des scripts.

Avec seulement 2 features imposées par la contrainte de l'exercice, la marge
de progression est naturellement limitée : les pistes d'amélioration
naturelles seraient d'autoriser davantage de variables, de tester d'autres
algorithmes (arbres, régression logistique) ou d'enrichir les features
(interactions, feature engineering sur `Univ_soutenance`, etc.).

## Structure du projet

```
.
├── docs/
│   └── enonce.pdf              # Énoncé de l'exercice
├── src/
│   ├── exploration.py          # Analyse exploratoire des données
│   ├── pipeline.py             # Pipeline de préparation des données (imputation, outliers, encodage, normalisation)
│   └── main.py                 # Sélection des hyperparamètres, entraînement, évaluation et modèle final
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── data_agreg.xlsx     # Donnée source (à ajouter manuellement, non versionné)
│   ├── processed/              # Snapshots intermédiaires du pipeline — généré, non versionné
│   └── models/                 # Transformateurs + modèle KNN final (.pickle/.pkl) — généré, non versionné
├── figure/                     # Graphiques générés (exploration, matrices de confusion, courbes ROC) — non versionné
└── table/                      # Résultats structurés (outliers détectés, sélection de modèle) — non versionné
```

## Installation et utilisation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt

# Placer le fichier de données dans data/raw/data_agreg.xlsx
# (à la racine du projet, à côté de src/)

python src/exploration.py   # analyse exploratoire
python src/main.py           # entraînement et évaluation du modèle
```

> Les scripts sont autonomes vis-à-vis du répertoire de travail (chemins
> résolus depuis l'emplacement du script) : ils peuvent être lancés aussi
> bien depuis la racine que depuis `src/`.

## Sources

- [KNN Classifier - Theory and Python Implementation](https://medium.com/@sahin.samia/demystifying-k-neighbors-classifier-knn-theory-and-python-implementation-from-scratch-f5e76d6f2d48)
- [Cross-validation with KNN, AUC computation](https://medium.com/@svanillasun/how-to-deal-with-cross-validation-based-on-knn-algorithm-compute-auc-based-on-naive-bayes-ff4b8284cff4)
- [ROC Curve - Step by Step Explanation](https://medium.com/@jonathan.barsotti/receiving-operating-characteristic-roc-curve-a-step-by-step-explanation-1954fb1b18c2)
- [KNN Classifier Tutorial (Kaggle)](https://www.kaggle.com/code/prashant111/knn-classifier-tutorial)
- [A Comprehensive Guide to Outliers in Machine Learning](https://medium.com/@samiraalipour/a-comprehensive-guide-to-outliers-in-machine-learning-detection-handling-and-impact-f7d965bba7a5)
