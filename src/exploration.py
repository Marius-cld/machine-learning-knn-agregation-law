"""Analyse exploratoire du jeu de données sur les concours d'agrégation de
droit : profil statistique des variables, détection des valeurs aberrantes
et visualisations (distributions, corrélations).

Sources :
- https://medium.com/@samiraalipour/a-comprehensive-guide-to-outliers-in-machine-learning-detection-handling-and-impact-f7d965bba7a5
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT_DIR / "data" / "raw" / "data_agreg.xlsx"
FIGURE_DIR = ROOT_DIR / "figure"
TABLE_DIR = ROOT_DIR / "table"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    ### Import de la base de données ###
    df = pd.read_excel(DATA_XLSX)

    ### Aperçu général ###
    print(df.columns)
    print(df.shape)
    print(df.info())

    df_num = df.select_dtypes(include=["number"]).drop(columns="Unnamed: 0")
    df_cat = df.select_dtypes(include=["object"]).drop(columns=["id", "id_DT"])

    print(df_cat.describe())
    print(df_num.describe())

    ### Détection des valeurs aberrantes ###
    df_outlier = pd.DataFrame(index=df.index)

    # Variables catégorielles : une catégorie est considérée comme rare si elle
    # représente moins de 10% des observations.
    threshold = 0.1 * len(df)
    for col in df_cat.columns:
        category_counts = df[col].value_counts()
        rare_categories = category_counts[category_counts < threshold].index
        df_outlier[f"{col}_Anomaly"] = df[col].isin(rare_categories)

    # Variables quantitatives : méthode IQR, adaptée aux distributions asymétriques.
    for col in df_num.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        df_outlier[f"{col}_Outlier"] = df[col].where(is_outlier, np.nan)

    print(df_outlier.info())
    print(df_outlier.describe())
    df_outlier.to_excel(TABLE_DIR / "outlier.xlsx", index=False)

    ### Représentation graphique ###

    # Tableaux de statistiques descriptives (numériques et catégorielles)
    for name, sub_df in (("num", df_num), ("cat", df_cat)):
        desc_stats = sub_df.describe().round(2)
        na_counts = sub_df.isna().sum().to_frame().T
        na_counts.index = ["NA"]
        desc_stats = pd.concat([desc_stats, na_counts])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        table = ax.table(
            cellText=desc_stats.values,
            colLabels=desc_stats.columns,
            rowLabels=desc_stats.index,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        plt.savefig(
            FIGURE_DIR / f"statistiques_descriptives_{name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    # Distributions des variables quantitatives (histogramme + boxplot)
    quantitative_vars = {
        "Age": "age",
        "Delai_soutenance-concours": "delai_soutenance",
        "Distance_univ_soutenance-paris": "distance_univ_soutenance",
        "Concurrence": "concurrence",
        "Publication": "publication",
    }
    for column, filename in quantitative_vars.items():
        plt.figure(figsize=(15, 6), dpi=300)
        plt.subplot(1, 2, 1)
        sns.histplot(data=df_num, x=column, kde=True)
        plt.title(f"Histogramme - {column}")
        plt.subplot(1, 2, 2)
        sns.boxplot(df_num[column])
        plt.title(f"Boxplot - {column}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{filename}.png")
        plt.close()

    # Distributions des variables qualitatives
    qualitative_vars = {
        "resultat": "resultat",
        "Period": "period",
        "Specialite": "specialite",
        "Editeur_de_la_these": "editeur",
        "Univ_soutenance": "soutenance",
        "Genre": "genre",
    }
    for column, filename in qualitative_vars.items():
        plt.figure(figsize=(10, 8), dpi=300)
        sns.countplot(data=df_cat, x=column)
        plt.title(column)
        plt.savefig(FIGURE_DIR / f"{filename}.png")
        plt.close()

    ### Corrélation entre variables quantitatives ###
    corr_matrix = df_num.corr(method="pearson")
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, linewidths=0.5)
    plt.title("Corrélation entre les variables numériques")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_num.png")
    plt.close()

    # Corrélation entre variables numériques et la cible
    df_temp = df_num.copy()
    df_temp["target"] = df["resultat"].map({"admis": 1, "non-admis": 0})

    corr_matrix = df_temp.corr(method="pearson")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, linewidths=0.5)
    plt.title("Corrélation avec la variable cible")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_target.png")
    plt.close()

    print("Terminé.")


if __name__ == "__main__":
    main()
