"""Pipeline de préparation des données pour la modélisation KNN.

Gère, dans l'ordre : l'imputation des valeurs manquantes, le traitement des
valeurs aberrantes (méthode IQR), l'encodage One-Hot des variables
catégorielles et la normalisation des variables.

Les transformateurs sont ajustés (`fit`) sur le jeu d'entraînement puis
sauvegardés, afin d'être réappliqués tels quels (`transform` uniquement) sur
le jeu de test — évitant toute fuite d'information entre les deux jeux.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "data" / "models"

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def pipeline(df, train=True, scaler_name="standard"):
    """Prépare un DataFrame pour la modélisation.

    Paramètres
    ----------
    df : pd.DataFrame
        Données à préparer (features uniquement).
    train : bool
        True pour préparer le jeu d'entraînement : les transformateurs sont
        ajustés sur `df` puis sauvegardés. False pour préparer le jeu de
        test : les transformateurs sauvegardés lors de l'entraînement sont
        rechargés et simplement appliqués.
    scaler_name : {"standard", "minmax"}
        Normalisation à appliquer aux données (ignoré si train=False, le
        scaler sauvegardé à l'entraînement est réutilisé).

    Retourne
    -------
    pd.DataFrame
        Données préparées : variables numériques imputées/écrêtées et
        variables catégorielles imputées/encodées, le tout normalisé.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_excel(PROCESSED_DIR / "df.xlsx", index=False)

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

    ### Gestion des valeurs manquantes ###
    if train:
        imputer_num = SimpleImputer(strategy="median")
        imputer_num.fit(df[num_cols])

        imputer_cat = SimpleImputer(strategy="most_frequent")
        imputer_cat.fit(df[cat_cols])

        with open(MODELS_DIR / "miss_value.pickle", "wb") as f:
            pickle.dump((imputer_num, imputer_cat), f)
    else:
        with open(MODELS_DIR / "miss_value.pickle", "rb") as f:
            imputer_num, imputer_cat = pickle.load(f)

    df_num = pd.DataFrame(imputer_num.transform(df[num_cols]), columns=num_cols)
    df_cat = pd.DataFrame(imputer_cat.transform(df[cat_cols]), columns=cat_cols)

    ### Gestion des outliers (méthode IQR, adaptée aux distributions asymétriques) ###
    if train:
        q1, q3 = df_num.quantile(0.25), df_num.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        with open(MODELS_DIR / "outlier.pickle", "wb") as f:
            pickle.dump((lower_bound, upper_bound), f)
    else:
        with open(MODELS_DIR / "outlier.pickle", "rb") as f:
            lower_bound, upper_bound = pickle.load(f)

    df_num = df_num.clip(lower=lower_bound, upper=upper_bound, axis=1)

    ### Encodage One-Hot des variables catégorielles ###
    if train:
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoder.fit(df_cat)
        with open(MODELS_DIR / "encoder.pickle", "wb") as f:
            pickle.dump(cat_encoder, f)
    else:
        with open(MODELS_DIR / "encoder.pickle", "rb") as f:
            cat_encoder = pickle.load(f)

    encoded = cat_encoder.transform(df_cat)
    encoded_cols = cat_encoder.get_feature_names_out(df_cat.columns)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_cat.index)

    ### Normalisation ###
    df_final = pd.concat([df_num, df_encoded], axis=1)

    if train:
        scaler = SCALERS[scaler_name]()
        scaler.fit(df_final)
        with open(MODELS_DIR / "scaler.pickle", "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(MODELS_DIR / "scaler.pickle", "rb") as f:
            scaler = pickle.load(f)

    scaled = scaler.transform(df_final)
    df_scaled = pd.DataFrame(scaled, columns=df_final.columns, index=df_final.index)
    df_scaled.to_excel(PROCESSED_DIR / "df_scaled.xlsx", index=False)

    return df_scaled
