# Train model pipeline
#
# ## By:
# [Susana Gutiérrez] (https://github.com/Susanagg05)
#
# ## Date:
# 2025-04-07
#
# ## Description:
# This script is a simple training pipeline for a machine learning model.
# It includes data loading, preprocessing, model training, and evaluation.
# Based on Melbourne housing dataset.

# import libraries
# Simple Train Pipeline
#
# ## By:
# [Susana Gutiérrez]https://github.com/Susanagg05)
#
# ## Date:
# 2025-04-07
#
# ## Description:
#
# Pipeline script for training the first selected regression model
# based on the NYC housing dataset.

# Import  libraries


# Base inicial del nuevo pipeline de entrenamiento con datos de Melbourne

# 1. Importar librerías necesarias
# simple_train_pipeline.py corregido actualizado

# Importar librerías necesarias
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.experimental.enable_halving_search_cv  # noqa: F401
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor

# Cargar dataset desde URL
url_data = "https://raw.githubusercontent.com/JoseRZapata/Data_analysis_notebooks/refs/heads/main/data/datasets/Melbourne_housing_FULL_data.csv"
melbourne_df = pd.read_csv(url_data, low_memory=False)

# Reemplazar valores nulos conocidos por np.nan
melbourne_df.replace(["NULL", "None", "", "?", " ", "  ", " -  "], np.nan, inplace=True)

# Conversión de columnas categóricas
categoricas_base = ["Type", "Method", "Suburb", "Regionname", "CouncilArea"]
for col in categoricas_base:
    if col in melbourne_df.columns:
        melbourne_df[col] = melbourne_df[col].astype("category")

# Definir variables relevantes
selected_features = [
    "Type",
    "Method",
    "Suburb",
    "Rooms",
    "Distance",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Regionname",
    "CouncilArea",
    "Price",
]
melbourne_df = melbourne_df[selected_features].copy()

# Convertir columnas numéricas a tipo float
total_numeric_cols = ["Rooms", "Distance", "Bathroom", "Landsize", "BuildingArea", "YearBuilt"]
melbourne_df[total_numeric_cols] = melbourne_df[total_numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Reemplazar outliers extremos por NaN en 'Price'
lower_bound = melbourne_df["Price"].quantile(0.028)
upper_bound = melbourne_df["Price"].quantile(0.99)
melbourne_df.loc[
    (melbourne_df["Price"] < lower_bound) | (melbourne_df["Price"] > upper_bound), "Price"
] = np.nan

# Selección de variables numéricas por correlación
correlation_threshold = 0.3
correlation_matrix = melbourne_df[[*total_numeric_cols, "Price"]].corr()
selected_num_cols = correlation_matrix["Price"].abs().sort_values(ascending=False)
numeric_features = selected_num_cols[selected_num_cols > correlation_threshold].index.tolist()
if "Price" in numeric_features:
    numeric_features.remove("Price")

print("Variables numéricas seleccionadas tras la matriz de correlación:", numeric_features)


# Función para eliminar outliers en variables numéricas
def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df_filtered = df.copy()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = df_filtered[
            (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
        ]
    return df_filtered


# Aplicar limpieza de outliers
melbourne_df = remove_outliers_iqr(melbourne_df, numeric_features)

# Separar variables predictoras y objetivo
target = "Price"
X = melbourne_df.drop(columns=[target])
y = melbourne_df[target]

# Separar en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 🔥 Limpieza de datos para evitar NaNs o infs en entrenamiento y prueba
def clean_data(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = x.copy()
    df["target"] = y
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target"])
    return df.drop(columns=["target"]), df["target"]


x_train, y_train = clean_data(x_train, y_train)
x_test, y_test = clean_data(x_test, y_test)

# Construcción de pipelines para preprocesamiento
categorical_features = ["Type", "Method", "Suburb", "Regionname", "CouncilArea"]

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# Pipeline de modelado
xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("model", XGBRegressor(random_state=42))])

# Hiperparámetros y búsqueda extendida
parametros_xgb = {
    "model__n_estimators": [100, 300, 500],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__max_depth": [3, 5, 7],
}

halving_grid_search_xgb = HalvingGridSearchCV(
    xgb_pipeline, parametros_xgb, factor=2, cv=3, scoring="r2", return_train_score=True, n_jobs=-1
)

halving_grid_search_xgb.fit(x_train, y_train)

print(f"Mejores parámetros para XGBoost: {halving_grid_search_xgb.best_params_}")

mejor_modelo_xgb = halving_grid_search_xgb.best_estimator_

# Evaluación
preds = mejor_modelo_xgb.predict(x_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print("\n--- Métricas de evaluación ---")
print(f"R²    : {r2:.4f}")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")

# Guardar modelo si supera umbral
BASELINE_SCORE = 0.60
if r2 > BASELINE_SCORE:
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    dump(mejor_modelo_xgb, output_path / "melbourne_model.joblib")
    print("\n✅ Modelo guardado en la carpeta 'models'")
else:
    print("\n❌ Modelo no superó el umbral mínimo de validación")
