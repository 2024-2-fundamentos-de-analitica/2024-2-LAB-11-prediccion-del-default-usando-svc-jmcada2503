# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import glob
import gzip
import json
import pickle
import zipfile
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def load_dataset():
    df_test = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    df_train = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    return df_train, df_test

def preprocess_data(df):
    df_temp = df.copy()
    df_temp = df_temp.rename(columns={"default payment next month": "default"})
    df_temp = df_temp.drop(columns=["ID"])
    df_temp = df_temp.loc[df["MARRIAGE"] != 0]
    df_temp = df_temp.loc[df["EDUCATION"] != 0]
    df_temp["EDUCATION"] = df_temp["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    df_temp = df_temp.dropna()
    return df_temp

def separate_features_and_target(df):
    return df.drop(columns=["default"]), df["default"]

def build_pipeline(x_train):
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = list(set(x_train.columns).difference(categorical_features))
    feature_transformer = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            (
                "scaler",
                StandardScaler(with_mean=True, with_std=True),
                numerical_features,
            ),
        ],
        remainder="passthrough",
    )

    model_pipeline = Pipeline(
        [
            ("preprocessor", feature_transformer),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
        ]
    )

    return model_pipeline

def configure_estimator(pipeline, x_train):
    search_params = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }

    cross_validator = StratifiedKFold(n_splits=10)

    scoring_function = make_scorer(balanced_accuracy_score)

    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=search_params, scoring=scoring_function, cv=cross_validator, n_jobs=-1
    )

    return grid_search

def create_output_directory(output_path):
    if os.path.exists(output_path):
        for file in glob(f"{output_path}/*"):
            os.remove(file)
        os.rmdir(output_path)
    os.makedirs(output_path)

def save_model(file_path, estimator):
    create_output_directory("files/models/")
    with gzip.open(file_path, "wb") as f:
        pickle.dump(estimator, f)

def compute_metrics(dataset_name, y_actual, y_predicted):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_actual, y_predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_actual, y_predicted),
        "recall": recall_score(y_actual, y_predicted, zero_division=0),
        "f1_score": f1_score(y_actual, y_predicted, zero_division=0),
    }

def generate_confusion_matrix(dataset_name, y_actual, y_predicted):
    cm = confusion_matrix(y_actual, y_predicted)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

data_train, data_test = load_dataset()
data_train = preprocess_data(data_train)
data_test = preprocess_data(data_test)
x_train, y_train = separate_features_and_target(data_train)
x_test, y_test = separate_features_and_target(data_test)
pipeline = build_pipeline(x_train)

estimator = configure_estimator(pipeline, x_train)
estimator.fit(x_train, y_train)

save_model(
    os.path.join("files/models/", "model.pkl.gz"),
    estimator,
)

y_test_pred = estimator.predict(x_test)
test_metrics = compute_metrics("test", y_test, y_test_pred)
y_train_pred = estimator.predict(x_train)
train_metrics = compute_metrics("train", y_train, y_train_pred)

test_confusion_data = generate_confusion_matrix("test", y_test, y_test_pred)
train_confusion_data = generate_confusion_matrix("train", y_train, y_train_pred)

os.makedirs("files/output/", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(train_metrics) + "\n")
    file.write(json.dumps(test_metrics) + "\n")
    file.write(json.dumps(train_confusion_data) + "\n")
    file.write(json.dumps(test_confusion_data) + "\n")