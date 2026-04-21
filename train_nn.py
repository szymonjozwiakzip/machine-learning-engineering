#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "is_canceled"
DEFAULT_EXPERIMENT_NAME = "ium-project-hotel-booking"


def load_mlflow_modules():
    try:
        mlflow_module = importlib.import_module("mlflow")
        mlflow_sklearn_module = importlib.import_module("mlflow.sklearn")
        return mlflow_module, mlflow_sklearn_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow nie jest zainstalowany. Uruchom: uv sync (lub pip install mlflow)."
        ) from exc


def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    max_iter: int = 80,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train(
    train_path: Path,
    model_path: Path,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    max_iter: int = 80,
    dev_path: Path | None = Path("prepared_data/hotel_booking_dev.csv"),
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str | None = None,
    mlflow_uri_output: Path = Path("models/latest_mlflow_model_uri.txt"),
) -> None:
    data = pd.read_csv(train_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Brakuje kolumny docelowej: {TARGET_COLUMN}")

    x_train = data.drop(columns=[TARGET_COLUMN])
    y_train = data[TARGET_COLUMN]

    numeric_features = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in x_train.columns if c not in numeric_features]

    if tracking_uri:
        mlflow, mlflow_sklearn = load_mlflow_modules()
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow, mlflow_sklearn = load_mlflow_modules()
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "hidden_layer_sizes": ",".join(str(x) for x in hidden_layer_sizes),
                "activation": activation,
                "max_iter": max_iter,
                "train_rows": len(data),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "train_path": str(train_path),
            }
        )

        model = build_model(numeric_features, categorical_features, hidden_layer_sizes, activation, max_iter)
        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        for name, value in calculate_metrics(y_train, train_pred, prefix="train").items():
            mlflow.log_metric(name, value)

        if dev_path is not None and dev_path.exists():
            dev_data = pd.read_csv(dev_path)
            if TARGET_COLUMN in dev_data.columns:
                x_dev = dev_data.drop(columns=[TARGET_COLUMN])
                y_dev = dev_data[TARGET_COLUMN]
                dev_pred = model.predict(x_dev)
                for name, value in calculate_metrics(y_dev, dev_pred, prefix="dev").items():
                    mlflow.log_metric(name, value)
                mlflow.log_param("dev_path", str(dev_path))

        model_info = mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        mlflow_uri_output.parent.mkdir(parents=True, exist_ok=True)
        mlflow_uri_output.write_text(model_info.model_uri, encoding="utf-8")

        mlflow.log_param("joblib_model_path", str(model_path))
        mlflow.log_param("mlflow_model_uri", model_info.model_uri)

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"MLflow model URI: {model_info.model_uri}")
        print(f"URI modelu zapisane do: {mlflow_uri_output}")
        print(f"Model (joblib) zapisany do: {model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trenowanie sieci neuronowej (MLPClassifier)")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("prepared_data/hotel_booking_train.csv"),
        help="Ścieżka do zbioru treningowego CSV",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hotel_mlp.joblib"),
        help="Ścieżka zapisu wytrenowanego modelu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Maksymalna liczba epok trenowania (max_iter MLPClassifier)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="64,32",
        help="Rozmiary warstw ukrytych oddzielone przecinkami, np. 128,64,32",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "logistic"],
        help="Funkcja aktywacji neuronów",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("prepared_data/hotel_booking_dev.csv"),
        help="Ścieżka do zbioru walidacyjnego CSV (opcjonalnie do metryk dev)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Tracking URI MLflow, np. http://localhost:5000",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="Nazwa eksperymentu MLflow",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help="Opcjonalna nazwa modelu do rejestru MLflow",
    )
    parser.add_argument(
        "--mlflow-uri-output",
        type=Path,
        default=Path("models/latest_mlflow_model_uri.txt"),
        help="Plik, w którym zostanie zapisany URI modelu MLflow",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    hidden_layer_sizes = tuple(int(x) for x in args.hidden_layers.split(","))
    train(
        args.train,
        args.model,
        hidden_layer_sizes,
        args.activation,
        args.epochs,
        args.dev,
        args.tracking_uri,
        args.experiment_name,
        args.registered_model_name,
        args.mlflow_uri_output,
    )


if __name__ == "__main__":
    main()
