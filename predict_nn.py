#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path

import joblib
import pandas as pd

TARGET_COLUMN = "is_canceled"


def load_mlflow_sklearn_module():
    try:
        return importlib.import_module("mlflow.sklearn")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow nie jest zainstalowany. Uruchom: uv sync (lub pip install mlflow)."
        ) from exc


def load_model(model_path: Path | None, model_uri: str | None, model_uri_file: Path | None):
    mlflow_sklearn = None

    if model_uri:
        mlflow_sklearn = load_mlflow_sklearn_module()
        print(f"Wczytywanie modelu z MLflow URI: {model_uri}")
        return mlflow_sklearn.load_model(model_uri)

    if model_uri_file is not None and model_uri_file.exists():
        uri_from_file = model_uri_file.read_text(encoding="utf-8").strip()
        if uri_from_file:
            mlflow_sklearn = load_mlflow_sklearn_module()
            print(f"Wczytywanie modelu z MLflow URI z pliku: {uri_from_file}")
            return mlflow_sklearn.load_model(uri_from_file)

    if model_path is None:
        raise ValueError("Musisz podać --model albo --model-uri (lub istniejący --model-uri-file)")

    print(f"Wczytywanie modelu z pliku joblib: {model_path}")
    return joblib.load(model_path)


def predict(model_path: Path | None, model_uri: str | None, model_uri_file: Path | None, test_path: Path, output_path: Path) -> None:
    model = load_model(model_path, model_uri, model_uri_file)
    test_data = pd.read_csv(test_path)

    x_test = test_data.drop(columns=[TARGET_COLUMN], errors="ignore")
    predictions = model.predict(x_test)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction_is_canceled": predictions}).to_csv(output_path, index=False)
    print(f"Predykcje zapisane do: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inferencja przy pomocy wytrenowanego modelu")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hotel_mlp.joblib"),
        help="Ścieżka do zapisanego modelu joblib",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="URI modelu MLflow, np. runs:/<run_id>/model lub models:/<name>/Production",
    )
    parser.add_argument(
        "--model-uri-file",
        type=Path,
        default=Path("models/latest_mlflow_model_uri.txt"),
        help="Plik z zapisanym URI modelu MLflow",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("prepared_data/hotel_booking_test.csv"),
        help="Ścieżka do zbioru testowego CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/hotel_booking_test_predictions.csv"),
        help="Ścieżka zapisu predykcji",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    predict(args.model, args.model_uri, args.model_uri_file, args.test, args.output)


if __name__ == "__main__":
    main()
