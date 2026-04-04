#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "is_canceled"


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=80,
        early_stopping=True,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train(train_path: Path, model_path: Path) -> None:
    data = pd.read_csv(train_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Brakuje kolumny docelowej: {TARGET_COLUMN}")

    x_train = data.drop(columns=[TARGET_COLUMN])
    y_train = data[TARGET_COLUMN]

    numeric_features = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in x_train.columns if c not in numeric_features]

    model = build_model(numeric_features, categorical_features)
    model.fit(x_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model zapisany do: {model_path}")


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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args.train, args.model)


if __name__ == "__main__":
    main()
