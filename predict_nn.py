#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import pandas as pd

TARGET_COLUMN = "is_canceled"


def predict(model_path: Path, test_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
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
        help="Ścieżka do zapisanego modelu",
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
    predict(args.model, args.test, args.output)


if __name__ == "__main__":
    main()
