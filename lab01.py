#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def load_raw_dataset() -> pd.DataFrame:
    local_candidates = [
        Path("output_dataset/hotel_bookings_raw.csv"),
        Path("data/hotel_bookings_raw.csv"),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
    return pd.read_csv(data_url)


def create_dataset(cutoff: int) -> None:
    output_dir = Path("prepared_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    hotel_data = load_raw_dataset()
    if cutoff > 0:
        hotel_data = hotel_data.sample(n=min(cutoff, len(hotel_data)), random_state=42)

    hotel_data = hotel_data.drop("company", axis=1, errors="ignore")
    hotel_data["agent"] = hotel_data["agent"].fillna(0)
    hotel_data["children"] = hotel_data["children"].fillna(0)
    hotel_data_clean = hotel_data.dropna(subset=["country"])

    shuffled = hotel_data_clean.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_rows = len(shuffled)
    n_train = int(n_rows * 0.6)
    n_dev = int(n_rows * 0.2)

    train_data = shuffled.iloc[:n_train]
    dev_data = shuffled.iloc[n_train:n_train + n_dev]
    test_data = shuffled.iloc[n_train + n_dev:]

    train_data.to_csv(output_dir / "hotel_booking_train.csv", index=False)
    dev_data.to_csv(output_dir / "hotel_booking_dev.csv", index=False)
    test_data.to_csv(output_dir / "hotel_booking_test.csv", index=False)

    with open("process_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write("=== lab01.py create-dataset ===\n")
        log_file.write(f"cutoff: {cutoff}\n")
        log_file.write(f"train rows: {train_data.shape[0]}\n")
        log_file.write(f"dev rows: {dev_data.shape[0]}\n")
        log_file.write(f"test rows: {test_data.shape[0]}\n")


def dataset_stats(input_path: str, output_path: str) -> None:
    data = pd.read_csv(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as stats_file:
        stats_file.write("=== Dataset stats ===\n")
        stats_file.write(f"source: {input_path}\n")
        stats_file.write(f"rows: {data.shape[0]}\n")
        stats_file.write(f"columns: {data.shape[1]}\n\n")
        stats_file.write("Missing values by column:\n")
        stats_file.write(data.isnull().sum().to_string())
        stats_file.write("\n\nNumeric summary:\n")
        stats_file.write(data.describe().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset processing for IUM labs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_create = subparsers.add_parser("create-dataset")
    parser_create.add_argument("--cutoff", type=int, default=500)

    parser_stats = subparsers.add_parser("dataset-stats")
    parser_stats.add_argument("--input", required=True)
    parser_stats.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create-dataset":
        create_dataset(args.cutoff)
    elif args.command == "dataset-stats":
        dataset_stats(args.input, args.output)


if __name__ == "__main__":
    main()
