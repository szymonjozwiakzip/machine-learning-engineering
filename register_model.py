import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# konfiguracja
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "ium-project-hotel-booking"
MODEL_NAME = "hotel-booking-mlp"
TARGET_COLUMN = "is_canceled"
TRAIN_PATH = Path("prepared_data/hotel_booking_train.csv")
DEV_PATH = Path("prepared_data/hotel_booking_dev.csv")
TEST_PATH = Path("prepared_data/hotel_booking_test.csv")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# wczytanie danych
train_data = pd.read_csv(TRAIN_PATH)
dev_data = pd.read_csv(DEV_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]
X_dev = dev_data.drop(columns=[TARGET_COLUMN])
y_dev = dev_data[TARGET_COLUMN]
X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]

numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

print(f"Dane treningowe: {X_train.shape}")
print(f"Cechy numeryczne: {len(numeric_features)}, kategoryczne: {len(categorical_features)}")

# budowanie pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

HIDDEN_LAYER_SIZES = (64, 32)
ACTIVATION = "relu"
MAX_ITER = 80

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver="adam",
        max_iter=MAX_ITER,
        early_stopping=True,
        random_state=42,
    )),
])

# trenowanie
print("Trenowanie modelu...")
model_pipeline.fit(X_train, y_train)


def get_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
    }


train_metrics = get_metrics(y_train, model_pipeline.predict(X_train), "train")
dev_metrics = get_metrics(y_dev, model_pipeline.predict(X_dev), "dev")
test_metrics = get_metrics(y_test, model_pipeline.predict(X_test), "test")

print("Metryki:")
for k, v in {**train_metrics, **dev_metrics, **test_metrics}.items():
    print(f"  {k}: {v:.4f}")

# model card
model_card = f"""# Model Card: Hotel Booking Cancellation Classifier

## Model Details
- **Nazwa modelu**: {MODEL_NAME}
- **Typ**: MLPClassifier (wielowarstwowy perceptron)
- **Architektura**: ukryte warstwy {HIDDEN_LAYER_SIZES}, aktywacja {ACTIVATION}, solver Adam
- **Framework**: scikit-learn
- **Data stworzenia**: 2026-04-25
- **Autor**: student IUM

## Zamierzone zastosowanie
- **Zadanie**: Klasyfikacja binarna – przewidywanie anulowania rezerwacji hotelowej
- **Wyjście**: `0` = rezerwacja utrzymana, `1` = rezerwacja anulowana
- **Przypadki użycia**: systemy zarządzania rezerwacjami, optymalizacja overbookingu

## Dane
- **Źródło**: Hotel Booking Demand dataset (Kaggle)
- **Podział**: train / dev / test
- **Rozmiar zbioru treningowego**: {len(X_train)} wierszy
- **Cechy numeryczne**: {len(numeric_features)}
- **Cechy kategoryczne**: {len(categorical_features)}
- **Kolumna docelowa**: `{TARGET_COLUMN}`

## Wyniki

| Zbiór | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Train | {train_metrics["train_accuracy"]:.4f} | {train_metrics["train_precision"]:.4f} | {train_metrics["train_recall"]:.4f} | {train_metrics["train_f1"]:.4f} |
| Dev   | {dev_metrics["dev_accuracy"]:.4f} | {dev_metrics["dev_precision"]:.4f} | {dev_metrics["dev_recall"]:.4f} | {dev_metrics["dev_f1"]:.4f} |
| Test  | {test_metrics["test_accuracy"]:.4f} | {test_metrics["test_precision"]:.4f} | {test_metrics["test_recall"]:.4f} | {test_metrics["test_f1"]:.4f} |

## Ograniczenia
- Model nie obsługuje nowych kategorii niewidzianych podczas treningu
- Nie uwzględnia sezonowości (brak explicite cech czasowych)
- Wymaga ponownego treningu w przypadku wykrycia data drift

## Etyka i bezpieczeństwo
- Model nie przetwarza danych osobowych
- Decyzje oparte na predykcjach modelu powinny być weryfikowane przez człowieka

## Utrzymanie
- **Monitorowanie**: miesięczna weryfikacja metryk na danych produkcyjnych
- **Retrain trigger**: spadek accuracy poniżej 80% lub wykrycie data drift
"""

# rejestracja w mlflow
with mlflow.start_run(run_name="hotel-booking-mlp-registry") as run:

    mlflow.log_params({
        "hidden_layer_sizes": ",".join(str(x) for x in HIDDEN_LAYER_SIZES),
        "activation": ACTIVATION,
        "max_iter": MAX_ITER,
        "solver": "adam",
        "train_rows": len(X_train),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
    })

    mlflow.log_metrics({**train_metrics, **dev_metrics, **test_metrics})

    # Zapis karty modelu jako artefaktu
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="model_card_", encoding="utf-8"
    ) as f:
        f.write(model_card)
        card_path = f.name

    mlflow.log_artifact(card_path, artifact_path="documentation")

    # Rejestracja modelu w Model Registry
    model_info = mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    print(f"\nRun ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Model zarejestrowany jako: '{MODEL_NAME}'")
    print("Karta modelu zapisana w artefaktach: documentation/model_card_*.md")

# weryfikacja
client = MlflowClient(tracking_uri=TRACKING_URI)

print("\n── Weryfikacja w Model Registry ──")
registered = client.get_registered_model(MODEL_NAME)
print(f"Nazwa: {registered.name}")

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions:
    print(f"  Wersja {v.version}: etap={v.current_stage}, run_id={v.run_id[:8]}...")

# Wylistuj artefakty dokumentacji
latest = versions[0]
arts = client.list_artifacts(latest.run_id, path="documentation")
print("Artefakty (documentation):")
for a in arts:
    print(f"  {a.path} ({a.file_size} B)")
