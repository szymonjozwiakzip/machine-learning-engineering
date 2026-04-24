# Model Card: Hotel Booking Cancellation Classifier

## Model Details
- **Nazwa modelu**: hotel-booking-mlp
- **Typ**: MLPClassifier (wielowarstwowy perceptron)
- **Architektura**: ukryte warstwy (64, 32), aktywacja relu, solver Adam
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
- **Rozmiar zbioru treningowego**: 298 wierszy
- **Cechy numeryczne**: 18
- **Cechy kategoryczne**: 12
- **Kolumna docelowa**: `is_canceled`

## Wyniki

| Zbiór | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Train | 0.8993 | 1.0000 | 0.7479 | 0.8558 |
| Dev   | 0.8283 | 0.9231 | 0.6154 | 0.7385 |
| Test  | 0.7921 | 1.0000 | 0.5714 | 0.7273 |

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
