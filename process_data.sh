#!/usr/bin/env bash
set -euo pipefail

CUTOFF_VAL=${1:-500}

OUTPUT_DIR="output_dataset"
DATA_URL="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
RAW_FILE="${OUTPUT_DIR}/hotel_bookings_raw.csv"
SHUFFLED_CUT_BODY="${OUTPUT_DIR}/hotel_bookings_shuffled_cut.tmp"

mkdir -p "${OUTPUT_DIR}"

{
    echo "=== process_data.sh ==="
    echo "Pobieranie danych: ${DATA_URL}"
    echo "Zastosowana wartość CUTOFF: ${CUTOFF_VAL}"
} > process_log.txt

wget -q -O "${RAW_FILE}" "${DATA_URL}"

HEADER_LINE="$(head -n 1 "${RAW_FILE}")"

tail -n +2 "${RAW_FILE}" | shuf | head -n "${CUTOFF_VAL}" > "${SHUFFLED_CUT_BODY}"

TOTAL_ROWS="$(wc -l < "${SHUFFLED_CUT_BODY}")"
TRAIN_ROWS=$((TOTAL_ROWS * 70 / 100))
DEV_ROWS=$((TOTAL_ROWS * 15 / 100))
TEST_ROWS=$((TOTAL_ROWS - TRAIN_ROWS - DEV_ROWS))

head -n "${TRAIN_ROWS}" "${SHUFFLED_CUT_BODY}" > "${OUTPUT_DIR}/train_body.tmp"
tail -n +$((TRAIN_ROWS + 1)) "${SHUFFLED_CUT_BODY}" > "${OUTPUT_DIR}/remaining_body.tmp"
head -n "${DEV_ROWS}" "${OUTPUT_DIR}/remaining_body.tmp" > "${OUTPUT_DIR}/dev_body.tmp"
tail -n "${TEST_ROWS}" "${OUTPUT_DIR}/remaining_body.tmp" > "${OUTPUT_DIR}/test_body.tmp"

{ echo "${HEADER_LINE}"; cat "${OUTPUT_DIR}/train_body.tmp"; } > "${OUTPUT_DIR}/hotel_booking_train.csv"
{ echo "${HEADER_LINE}"; cat "${OUTPUT_DIR}/dev_body.tmp"; } > "${OUTPUT_DIR}/hotel_booking_dev.csv"
{ echo "${HEADER_LINE}"; cat "${OUTPUT_DIR}/test_body.tmp"; } > "${OUTPUT_DIR}/hotel_booking_test.csv"

cut -d',' -f1-8 "${RAW_FILE}" | head -n $((CUTOFF_VAL + 1)) > "${OUTPUT_DIR}/hotel_booking_reduced.csv"

rm -f "${OUTPUT_DIR}"/*.tmp "${SHUFFLED_CUT_BODY}"

{
    echo "Przetwarzanie zakończone pomyślnie."
    echo "Liczba rekordów po odcięciu (CUTOFF): ${TOTAL_ROWS}"
    echo "Podział: train=${TRAIN_ROWS}, dev=${DEV_ROWS}, test=${TEST_ROWS}"
    echo "Wygenerowane pliki w folderze ${OUTPUT_DIR}:"
    ls -1 "${OUTPUT_DIR}"/*.csv
} >> process_log.txt