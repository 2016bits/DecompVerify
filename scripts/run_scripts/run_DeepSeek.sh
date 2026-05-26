#!/usr/bin/env bash
# Full pipeline on HOVER_subset (200 dev claims) with qc_plan7.0.
# Run from the repository root.
#
# max_workers is intentionally low because qc_plan* talks to the DeepSeek-V3.2
# endpoint and the API rate-limits aggressive concurrency.

set -eu

DATASET=HOVER_subset
PLAN=qc_plan7.0
DATA_TYPE=dev
CLASS_NUM=2
START=0
END=200
MAX_WORKERS=2

python scripts/decompose_atomic_facts.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --start "${START}" --end "${END}" --max_workers "${MAX_WORKERS}"

python scripts/generate_question.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --start "${START}" --end "${END}" --max_workers "${MAX_WORKERS}"

python scripts/get_answer.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --start "${START}" --end "${END}" --max_workers "${MAX_WORKERS}"

python scripts/verify_atomic_facts.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --start "${START}" --end "${END}" --max_workers "${MAX_WORKERS}"

python scripts/aggregate_labels.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --start "${START}" --end "${END}" --max_workers "${MAX_WORKERS}"

python scripts/evaluate.py \
  --plan "${PLAN}" --dataset "${DATASET}" --data_type "${DATA_TYPE}" --class_num "${CLASS_NUM}" \
  --results_file_num 1 \
  --in_path "./data/${DATASET}/${PLAN}/${DATA_TYPE}_${CLASS_NUM}_aggregate_${START}_${END}.json" \
  --out_path "./data/${DATASET}/${PLAN}/${DATA_TYPE}_${CLASS_NUM}_eval_by_hop_${START}_${END}.json"
