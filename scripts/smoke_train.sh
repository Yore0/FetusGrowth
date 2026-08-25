#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DATASET="${REPO_ROOT}/data/smoke_train_regression.jsonl" \\
VAL_DATASET="${REPO_ROOT}/data/smoke_val_regression.jsonl" \\
OUTPUT_DIR="${REPO_ROOT}/outputs/smoke" \\
RUN_NAME="continuous_v2_smoke" \\
MAX_STEPS="${MAX_STEPS:-4}" \\
EVAL_STEPS="${EVAL_STEPS:-2}" \\
EVAL_DELAY="${EVAL_DELAY:-0}" \\
SAVE_STEPS="${SAVE_STEPS:-2}" \\
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \\
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \\
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \\
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \\
bash "${REPO_ROOT}/scripts/train.sh"
