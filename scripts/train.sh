#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MS_SWIFT_ROOT:?Set MS_SWIFT_ROOT to an ms-swift checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-9B base model}"

TRAIN_DATASET="${TRAIN_DATASET:-${REPO_ROOT}/data/train_continuous_v2_regression.jsonl}"
VAL_DATASET="${VAL_DATASET:-${REPO_ROOT}/data/val_continuous_clean_regression.jsonl}"
PLUGIN="${REPO_ROOT}/plugins/outcome_regression.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29609}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
EVAL_STEPS="${EVAL_STEPS:-250}"
EVAL_DELAY="${EVAL_DELAY:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
TRAIN_SEED="${TRAIN_SEED:-20260809}"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-continuous_v2_ce_reg_linear_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/train_${RUN_TIMESTAMP}.log"

for path in "${TRAIN_DATASET}" "${VAL_DATASET}" "${PLUGIN}"; do
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${MS_SWIFT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[config] repo_root=${REPO_ROOT}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] model_loss=ce_plus_standardized_smooth_l1_linear_heads"
echo "[config] effective_batch=$((NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))"

NPROC_PER_NODE="${NPROC_PER_NODE}" \\
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \\
MASTER_PORT="${MASTER_PORT}" \\
OUTCOME_HEAD_ARCH="linear" \\
OUTCOME_HEAD_DROPOUT="0" \\
OUTCOME_LOSS_MODE="ce_plus_regression" \\
OUTCOME_CE_WEIGHT="1.0" \\
OUTCOME_REG_WEIGHT="1.0" \\
OUTCOME_LOSS_TYPE="smooth_l1" \\
OUTCOME_HUBER_BETA="0.5" \\
OUTCOME_CENTERED_LOSS_WEIGHT="0" \\
OUTCOME_OUTPUT_INIT_STD="0.001" \\
swift sft \\
  --model "${MODEL_PATH}" \\
  --dataset "${TRAIN_DATASET}" \\
  --val_dataset "${VAL_DATASET}" \\
  --split_dataset_ratio 0 \\
  --dataset_shuffle false \\
  --train_dataloader_shuffle true \\
  --remove_unused_columns false \\
  --new_special_tokens \\
    '<|delivery_outcome_query|>' \\
    '<|birth_weight_outcome_query|>' \\
    '<|birth_length_outcome_query|>' \\
  --external_plugins "${PLUGIN}" \\
  --seed "${TRAIN_SEED}" \\
  --data_seed "${TRAIN_SEED}" \\
  --torch_dtype bfloat16 \\
  --tuner_type lora \\
  --target_modules all-linear \\
  --modules_to_save outcome_regression \\
  --freeze_vit true \\
  --freeze_aligner true \\
  --max_steps "${MAX_STEPS}" \\
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \\
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \\
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \\
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \\
  --learning_rate "${LEARNING_RATE}" \\
  --weight_decay "${WEIGHT_DECAY}" \\
  --optimizer default \\
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \\
  --warmup_ratio "${WARMUP_RATIO}" \\
  --lora_rank 128 \\
  --lora_alpha 256 \\
  --lora_dtype bfloat16 \\
  --gradient_checkpointing true \\
  --attn_impl sdpa \\
  --use_logits_to_keep true \\
  --use_liger_kernel false \\
  --packing false \\
  --padding_free false \\
  --lazy_tokenize true \\
  --enable_thinking false \\
  --add_non_thinking_prefix false \\
  --disable_ignore_empty_think true \\
  --dataset_num_proc 24 \\
  --dataloader_num_workers 8 \\
  --dataloader_persistent_workers true \\
  --dataloader_prefetch_factor 4 \\
  --eval_strategy steps \\
  --eval_steps "${EVAL_STEPS}" \\
  --eval_delay "${EVAL_DELAY}" \\
  --save_strategy steps \\
  --save_steps "${SAVE_STEPS}" \\
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \\
  --logging_steps "${LOGGING_STEPS}" \\
  --logging_first_step true \\
  --load_best_model_at_end true \\
  --metric_for_best_model loss \\
  --greater_is_better false \\
  --max_length "${MAX_LENGTH}" \\
  --deepspeed zero2 \\
  --output_dir "${OUTPUT_DIR}" \\
  --run_name "${RUN_NAME}" \\
  --report_to tensorboard
