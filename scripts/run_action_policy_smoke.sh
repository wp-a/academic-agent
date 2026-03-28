#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_RUN="/root/miniconda3/bin/conda run -n base"
MODEL_DIR="models/hf_cache/prajjwal1_bert_tiny"
OUTPUT_DIR="outputs/action_policy_smoke_pipeline"
EXPORT_FILE="data/exports/train_agent_action_classification.jsonl"
TRAIN_FILE="data/exports/train_agent_action_cls_train.jsonl"
EVAL_FILE="data/exports/train_agent_action_cls_eval.jsonl"
TEXT_FILE="$OUTPUT_DIR/infer_input.txt"

mkdir -p outputs

$CONDA_RUN python -m train_agent.trajectories.export_from_deep_review \
  --input data/demo_trajectories.jsonl \
  --output "$EXPORT_FILE"

$CONDA_RUN python -m train_agent.scripts.split_action_data \
  --input "$EXPORT_FILE" \
  --train_output "$TRAIN_FILE" \
  --eval_output "$EVAL_FILE" \
  --eval_ratio 0.25 \
  --seed 42

rm -rf "$OUTPUT_DIR"
$CONDA_RUN python -m train_agent.trainers.train_action_policy \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --model_name_or_path "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_length 256 \
  --max_steps 4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --logging_steps 1 \
  --eval_steps 1 \
  --save_steps 2

$CONDA_RUN python -m train_agent.scripts.write_first_eval_text \
  --input_jsonl "$EVAL_FILE" \
  --output_text "$TEXT_FILE"

$CONDA_RUN python -m train_agent.scripts.infer_action_policy \
  --model_dir "$OUTPUT_DIR" \
  --state_text_file "$TEXT_FILE" \
  --max_length 256
