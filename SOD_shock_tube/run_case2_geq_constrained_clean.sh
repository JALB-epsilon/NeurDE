#!/usr/bin/env bash
set -euo pipefail

cd /home/jx24/NeurDE_clean/NeurDE/SOD_shock_tube

export CUDA_VISIBLE_DEVICES=3
PYTHON_BIN=/scratch/jx24/envs/kinet/bin/python

STAGE1_DIR="results/case2/stage1_geq_constrained_exact_clean"
STAGE2_DIR="results/case2/stage2_geq_constrained_exact_clean"

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"

"$PYTHON_BIN" train_stage_1_geq_constrained.py \
  --device 0 \
  --case 2 \
  --num_samples 500 \
  --batch_size 16 \
  --epochs_override 20 \
  --model_dir_override "$STAGE1_DIR"

PRETRAIN=$(
  find "$STAGE1_DIR" -maxdepth 1 -type f -name 'best_model_2_epoch_*.pt' -printf '%T@ %p\n' \
  | sort -n \
  | tail -1 \
  | cut -d' ' -f2-
)

if [[ -z "${PRETRAIN}" ]]; then
  PRETRAIN=$(
    find "$STAGE1_DIR" -maxdepth 1 -type f -name 'last_model_2_epoch_*.pt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
  )
fi

if [[ -z "${PRETRAIN}" ]]; then
  echo "No pretrained checkpoint found in $STAGE1_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" train_stage_2_geq_constrained_exact.py \
  --device 0 \
  --pre_trained_path "$PRETRAIN" \
  --case 2 \
  --num_samples 500 \
  --batch_size 4 \
  --epochs_override 100 \
  --rollout_override 4 \
  --lr_override 1e-5 \
  --model_dir_override "$STAGE2_DIR"
