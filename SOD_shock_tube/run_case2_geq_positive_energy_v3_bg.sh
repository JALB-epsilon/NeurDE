#!/usr/bin/env bash
# positive_energy v3: exact energy renorm + Levermore shape anchor + longer curriculum.
# Uses the stage-1 positive checkpoint for initialization, not the unstable stage-2 positive N8 run.
# Default GPU 3; override with `GPU_ID=<n>`.
set -e

RESULTS=/scratch/jx24/NeurDE_clean/NeurDE/SOD_shock_tube/results/case2
LOGS=$RESULTS/logs
mkdir -p "$LOGS"
GPU_ID=${GPU_ID:-3}

STAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="$RESULTS/stage2_geq_positive_energy_v3_${STAMP}_gpu${GPU_ID}"

SEED_CKPT="$RESULTS/stage1_geq_positive_main_clean/best_model_2_epoch_100_top_1_loss_0.078.pt"
SHAPE_ANCHOR_WEIGHT=0.02
SHAPE_ANCHOR_DECAY_EPOCHS=75

run_stage() {
    local rollout=$1 seed=$2 outdir=$3 epochs=${4:-150}
    python train_stage_2_geq_exact.py \
        --device "$GPU_ID" --case 2 \
        --geq_mode positive_energy \
        --logit_clip 5.0 \
        --shape_anchor_weight "$SHAPE_ANCHOR_WEIGHT" \
        --shape_anchor_decay_epochs "$SHAPE_ANCHOR_DECAY_EPOCHS" \
        --pre_trained_path "$seed" \
        --model_dir_override "$outdir" \
        --rollout_override "$rollout" \
        --epochs_override "$epochs" \
        --lr_override 1e-5 \
        --batch_size 4 --num_workers 4
}

best_ckpt() {
    local dir=$1
    ls "$dir"/best_model_*.pt 2>/dev/null | sort -t_ -k6 -n | head -1
}

echo "=== positive_energy v3 curriculum start ==="
echo "GPU: $GPU_ID"
echo "Seed: $SEED_CKPT"
echo "Shape anchor: weight=$SHAPE_ANCHOR_WEIGHT decay_epochs=$SHAPE_ANCHOR_DECAY_EPOCHS"

echo "--- N=1 (150ep) ---"
run_stage 1 "$SEED_CKPT" "$BASE_DIR/n1" 150 \
  2>&1 | tee "$LOGS/positive_energy_v3_${STAMP}_gpu${GPU_ID}.log"
N1_BEST=$(best_ckpt "$BASE_DIR/n1")
echo "n1 best : $N1_BEST"

echo "--- N=4 (150ep) ---"
run_stage 4 "$N1_BEST" "$BASE_DIR/n4" 150 \
  2>&1 | tee -a "$LOGS/positive_energy_v3_${STAMP}_gpu${GPU_ID}.log"
N4_BEST=$(best_ckpt "$BASE_DIR/n4")
echo "n4 best : $N4_BEST"

echo "--- N=8 (150ep) ---"
run_stage 8 "$N4_BEST" "$BASE_DIR/n8" 150 \
  2>&1 | tee -a "$LOGS/positive_energy_v3_${STAMP}_gpu${GPU_ID}.log"
N8_BEST=$(best_ckpt "$BASE_DIR/n8")
echo "n8 best : $N8_BEST"

echo "--- N=16 (150ep) ---"
run_stage 16 "$N8_BEST" "$BASE_DIR/n16" 150 \
  2>&1 | tee -a "$LOGS/positive_energy_v3_${STAMP}_gpu${GPU_ID}.log"
N16_BEST=$(best_ckpt "$BASE_DIR/n16")
echo "n16 best : $N16_BEST"

echo "=== positive_energy v3 done ==="
echo "n1  best : $N1_BEST"
echo "n4  best : $N4_BEST"
echo "n8  best : $N8_BEST"
echo "n16 best : $N16_BEST"
