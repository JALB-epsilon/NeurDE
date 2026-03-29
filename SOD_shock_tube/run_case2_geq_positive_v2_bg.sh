#!/usr/bin/env bash
set -euo pipefail

# Positive-v2: stable positive head + exact-Sod Geq teacher + soft energy penalty.
# Separate experiment from the original positive curriculum.
# Curriculum: N=1 (100 ep) -> N=4 (100 ep) -> N=8 (100 ep) -> N=16 (150 ep).

GPU_ID="${1:-5}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/jx24/envs/kinet/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_case2_positive_v2}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"

LOG_DIR="results/case2/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/positive_v2_${STAMP}_gpu${GPU_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

STAGE1_CKPT="results/case2/stage1_geq_positive_main_clean/best_model_2_epoch_100_top_1_loss_0.078.pt"
BASE_DIR="results/case2/stage2_geq_positive_v2_${STAMP}_gpu${GPU_ID}"
mkdir -p "${BASE_DIR}"

EXACT_GEQ_TEACHER_WEIGHT="${EXACT_GEQ_TEACHER_WEIGHT:-0.05}"
EXACT_GEQ_TEACHER_DECAY_EPOCHS="${EXACT_GEQ_TEACHER_DECAY_EPOCHS:-100}"
SOFT_ENERGY_WEIGHT="${SOFT_ENERGY_WEIGHT:-0.01}"
SOFT_ENERGY_DECAY_EPOCHS="${SOFT_ENERGY_DECAY_EPOCHS:-100}"

run_stage() {
  local rollout="$1" epochs="$2" pretrain="$3" model_dir="$4"
  echo "==> positive_v2 rollout=${rollout}, epochs=${epochs}"
  mkdir -p "${model_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_stage_2_geq_exact.py \
    --device 0 \
    --case 2 \
    --num_samples 500 \
    --batch_size 4 \
    --num_workers 4 \
    --pre_trained_path "${pretrain}" \
    --model_dir_override "${model_dir}" \
    --epochs_override "${epochs}" \
    --rollout_override "${rollout}" \
    --lr_override 1e-5 \
    --geq_mode positive \
    --exact_geq_teacher_weight "${EXACT_GEQ_TEACHER_WEIGHT}" \
    --exact_geq_teacher_decay_epochs "${EXACT_GEQ_TEACHER_DECAY_EPOCHS}" \
    --soft_energy_weight "${SOFT_ENERGY_WEIGHT}" \
    --soft_energy_decay_epochs "${SOFT_ENERGY_DECAY_EPOCHS}"
}

get_best() {
  find "$1" -maxdepth 1 -type f -name 'best_model*.pt' | sort | tail -n 1
}

echo "=== positive_v2 curriculum start ==="
echo "GPU: ${GPU_ID}"
echo "Log: ${LOG_FILE}"
echo "Exact Geq teacher: weight=${EXACT_GEQ_TEACHER_WEIGHT}, decay_epochs=${EXACT_GEQ_TEACHER_DECAY_EPOCHS}"
echo "Soft energy penalty: weight=${SOFT_ENERGY_WEIGHT}, decay_epochs=${SOFT_ENERGY_DECAY_EPOCHS}"

N1_DIR="${BASE_DIR}/n1"
run_stage 1 100 "${STAGE1_CKPT}" "${N1_DIR}"
N1_BEST="$(get_best "${N1_DIR}")"
[[ -z "${N1_BEST}" ]] && { echo "No best ckpt after n1"; exit 1; }

N4_DIR="${BASE_DIR}/n4"
run_stage 4 100 "${N1_BEST}" "${N4_DIR}"
N4_BEST="$(get_best "${N4_DIR}")"
[[ -z "${N4_BEST}" ]] && { echo "No best ckpt after n4"; exit 1; }

N8_DIR="${BASE_DIR}/n8"
run_stage 8 100 "${N4_BEST}" "${N8_DIR}"
N8_BEST="$(get_best "${N8_DIR}")"
[[ -z "${N8_BEST}" ]] && { echo "No best ckpt after n8"; exit 1; }

N16_DIR="${BASE_DIR}/n16"
run_stage 16 150 "${N8_BEST}" "${N16_DIR}"
N16_BEST="$(get_best "${N16_DIR}")"

echo "=== positive_v2 curriculum done ==="
echo "n1 best : ${N1_BEST}"
echo "n4 best : ${N4_BEST}"
echo "n8 best : ${N8_BEST}"
echo "n16 best : ${N16_BEST:-none}"
