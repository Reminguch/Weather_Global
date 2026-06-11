#!/bin/bash
#SBATCH --job-name=res1-gcm-k50
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/res1_official_gc_mamba_bptt16_k50_%x_%A_%a.out
#SBATCH --error=logs/res1_official_gc_mamba_bptt16_k50_%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt16_k_sweep}

RUN_NAMES=(
  official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k1_bptt16_bs1_optimal
  official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k4_bptt16_bs1_optimal
  official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k8_bptt16_bs1_optimal
)
TARGET_STEPS_SET=(1 4 8)
RESUME_STEPS=(12000 20000 16000)
MAX_STEPS_SET=(62000 70000 66000)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
EXPECTED_TASKS=${#RUN_NAMES[@]}
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${TASK_ID} outside expected range 0-$((EXPECTED_TASKS - 1))" >&2
  exit 1
fi

RUN_NAME="${RUN_NAMES[$TASK_ID]}"
TARGET_STEPS="${TARGET_STEPS_SET[$TASK_ID]}"
RESUME_STEP="${RESUME_STEPS[$TASK_ID]}"
MAX_STEPS="${MAX_STEPS_SET[$TASK_ID]}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
RESUME_CKPT="${RUN_DIR}/ckpt_step${RESUME_STEP}.npz"

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/res1" ]]; then
  echo "Missing prepared res1 data store: ${PREPARED_DATA_ROOT}/res1" >&2
  exit 1
fi

echo "Continuing ${RUN_NAME}: target_steps=${TARGET_STEPS} resume_step=${RESUME_STEP} max_steps=${MAX_STEPS}"
echo "Resume checkpoint: ${RESUME_CKPT}"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year 2022
  --train-start-year 2015
  --train-end-year 2021
  --resolution 1.0
  --mesh-size 5
  --width 512
  --processor-msg-steps 16
  --input-duration 12h
  --target-steps "${TARGET_STEPS}"
  --len-segment 32
  --bptt-steps 16
  --chunk-load-workers 6
  --batch-size 1
  --eval-batch-size 1
  --eval-num-segments 16
  --final-eval-num-segments all
  --data-cache-mode never
  --data-cache-max-gib 48
  --batch-builder prepared_array
  --max-steps "${MAX_STEPS}"
  --eval-every 2000
  --checkpoint-every 2000
  --lr 1e-4
  --weight-decay 1e-4
  --precision bf16
  --seed 0
  --memory-mode optimal
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner 128
  --temporal-d-state 64
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers 1
  --temporal-dropout 0.0
  --temporal-insert-count 2
  --ckpt-in "${RESUME_CKPT}"
  --resume-step "${RESUME_STEP}"
  --trainable-part mamba
  --zero-init-temporal-out
)

printf 'Resolved command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
