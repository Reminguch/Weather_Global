#!/bin/bash
#SBATCH --job-name=gc-hist20k
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/gc_hist20k_%A_%a.out
#SBATCH --error=logs/gc_hist20k_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"
mkdir -p logs

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASE_OUT="artifacts/checkpoints/graphcast_2020_2021_val2022"
DATA_PATH="data/graphcast/graphcast/dataset/wb2_res1_levels13_2020_2022.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="gpu_2000steps_bs1_e500_hist4"
    INPUT_DURATION="24h"
    N_INPUT_STEPS=4
    ;;
  1)
    RUN_NAME="gpu_2000steps_bs1_e500_hist6"
    INPUT_DURATION="36h"
    N_INPUT_STEPS=6
    ;;
  2)
    RUN_NAME="gpu_2000steps_bs1_e500_hist8"
    INPUT_DURATION="48h"
    N_INPUT_STEPS=8
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

CKPT_IN="${BASE_OUT}/${RUN_NAME}/ckpt_step2000.npz"
if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing resume checkpoint: ${CKPT_IN}"
  exit 1
fi

echo "Running ${RUN_NAME} from step2000 -> step20000"
python -u scripts/training/train_graphcast.py \
  --data-path "${DATA_PATH}" \
  --val-year 2022 \
  --train-start-year 2020 \
  --train-end-year 2021 \
  --resolution 2.0 \
  --mesh-size 4 \
  --width 128 \
  --processor-msg-steps 1 \
  --batch-size 1 \
  --max-steps 20000 \
  --eval-every 2000 \
  --eval-batch-size 32 \
  --checkpoint-every 1000 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 0 \
  --precision bf16 \
  --input-duration "${INPUT_DURATION}" \
  --out-dir "${BASE_OUT}" \
  --run-name "${RUN_NAME}" \
  --ckpt-in "${CKPT_IN}" \
  --stats-dir "${STATS_DIR}" \
  --resume-step 2000

OUT_DIR="plots/analyze_models"
mkdir -p "${OUT_DIR}"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr"

for STEP in 10000 20000; do
  CKPT="${BASE_OUT}/${RUN_NAME}/ckpt_step${STEP}.npz"
  if [[ ! -f "${CKPT}" ]]; then
    echo "Missing checkpoint for eval: ${CKPT}"
    exit 1
  fi

  python scripts/analyze_models/mae_vs_lead.py \
    --model-group baseline \
    --baseline-ckpt "${CKPT}" \
    --dataset-dir "${EVAL_DATA}" \
    --stats-dir "${STATS_DIR}" \
    --n-eval-days 10 \
    --hours-per-step 6 \
    --n-input-steps "${N_INPUT_STEPS}" \
    --max-lead-steps 24 \
    --window-batch-size 8 \
    --output-dir "${OUT_DIR}"

  cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.csv" "${OUT_DIR}/nyc_mae_vs_lead_${RUN_NAME}_step${STEP}.csv"
  cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.png" "${OUT_DIR}/nyc_mae_vs_lead_${RUN_NAME}_step${STEP}.png"
done

echo "Completed ${RUN_NAME} long-run and MAE eval."
