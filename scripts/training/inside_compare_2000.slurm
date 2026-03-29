#!/bin/bash
#SBATCH --job-name=gc-inside2k
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/gc_inside2k_%A_%a.out
#SBATCH --error=logs/gc_inside2k_%A_%a.err

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
OUT_DIR="plots/analyze_models"
DATA_PATH="data/graphcast/graphcast/dataset/wb2_res1_levels13_2020_2022.zarr"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
mkdir -p "${OUT_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="cmp2000_inside_hist2_none"
    INPUT_DURATION="12h"
    N_INPUT_STEPS=2
    TEMPORAL_BACKBONE="none"
    ;;
  1)
    RUN_NAME="cmp2000_inside_hist2_mamba"
    INPUT_DURATION="12h"
    N_INPUT_STEPS=2
    TEMPORAL_BACKBONE="mamba"
    ;;
  2)
    RUN_NAME="cmp2000_inside_hist4_none"
    INPUT_DURATION="24h"
    N_INPUT_STEPS=4
    TEMPORAL_BACKBONE="none"
    ;;
  3)
    RUN_NAME="cmp2000_inside_hist4_mamba"
    INPUT_DURATION="24h"
    N_INPUT_STEPS=4
    TEMPORAL_BACKBONE="mamba"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Running ${RUN_NAME}"
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
  --max-steps 2000 \
  --eval-every 500 \
  --eval-batch-size 8 \
  --checkpoint-every 500 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 0 \
  --precision bf16 \
  --input-duration "${INPUT_DURATION}" \
  --temporal-backbone "${TEMPORAL_BACKBONE}" \
  --temporal-location mesh_processor_interleaved \
  --out-dir "${BASE_OUT}" \
  --run-name "${RUN_NAME}"

for STEP in 500 1000 2000; do
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

  cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.csv" \
    "${OUT_DIR}/nyc_mae_vs_lead_${RUN_NAME}_step${STEP}.csv"
  cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.png" \
    "${OUT_DIR}/nyc_mae_vs_lead_${RUN_NAME}_step${STEP}.png"
done

echo "Completed ${RUN_NAME}"
