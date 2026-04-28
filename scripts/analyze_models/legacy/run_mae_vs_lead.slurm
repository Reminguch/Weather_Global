#!/bin/bash
#SBATCH --job-name=gc-mae
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gc_mae_%j.out
#SBATCH --error=logs/gc_mae_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MODEL_GROUP="${MODEL_GROUP:-res2}"     # res2 | res4 | baseline
DATASET_DIR="${DATASET_DIR:-data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc}"
N_EVAL_DAYS="${N_EVAL_DAYS:-40}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-8}"
MAX_LEAD_STEPS="${MAX_LEAD_STEPS:-}"   # optional override; defaults in script
DISCARD_START_STEPS="${DISCARD_START_STEPS:-14}"
DISCARD_END_STEPS="${DISCARD_END_STEPS:-0}"

CMD=(
  python -u scripts/analyze_models/legacy/mae_vs_lead.py
  --model-group "${MODEL_GROUP}"
  --dataset-dir "${DATASET_DIR}"
  --n-eval-days "${N_EVAL_DAYS}"
  --window-batch-size "${WINDOW_BATCH_SIZE}"
  --discard-start-steps "${DISCARD_START_STEPS}"
  --discard-end-steps "${DISCARD_END_STEPS}"
)

if [[ -n "${MAX_LEAD_STEPS}" ]]; then
  CMD+=(--max-lead-steps "${MAX_LEAD_STEPS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
