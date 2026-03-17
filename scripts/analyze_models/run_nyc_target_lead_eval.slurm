#!/bin/bash
#SBATCH --job-name=gc-nyc-lead
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gc_nyc_targetlead_%j.out
#SBATCH --error=logs/gc_nyc_targetlead_%j.err

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DATASET_DIR="${DATASET_DIR:-data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc}"
STATS_DIR="${STATS_DIR:-data/graphcast/graphcast/stats}"
CKPT_ROOT="${CKPT_ROOT:-artifacts/checkpoints/graphcast_res4_stream}"
RUN_PREFIX="${RUN_PREFIX:-res4}"
TARGET_LEAD_DAYS="${TARGET_LEAD_DAYS:-6}"
N_EVAL_DAYS="${N_EVAL_DAYS:-40}"
N_EXTRA_STEPS="${N_EXTRA_STEPS:-14}"
OUTPUT_DIR="${OUTPUT_DIR:-plots/analyze_models}"
OUTPUT_STEM="${OUTPUT_STEM:-nyc_targetlead_${RUN_PREFIX}_${TARGET_LEAD_DAYS}d}"
SINGLE_CKPT="${SINGLE_CKPT:-}"

CMD=(
  python -u scripts/analyze_models/nyc_target_lead_eval.py
  --dataset-dir "${DATASET_DIR}"
  --stats-dir "${STATS_DIR}"
  --ckpt-root "${CKPT_ROOT}"
  --run-prefix "${RUN_PREFIX}"
  --target-lead-days "${TARGET_LEAD_DAYS}"
  --n-eval-days "${N_EVAL_DAYS}"
  --n-extra-steps "${N_EXTRA_STEPS}"
  --output-dir "${OUTPUT_DIR}"
  --output-stem "${OUTPUT_STEM}"
)

if [[ -n "${SINGLE_CKPT}" ]]; then
  CMD+=(--single-ckpt "${SINGLE_CKPT}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
