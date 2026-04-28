#!/bin/bash
#SBATCH --job-name=mamba-wmse
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH --output=logs/mamba_wmse_%j.out
#SBATCH --error=logs/mamba_wmse_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

N_EVAL_DAYS="${N_EVAL_DAYS:-365}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-8}"
RESOLUTIONS="${RESOLUTIONS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-plots/analyze_models/data/mamba_res}"
OUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR:-plots/analyze_models/images/mamba_res}"
TMP_SHARD_DIR="${TMP_SHARD_DIR:-${OUTPUT_DATA_DIR}/single_run}"

CMD=(
  python -u scripts/analyze_models/unified_resolution_eval.py
  --families graphcast residual_mamba
  --metrics weighted_allvars per_variable
  --eval-modes cold warm
  --n-eval-days "${N_EVAL_DAYS}"
  --lead-days 1 2 4
  --window-batch-size "${WINDOW_BATCH_SIZE}"
  --warmup-steps "${WARMUP_STEPS}"
  --trunk-steps "${TRUNK_STEPS}"
  --output-data-dir "${TMP_SHARD_DIR}"
  --output-csv-name "mamba_wmse_vs_res_single.csv"
)

if [[ -n "${RESOLUTIONS}" ]]; then
  CMD+=(--resolutions ${RESOLUTIONS})
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

MERGE_CMD=(
  python -u scripts/analyze_models/merge_resolution_eval.py
  --input-data-dir "${TMP_SHARD_DIR}"
  --output-data-dir "${OUTPUT_DATA_DIR}"
  --output-image-dir "${OUTPUT_IMAGE_DIR}"
  --output-csv-name "mamba_wmse_vs_res.csv"
  --shard-glob "mamba_wmse_vs_res_*.csv"
  --plot-prefix "mamba"
  --lead-days 1 2 4
)

echo "Running: ${MERGE_CMD[*]}"
"${MERGE_CMD[@]}"
