#!/bin/bash
#SBATCH --job-name=mamba-wmse
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH --output=logs/mamba_wmse_%A_%a.out
#SBATCH --error=logs/mamba_wmse_%A_%a.err

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
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
SHARD_DATA_DIR="${SHARD_DATA_DIR:-plots/analyze_models/data/mamba_res/shards}"
RESOLUTIONS="${RESOLUTIONS:?RESOLUTIONS must be provided as a space-separated list}"

read -r -a RES_ARRAY <<< "${RESOLUTIONS}"
RES="${RES_ARRAY[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${SHARD_DATA_DIR}"

CMD=(
  python -u scripts/analyze_models/unified_resolution_eval.py
  --families graphcast residual_mamba
  --metrics weighted_allvars per_variable
  --eval-modes cold warm
  --n-eval-days "${N_EVAL_DAYS}"
  --lead-days 1 2 4
  --window-batch-size "${WINDOW_BATCH_SIZE}"
  --resolutions "${RES}"
  --output-data-dir "${SHARD_DATA_DIR}"
  --output-csv-name "mamba_wmse_vs_res_shard_res${RES}.csv"
  --warmup-steps "${WARMUP_STEPS}"
  --trunk-steps "${TRUNK_STEPS}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
