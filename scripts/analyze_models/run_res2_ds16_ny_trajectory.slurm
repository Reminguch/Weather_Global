#!/bin/bash
#SBATCH --job-name=res2-ds16-ny
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/res2_ds16_ny_%A_%a.out
#SBATCH --error=logs/res2_ds16_ny_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"

source scripts/graphcast_env.sh

LEAD_DAYS_ARRAY=(${LEAD_DAYS_LIST:-2 4 6})
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LEAD_DAY="${LEAD_DAYS_ARRAY[$TASK_INDEX]}"

python -u scripts/analyze_models/res2_ds16_ny_trajectory.py \
  --lead-days "${LEAD_DAY}" \
  --plot-mode "${NY_PLOT_MODE:-target_steps}" \
  --batch-size "${NY_BATCH_SIZE:-8}"
