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
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

CMD=(
  python -u scripts/analyze_models/mamba_wmse_vs_res.py
  --n-eval-days "${N_EVAL_DAYS}"
  --window-batch-size "${WINDOW_BATCH_SIZE}"
)

if [[ -n "${RESOLUTIONS}" ]]; then
  CMD+=(--resolutions ${RESOLUTIONS})
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
