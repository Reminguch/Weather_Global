#!/bin/bash
#SBATCH --job-name=res-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:40:00
#SBATCH --output=logs/resolution_eval_%A_%a.out
#SBATCH --error=logs/resolution_eval_%A_%a.err

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
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
METRICS="${METRICS:-weighted_allvars per_variable}"
EVAL_MODES="${EVAL_MODES:-cold warm}"
SHARD_DATA_DIR="${SHARD_DATA_DIR:-plots/analyze_models/data/resolution_eval/shards}"
SHARD_SPECS="${SHARD_SPECS:?SHARD_SPECS must be provided as a space-separated list of family:res values}"
CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS:-}"

read -r -a SHARD_ARRAY <<< "${SHARD_SPECS}"
SPEC="${SHARD_ARRAY[$SLURM_ARRAY_TASK_ID]}"
FAMILY="${SPEC%%:*}"
RES="${SPEC##*:}"

mkdir -p "${SHARD_DATA_DIR}"

CMD=(
  python -u scripts/analyze_models/unified_resolution_eval.py
  --families "${FAMILY}"
  --resolutions "${RES}"
  --n-eval-days "${N_EVAL_DAYS}"
  --window-batch-size "${WINDOW_BATCH_SIZE}"
  --warmup-steps "${WARMUP_STEPS}"
  --trunk-steps "${TRUNK_STEPS}"
  --output-data-dir "${SHARD_DATA_DIR}"
  --output-csv-name "resolution_eval_${FAMILY}_res${RES}.csv"
  --metrics ${METRICS}
  --eval-modes ${EVAL_MODES}
)
if [[ -n "${CHECKPOINT_ROOTS}" ]]; then
  CMD+=(--checkpoint-roots ${CHECKPOINT_ROOTS})
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
