#!/bin/bash
#SBATCH --job-name=res-eval-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/resolution_eval_merge_%j.out
#SBATCH --error=logs/resolution_eval_merge_%j.err

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

SHARD_DATA_DIR="${SHARD_DATA_DIR:-plots/analyze_models/data/resolution_eval/shards}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-plots/analyze_models/data/resolution_eval}"
OUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR:-plots/analyze_models/images/resolution_eval}"
SHARD_SPECS="${SHARD_SPECS:?SHARD_SPECS must be provided as a space-separated list of family:res values}"
PLOT_PREFIX="${PLOT_PREFIX:-resolution_eval}"

CMD=(
  python -u scripts/analyze_models/merge_resolution_eval.py
  --input-data-dir "${SHARD_DATA_DIR}"
  --output-data-dir "${OUTPUT_DATA_DIR}"
  --output-image-dir "${OUTPUT_IMAGE_DIR}"
  --output-csv-name "resolution_eval.csv"
  --shard-glob "resolution_eval_*.csv"
  --plot-prefix "${PLOT_PREFIX}"
  --expected-shards ${SHARD_SPECS}
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
