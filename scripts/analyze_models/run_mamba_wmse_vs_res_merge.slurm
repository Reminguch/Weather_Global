#!/bin/bash
#SBATCH --job-name=mamba-wmse-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mamba_wmse_merge_%j.out
#SBATCH --error=logs/mamba_wmse_merge_%j.err

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

SHARD_DATA_DIR="${SHARD_DATA_DIR:-plots/analyze_models/data/mamba_res/shards}"
RESOLUTIONS="${RESOLUTIONS:?RESOLUTIONS must be provided as a space-separated list}"

CMD=(
  python -u scripts/analyze_models/merge_mamba_wmse_vs_res.py
  --input-data-dir "${SHARD_DATA_DIR}"
  --resolutions ${RESOLUTIONS}
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
