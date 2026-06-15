#!/bin/bash
#SBATCH --job-name=res2-ds16-plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/res2_ds16_merge_plot_%j.out
#SBATCH --error=logs/res2_ds16_merge_plot_%j.err

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

EXPERIMENT="${EXPERIMENT:-res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d}"
DATA_DIR="${DATA_DIR:-plots/analyze_models/data/resolution_eval/${EXPERIMENT}}"
SHARD_DIR="${SHARD_DIR:-${DATA_DIR}/shards}"
IMAGE_DIR="${IMAGE_DIR:-plots/analyze_models/images/resolution_eval/${EXPERIMENT}}"
PLOT_PREFIX="${PLOT_PREFIX:-${EXPERIMENT}}"
LEAD_STEPS="${LEAD_STEPS:-4 8 12 16 20 24 28 32 36}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-graphcast:2 gc_mamba:2}"

python -u scripts/analyze_models/merge_resolution_eval.py \
  --input-data-dir "${SHARD_DIR}" \
  --output-data-dir "${DATA_DIR}" \
  --output-image-dir "${IMAGE_DIR}/default_merge" \
  --output-csv-name resolution_eval.csv \
  --shard-glob "resolution_eval_*.csv" \
  --plot-prefix "${PLOT_PREFIX}" \
  --expected-shards ${EXPECTED_SHARDS} \
  --resolutions 2 \
  --lead-steps ${LEAD_STEPS}

python -u scripts/analyze_models/plot_res2_ds16_gc_mamba_lead_curves.py \
  --input-csv "${DATA_DIR}/resolution_eval.csv" \
  --output-data-dir "${DATA_DIR}" \
  --output-image-dir "${IMAGE_DIR}" \
  --output-prefix "${PLOT_PREFIX}"
