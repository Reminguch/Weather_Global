#!/bin/bash
#SBATCH --job-name=small-mamba-plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/small_mamba_freeze_release_plot_%j.out
#SBATCH --error=logs/small_mamba_freeze_release_plot_%j.err

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

GC_MAMBA_CSV="${GC_MAMBA_CSV:-plots/analyze_models/data/resolution_eval/small_gc_mamba_freeze_release_warm/resolution_eval.csv}"
RESIDUAL_MAMBA_CSV="${RESIDUAL_MAMBA_CSV:-plots/analyze_models/data/resolution_eval/small_residual_mamba_freeze_release_warm/resolution_eval.csv}"
OUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR:-plots/analyze_models/images/resolution_eval/small_mamba_freeze_release_warm}"
PLOT_LEAD_STEPS="${PLOT_LEAD_STEPS:-}"

CMD=(
  python -u scripts/analyze_models/plot_small_mamba_freeze_release.py
  --gc-mamba-csv "${GC_MAMBA_CSV}"
  --residual-mamba-csv "${RESIDUAL_MAMBA_CSV}"
  --output-image-dir "${OUTPUT_IMAGE_DIR}"
)
if [[ -n "${PLOT_LEAD_STEPS}" ]]; then
  CMD+=(--lead-steps ${PLOT_LEAD_STEPS})
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
