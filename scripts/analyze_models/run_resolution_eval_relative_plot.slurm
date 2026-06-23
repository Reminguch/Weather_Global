#!/bin/bash
#SBATCH --job-name=res-eval-relplot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/resolution_eval_relative_plot_%j.out
#SBATCH --error=logs/resolution_eval_relative_plot_%j.err

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

CANDIDATE_CSV="${CANDIDATE_CSV:?CANDIDATE_CSV must be set}"
BASELINE_CSV="${BASELINE_CSV:?BASELINE_CSV must be set}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:?OUTPUT_DATA_DIR must be set}"
OUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR:?OUTPUT_IMAGE_DIR must be set}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-relative_to_baseline}"
CANDIDATE_EVAL_MODE="${CANDIDATE_EVAL_MODE:-warm}"
BASELINE_EVAL_MODE="${BASELINE_EVAL_MODE:-cold}"
METRIC_KIND="${METRIC_KIND:-rmse_k}"
VARIABLE="${VARIABLE:-2m_temperature}"
LEAD_STEPS="${LEAD_STEPS:-4 8 16 24 32 40}"
BASELINE_LABEL="${BASELINE_LABEL:-GC_small}"
TITLE="${TITLE:-}"

CMD=(
  python -u scripts/analyze_models/plot_resolution_eval_relative_to_baseline.py
  --candidate-csv "${CANDIDATE_CSV}"
  --baseline-csv "${BASELINE_CSV}"
  --output-data-dir "${OUTPUT_DATA_DIR}"
  --output-image-dir "${OUTPUT_IMAGE_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --candidate-eval-mode "${CANDIDATE_EVAL_MODE}"
  --baseline-eval-mode "${BASELINE_EVAL_MODE}"
  --metric-kind "${METRIC_KIND}"
  --variable "${VARIABLE}"
  --lead-steps ${LEAD_STEPS}
  --baseline-label "${BASELINE_LABEL}"
)
if [[ -n "${TITLE}" ]]; then
  CMD+=(--title "${TITLE}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
