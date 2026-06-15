#!/bin/bash
#SBATCH --job-name=lead-compare-plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/two_model_lead_plot_%j.out
#SBATCH --error=logs/two_model_lead_plot_%j.err

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
OUTPUT_PREFIX="${OUTPUT_PREFIX:-two_model_lead_comparison}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-Candidate}"
BASELINE_LABEL="${BASELINE_LABEL:-Baseline}"
RATIO_LABEL="${RATIO_LABEL:-}"
LEAD_STEPS="${LEAD_STEPS:-4 8 16 24 32 40}"
EVAL_MODE="${EVAL_MODE:-warm}"

CMD=(
  python scripts/analyze_models/plot_two_model_lead_comparison.py
  --candidate-csv "${CANDIDATE_CSV}"
  --baseline-csv "${BASELINE_CSV}"
  --output-data-dir "${OUTPUT_DATA_DIR}"
  --output-image-dir "${OUTPUT_IMAGE_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --candidate-label "${CANDIDATE_LABEL}"
  --baseline-label "${BASELINE_LABEL}"
  --lead-steps ${LEAD_STEPS}
  --eval-mode "${EVAL_MODE}"
)
if [[ -n "${RATIO_LABEL}" ]]; then
  CMD+=(--ratio-label "${RATIO_LABEL}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
