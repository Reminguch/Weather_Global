#!/bin/bash
#SBATCH --job-name=res1-lead-plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/res1_lead_plot_%j.out
#SBATCH --error=logs/res1_lead_plot_%j.err

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

DATA_DIR="${DATA_DIR:-plots/analyze_models/data/resolution_eval/res1_official_bptt16_k_sweep_warm_res1grid_k1_40}"
IMAGE_DIR="${IMAGE_DIR:-plots/analyze_models/images/resolution_eval/res1_official_bptt16_k_sweep_warm_res1grid_k1_40}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-lead_curves_res1_existing_partial}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-res1_bptt16_k_sweep}"
MAMBA_GLOB="${MAMBA_GLOB:-shards/resolution_eval_*mamba_res1_warm_res1grid_k1_40*.csv}"
MAMBA_CSVS="${MAMBA_CSVS:-}"
BASELINE_WEIGHTED_DIR="${BASELINE_WEIGHTED_DIR:-plots/analyze_models/data/resolution_eval/deepmind_graphcast_small_res1_cold/shards}"
BASELINE_RMSE_CSV="${BASELINE_RMSE_CSV:-plots/analyze_models/data/resolution_eval/res1_official_bptt16_k_sweep_warm_res1grid_k1_40/shards/resolution_eval_graphcast_res1_deepmind_small_rmse_k_2m_temperature_k1_40.csv}"
ALLOW_TEACHER_FORCED_RESIDUAL="${ALLOW_TEACHER_FORCED_RESIDUAL:-0}"

CMD=(
  python scripts/analyze_models/plot_res1_bptt16_k_sweep_lead_curves.py
  --data-dir "${DATA_DIR}"
  --image-dir "${IMAGE_DIR}"
  --output-subdir "${OUTPUT_SUBDIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --mamba-glob "${MAMBA_GLOB}"
  --baseline-weighted-dir "${BASELINE_WEIGHTED_DIR}"
)

if [[ -n "${BASELINE_RMSE_CSV}" ]]; then
  CMD+=(--baseline-rmse-csv "${BASELINE_RMSE_CSV}")
fi
if [[ -n "${MAMBA_CSVS}" ]]; then
  read -r -a MAMBA_CSV_ARRAY <<< "${MAMBA_CSVS}"
  CMD+=(--mamba-csvs "${MAMBA_CSV_ARRAY[@]}")
fi
if [[ "${ALLOW_TEACHER_FORCED_RESIDUAL}" == "1" ]]; then
  CMD+=(--allow-teacher-forced-residual)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
