#!/usr/bin/env bash
# Warm 2022 rollout evaluation for the 50k standard-GraphCast AR-k12 control.
#SBATCH --job-name=gc-r2-gc500k-k12-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gc_r2_gc500k_k12_eval_%j.out
#SBATCH --error=logs/gc_r2_gc500k_k12_eval_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_disable_hlo_passes=constant_folding"
source scripts/graphcast_env.sh

CKPT="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps/ckpt_best.npz"
OUT_DIR="plots/analyze_models/data/resolution_eval/res2_vanilla_gc_ar_k12_from_gc500k_50k_warm_leads1_9d"
OUT_CSV="resolution_eval_graphcast_res2_ar_k12_from_gc500k.csv"
[[ -f "${CKPT}" ]] || { echo "Missing best checkpoint: ${CKPT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

python -u scripts/analyze_models/unified_resolution_eval.py \
  --families graphcast --resolutions 2 --data-source prepared_array \
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir data/graphcast/graphcast/stats --n-eval-days 365 --window-batch-size 1 \
  --warmup-steps 24 --trunk-steps 32 --prepared-load-workers 2 --prepared-stream-block-steps 32 \
  --metric-grid-resolution 2 --output-data-dir "${OUT_DIR}" --output-csv-name "${OUT_CSV}" \
  --metrics weighted_allvars rmse_k --eval-modes warm --lead-aggregation endpoint \
  --lead-steps 4 8 12 16 20 24 28 32 36 --checkpoint-paths "${CKPT}"

echo "Completed: ${OUT_DIR}/${OUT_CSV}"
