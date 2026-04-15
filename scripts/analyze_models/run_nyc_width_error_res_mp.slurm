#!/bin/bash
# Submit once to run an array sweep over (res, mp):
#   sbatch scripts/analyze_models/run_nyc_width_error_res_mp.slurm
#
# Default sweep:
#   RES_LIST=(1 2 4 6 9 12 15)
#   MP_LIST=(1 2)
#   => 14 jobs total.
#
# This produces per-(res,mp) metric CSVs.
# For each run/width, the analysis script uses ckpt_best.npz.
#   plots/analyze_models/data/nyc_width_error_res{res}_mp{mp}.csv
#
# After all jobs finish, build MAE-vs-res plots from CSVs with:
#   python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py
# which writes:
#   plots/analyze_models/images/nyc_mae_vs_res_mp{mp}_lead{lead_days}d.png

#SBATCH --job-name=gc-width-mae
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=4:00:00
# 7 res values x 2 mp values = 14 tasks.
#SBATCH --array=0-13
#SBATCH --output=logs/gc_width_mae_%A_%a.out
#SBATCH --error=logs/gc_width_mae_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RES_LIST=(1 2 4 6 9 12 15)
MP_LIST=(1 2)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_RES="${#RES_LIST[@]}"
N_MP="${#MP_LIST[@]}"
N_TASKS=$((N_RES * N_MP))
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${N_TASKS}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; expected 0..$((N_TASKS - 1))"
  exit 2
fi

RES_IDX=$((IDX / N_MP))
MP_IDX=$((IDX % N_MP))
RES="${RES_LIST[${RES_IDX}]}"
MP="${MP_LIST[${MP_IDX}]}"

CMD=(
  python -u scripts/analyze_models/nyc_width_error_by_res_mp.py
  --res "${RES}"
  --mp "${MP}"
)

echo "Running task ${IDX}/${N_TASKS}: res=${RES}, mp=${MP}"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
