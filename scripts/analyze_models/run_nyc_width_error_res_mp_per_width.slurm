#!/bin/bash
# Submit once to run an array sweep over (res, mp, width):
#   sbatch scripts/analyze_models/run_nyc_width_error_res_mp_per_width.slurm
#
# Default sweep:
#   RES_LIST=(1 2 4 6 9 12 15)
#   MP_LIST=(1 2)
#   WIDTH_LIST=(128 256 512 1024)
#   => 56 jobs total, one GPU per (res, mp, width).
#
# Splitting by width avoids repeated JAX JIT recompilation across model sizes
# on the same GPU. Each job JIT-compiles exactly one model and runs inference
# at a batch size scaled to keep the A100 saturated:
#   w=128 → batch=64, w=256 → batch=32, w=512 → batch=16, w=1024 → batch=8
#
# Each job writes a per-width CSV:
#   plots/analyze_models/data/nyc_width_error_res{res}_mp{mp}_w{width}.csv
#
# After all jobs finish, build MAE-vs-res plots from CSVs with:
#   python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py --per-width-only
# which writes:
#   plots/analyze_models/images/nyc_mae_vs_res_mp{mp}_lead{lead_days}d.png

#SBATCH --job-name=gc-width-mae
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:30:00
# 7 res values x 2 mp values x 4 widths = 56 tasks.
#SBATCH --array=0-55
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
WIDTH_LIST=(128 256 512 1024)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_RES="${#RES_LIST[@]}"
N_MP="${#MP_LIST[@]}"
N_WIDTH="${#WIDTH_LIST[@]}"
N_TASKS=$((N_RES * N_MP * N_WIDTH))
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${N_TASKS}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; expected 0..$((N_TASKS - 1))"
  exit 2
fi

# Decompose flat index: fastest-varying = width, then mp, then res.
WIDTH_IDX=$((IDX % N_WIDTH))
MP_IDX=$(( (IDX / N_WIDTH) % N_MP ))
RES_IDX=$((IDX / (N_WIDTH * N_MP)))
RES="${RES_LIST[${RES_IDX}]}"
MP="${MP_LIST[${MP_IDX}]}"
WIDTH="${WIDTH_LIST[${WIDTH_IDX}]}"

# Scale batch size inversely with model width so the A100 stays saturated.
# w=128 → batch=64, w=256 → batch=32, w=512 → batch=16, w=1024 → batch=8.
# res=1 uses the full 181×360 grid; cap batch at 4 to avoid CPU OOM.
BATCH_SIZE=$((8 * 1024 / WIDTH))
if [[ "${RES}" -eq 1 ]]; then
  BATCH_SIZE=$(( BATCH_SIZE > 4 ? 4 : BATCH_SIZE ))
fi

CMD=(
  python -u scripts/analyze_models/nyc_width_error_by_res_mp.py
  --res "${RES}"
  --mp "${MP}"
  --width "${WIDTH}"
  --batch-size "${BATCH_SIZE}"
)

echo "Running task ${IDX}/${N_TASKS}: res=${RES}, mp=${MP}, width=${WIDTH}, batch_size=${BATCH_SIZE}"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
