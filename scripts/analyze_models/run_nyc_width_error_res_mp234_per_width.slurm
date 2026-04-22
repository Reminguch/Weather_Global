#!/bin/bash
# Submit once to run an array sweep over the res_sweep_mesh_mp234 checkpoints:
#   sbatch scripts/analyze_models/run_nyc_width_error_res_mp234_per_width.slurm
#
# Sweep:
#   (res, mp) pairs: (1, 4), (4, 3), (15, 2)
#   WIDTH_LIST=(128 256 512)
#   => 9 jobs total, one GPU per (res, mp, width).
#
# Each job writes a per-width CSV:
#   plots/analyze_models/data/mp234/nyc_width_error_res{res}_mp{mp}_w{width}.csv
#
# After all jobs finish, build mp234 MSE-vs-res plots from CSVs with:
#   python scripts/analyze_models/plot_nyc_width_error_mp234_vs_res.py \
#     --input-dir plots/analyze_models/data/mp234 \
#     --image-dir plots/analyze_models/images/mp234 \
#     --merged-data-dir plots/analyze_models/data/mp234 \
#     --metrics grid15_wmse

#SBATCH --job-name=gc-mp234-mae
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:30:00
# 3 (res, mp) pairs x 3 widths = 9 tasks.
#SBATCH --array=0-8
#SBATCH --output=logs/gc_mp234_width_mae_%A_%a.out
#SBATCH --error=logs/gc_mp234_width_mae_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs plots/analyze_models/data/mp234 plots/analyze_models/images/mp234

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RES_MP_PAIRS=("1:4" "4:3" "15:2")
WIDTH_LIST=(128 256 512)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_PAIR="${#RES_MP_PAIRS[@]}"
N_WIDTH="${#WIDTH_LIST[@]}"
N_TASKS=$((N_PAIR * N_WIDTH))
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${N_TASKS}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; expected 0..$((N_TASKS - 1))"
  exit 2
fi

# Decompose flat index: fastest-varying = width, then (res, mp) pair.
WIDTH_IDX=$((IDX % N_WIDTH))
PAIR_IDX=$((IDX / N_WIDTH))
PAIR="${RES_MP_PAIRS[${PAIR_IDX}]}"
RES="${PAIR%%:*}"
MP="${PAIR##*:}"
WIDTH="${WIDTH_LIST[${WIDTH_IDX}]}"

# Scale batch size inversely with model width so the A100 stays saturated.
# res=1 uses the full 181x360 grid; cap batch at 4 to avoid CPU OOM.
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
  --checkpoint-root artifacts/checkpoints/res_sweep_mesh_mp234
  --output-data-dir plots/analyze_models/data/mp234
)

echo "Running task ${IDX}/${N_TASKS}: res=${RES}, mp=${MP}, width=${WIDTH}, batch_size=${BATCH_SIZE}"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
