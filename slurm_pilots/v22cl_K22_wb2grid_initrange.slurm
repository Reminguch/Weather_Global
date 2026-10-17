#!/usr/bin/env bash
# K=22 GraphCast-Mamba (archived v22cl, per-K SWA) + its GraphCast-small baseline,
# cold start, zero memory, full-prediction feedback, 40 x 6 h, scored on the
# 1.5-deg WB2 grid (conservative regrid of forecast AND truth) vs the ERA5 in DATA.
#
# Usage (DATA must be a continuous 6-hourly 1-deg, 13-level zarr covering
# INIT_START-6h .. INIT_END+240h; values must not contain commas):
#   sbatch -A <account> --export=ALL,DATA=/path/era5_1deg_2023.zarr,INIT_START=2023-01-01T00,INIT_END=2023-12-30T00,EVERY_H=72,OUT=/path/out/K22_2023 \
#       slurm_pilots/v22cl_K22_wb2grid_initrange.slurm
# Writes ${OUT}.json (aggregated RMSE/MAE per variable/channel/lead) and
# ${OUT}_per_init.npz (per-init lat-weighted MSE, shape (n_init, 40, n_channel)).
#SBATCH --job-name=v22cl-K22-initrange
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=v22cl_K22_initrange_%j.out
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=${PYTHON:-/scratch/gpfs/DABANIN/lm8598/home_offload/Weather_Global_experiments_conda/envs/graphcast311/bin/python}
LM=/scratch/gpfs/DABANIN/lm8598/Weather_Global
CKPT=${CKPT:-${LM}/results/v22cl_r1_EMA/K22_ema.pkl}
DM_CKPT="${LM}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p "$(dirname "${OUT}")"
echo "DATA=${DATA}  inits=${INIT_START}..${INIT_END}/${EVERY_H}h  host=$(hostname)"
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean_wb2grid.py \
  --ckpt "${CKPT}" --ckpt-in "${DM_CKPT}" \
  --data-path "${DATA}" \
  --stats-dir ${LM}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --seed 0 \
  --init-range "${INIT_START}" "${INIT_END}" "${EVERY_H}" \
  --out-json "${OUT}.json" --per-init-npz "${OUT}_per_init.npz"
echo done
