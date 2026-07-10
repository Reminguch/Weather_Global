#!/usr/bin/env bash
# DIAG: run eval_v22_clean on SAME anchor as city_trace (val_idx=780 = 2022-07-15T00)
# with cold_full, K=22 SWA, zero-init. Print global mean/RMSE — will compare to
# city_trace's global diag on same anchor.
#SBATCH --job-name=eval-clean-diag
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_clean_diag_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_clean_diag_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT=${ROOT}/results/v22cl_r1_EMA/K22_ema.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-eval-clean-diag
mkdir -p ${OUT_DIR}

echo "=== eval_v22_clean force_idx=780 (2022-07-15T00) K=22 SWA cold_full ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --seed 0 \
  --force-idx 780 \
  --out-json "${OUT_DIR}/K22_SWA_cold_full_force780.json"
echo "Done"
