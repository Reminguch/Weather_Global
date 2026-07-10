#!/usr/bin/env bash
# DIAG: v22 paper K=22 open-loop ckpt in COLD_BP mode at anchor Jul 15 2022.
# Print global mean/RMSE per lead to verify v22 paper cold_bp is not silently broken.
#SBATCH --job-name=v22paper-diag
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22paper_diag_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22paper_diag_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT=${ROOT}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22paper-diag
mkdir -p ${OUT_DIR}

# --- CITY TRACE: v22 paper cold_bp at Jul 15 (matches v22 paper reference numbers) ---
echo "=== v22 paper K=22 open-loop  cold_bp  anchor=2022-07-15 ==="
$PYTHON -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v23/eval_city_trace.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --target-steps 40 \
  --anchor-date 2022-07-15T00:00:00 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --feedback-mode cold_bp \
  --out-json "${OUT_DIR}/v22paper_K22_cold_bp_Jul15.json"
echo "=== done city_trace ==="

# --- GLOBAL EVAL: eval_v22_clean cold_bp on SAME anchor idx=780 ---
echo "=== eval_v22_clean force_idx=780  cold_bp ==="
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
  --eval-mode cold_bp --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --seed 0 \
  --force-idx 780 \
  --out-json "${OUT_DIR}/v22paper_K22_cold_bp_force780_global.json"
echo "Done"
