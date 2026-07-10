#!/usr/bin/env bash
# Warmup sweep on K=18 step 20000 — truth-warmup from h=0 protocol
# Sweep: W ∈ {4, 8, 16, 24, 40} steps (24h, 48h, 96h, 144h, 240h truth warmup)
# Scored mode: warm_bp (open-loop) and warm_full (closed-loop)
# All starting from h=0 (--residual-state-init zero)
#SBATCH --job-name=v22-warmup-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-9%6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_warmup_sweep_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_warmup_sweep_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba
mkdir -p ${OUT_DIR}

# Tasks: (W_steps, mode)  → W*6h truth warmup + scored {cold_bp | cold_full}
W_ARR=(   4    4     8    8     16   16    24   24    40   40)
MODE_ARR=(bp full  bp  full   bp  full   bp  full   bp  full)
W=${W_ARR[$SLURM_ARRAY_TASK_ID]}
MODE=${MODE_ARR[$SLURM_ARRAY_TASK_ID]}
HOURS=$((W * 6))

CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
OUT_JSON=${OUT_DIR}/K18_step20000_truthwarm_zero_${HOURS}h_${MODE}.json

echo "================================================"
echo "=== K=18 step20000 — truthwarm zero, W=${W} steps (${HOURS}h), score=warm_${MODE}"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} \
  --ckpt-in ${GCV1_K3} \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode warm_${MODE} \
  --residual-state-init zero \
  --target-steps 40 \
  --warmup-steps ${W} \
  --n-samples 32 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=18 step20000 W=${W}h ${MODE} done"
