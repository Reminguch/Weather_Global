#!/usr/bin/env bash
# Memory ablation: K=18 step20000 (res=2), 4 modes.
# Tasks:
#   0 normal_zeroinit    (none, W=0)               — reference (matches cold_bp+zero)
#   1 reset_ssm_every    (reset_ssm_every_step, W=0)
#   2 reset_all_every    (reset_all_every_step, W=0)
#   3 normal_W24         (none, W=24)              — reference for stale
#   4 stale_after_W24    (stale_after_warmup, W=24)
#SBATCH --job-name=v22-ablate-K18
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_ablate_K18_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_ablate_K18_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate
mkdir -p ${OUT_DIR}

CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl

MODES=(none reset_ssm_every_step reset_all_every_step none stale_after_warmup)
WARMUPS=(0   0                   0                   24    24)
TAGS=(normal_W0 reset_ssm_W0 reset_all_W0 normal_W24 stale_W24)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
W=${WARMUPS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
OUT_JSON=${OUT_DIR}/K18_step20000_ablate_${TAG}.json

echo "================================================"
echo "=== K=18 step20000 ablation: ${TAG}"
echo "===   mode=${MODE}  W=${W}"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_ablate.py \
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
  --target-steps 40 \
  --warmup-steps ${W} \
  --n-samples 32 \
  --seed 0 \
  --ablation-mode ${MODE} \
  --out-json "${OUT_JSON}"

echo "K=18 ablation ${TAG} done"
