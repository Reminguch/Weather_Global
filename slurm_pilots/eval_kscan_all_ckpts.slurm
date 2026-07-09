#!/usr/bin/env bash
# All-ckpts cold-eval for K-curriculum. Array indices 0..41 map to (K, step) pairs.
# 7 K (1,2,4,8,10,14,18) × 6 steps (2000,4000,6000,8000,10000,12000) = 42 evals.
# n_anchors=8 smoke for quick direction signal.
#SBATCH --job-name=eval-kscan-all
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_kscan_all_%A_%a.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_kscan_all_%A_%a.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

# Array index decoding: 7 K values × 6 step values.
K_LIST=(1 2 4 8 10 14 18)
STEP_LIST=(12000)
N_STEPS=1
IDX=${SLURM_ARRAY_TASK_ID}
K_IDX=$((IDX / N_STEPS))
STEP_IDX=$((IDX % N_STEPS))
K=${K_LIST[$K_IDX]}
STEP=${STEP_LIST[$STEP_IDX]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RM_CKPT=${ROOT}/results/resmamba_GCv1K3_openloop_Kscan/resmamba_GCv1K3_res2_K${K}_openloop_v22cfg/ckpt_step${STEP}.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-24-K-scan-all-ckpts
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/K${K}_step${STEP}.json

[ -f "${RM_CKPT}" ] || { echo "ERR: missing K=${K} step=${STEP} ckpt at ${RM_CKPT}"; exit 1; }
echo "=== K=${K} step=${STEP} idx=${IDX} ==="
echo "ckpt: ${RM_CKPT}"

$PYTHON -u scripts/training/eval_resmamba_rollout.py \
  --baseline-ckpt "${GCV1_K3}" \
  --residual-ckpt "${RM_CKPT}" \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --val-year 2022 \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-mp 6 --residual-mp 2 \
  --n-anchors 8 \
  --warmup-steps 0 \
  --target-steps 40 \
  --feedback-mode baseline \
  --seed 0 \
  --out-json ${OUT_JSON}

echo "K=${K} step=${STEP} done"
