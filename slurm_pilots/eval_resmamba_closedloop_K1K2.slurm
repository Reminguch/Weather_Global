#!/usr/bin/env bash
# Eval closed-loop trained res_mamba ckpts (K=1 step 20000, K=2 partial step 34000)
# with 24-step truth warmup + 40-step closed-loop AR (warm_full deployment mode).
# Compare against GCv1 K=3 baseline at res=2.
#SBATCH --job-name=eval-resmamba-closedloop
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_closedloop_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_closedloop_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
# K=1 (closed-loop, fully trained)
RM_K1=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K1_closed_20k/ckpt_step20000.npz
# K=2 partial (closed-loop, 14k of 20k steps when cancelled)
RM_K2=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K2_from_K1step20k/ckpt_step34000.npz

CKPTS=("$RM_K1" "$RM_K2")
TAGS=("K1_step20000" "K2partial_step34000")
RM=${CKPTS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-20-resmamba-closedloop-eval
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/eval_resmamba_${TAG}.json

# Need a dummy gc_mamba ckpt to satisfy the 3way script (we'll ignore that branch)
GCM_DUMMY=${ROOT}/results/gcmamba_v2/gcmamba_v2_GCv1K3_res2_K1_20k/ckpt_step19000.npz

echo "================================================"
echo "=== task=${SLURM_ARRAY_TASK_ID}  tag=${TAG}"
echo "=== res_mamba ckpt=${RM}"
echo "================================================"

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_3way.py \
  --gcv1-k3-ckpt "${GCV1_K3}" \
  --gc-mamba-ckpt "${GCM_DUMMY}" \
  --res-mamba-ckpt "${RM}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-mp 6 --res-mamba-mp 2 \
  --gcm-temporal-d-inner 128 --gcm-temporal-d-state 16 --gcm-temporal-d-conv 4 \
  --gcm-temporal-layers 1 --gcm-temporal-insert-count 1 \
  --rm-temporal-d-inner 128 --rm-temporal-d-state 16 --rm-temporal-d-conv 4 \
  --rm-temporal-layers 1 --rm-temporal-insert-count 2 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "eval ${TAG} done"
