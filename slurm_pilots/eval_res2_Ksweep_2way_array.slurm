#!/usr/bin/env bash
# res=2 K-sweep eval: GCv1 K=3 (baseline) vs gc_mamba K∈{1,2,4,8,12,16,20}.
# 32 anchors × (24 truth warmup + 40 closed-loop rollout).
# Same Mamba config as res=2 K-sweep training: postproc insert=1, d_inner=128 etc.
#SBATCH --job-name=eval-gcm-res2-Ksweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_gcm_res2_Ksweep_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_gcm_res2_Ksweep_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

K_LIST=(1 2 4 8 12 16 20)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
BASELINE_CKPT=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz

if [ ${K} -eq 1 ]; then
  GCMAMBA_CKPT=${ROOT}/results/gcmamba_v2/gcmamba_v2_GCv1K3_res2_K1_20k/ckpt_step19000.npz
  STEP=19000
else
  GCMAMBA_CKPT=${ROOT}/results/gcmamba_v2/gcmamba_v2_res2_K${K}_from_K1step19k/ckpt_step39000.npz
  STEP=39000
fi

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-16-gcmamba_v2
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/eval_res2_2way_K${K}_step${STEP}.json

echo "================================================"
echo "=== task=${SLURM_ARRAY_TASK_ID}  K=${K}  step=${STEP}  ckpt=${GCMAMBA_CKPT}"
echo "================================================"

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_2way.py \
  --baseline-ckpt "${BASELINE_CKPT}" \
  --gc-mamba-ckpt "${GCMAMBA_CKPT}" \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 6 \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-layers 1 --temporal-insert-count 1 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=${K} res=2 eval done — ${OUT_JSON}"
