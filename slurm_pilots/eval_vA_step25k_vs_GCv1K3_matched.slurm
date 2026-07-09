#!/usr/bin/env bash
# Eval 1 (matched diagnostic): V-A K=2 step25000 vs GCv1 K=3 baseline (= frozen baseline used during V-A training).
# Question: does the residual head learn ANY skill on its own training distribution?
# (Different question than vs GCv1 K=2 which is a different baseline.)
#SBATCH --job-name=eval-vA-step25k-vs-GCv1K3
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_vA_step25k_vs_GCv1K3_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_vA_step25k_vs_GCv1K3_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
# GCv1 K=3 = the frozen baseline used during V-A K=2 res=2 training
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
# V-A K=2 res=2 step 25000 (λ=0.3 plateau, eval_rotating=2.46 in train log)
RM_VA=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K2_vA_smoke/ckpt_step25000.npz
# Dummy gc_mamba (eval_K1_3way needs all 3 args; we only care about res_mamba branch)
GCM_DUMMY=${ROOT}/results/gcmamba_v2/gcmamba_v2_GCv1K3_res2_K1_20k/ckpt_step19000.npz

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-22-vA-K2-eval
mkdir -p ${OUT_DIR}

for p in "$GCV1_K3" "$RM_VA" "$GCM_DUMMY"; do
  [ -f "$p" ] && echo "✓ $p" || { echo "✗ MISSING: $p"; exit 1; }
done

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_3way.py \
  --gcv1-k3-ckpt "${GCV1_K3}" \
  --gc-mamba-ckpt "${GCM_DUMMY}" \
  --res-mamba-ckpt "${RM_VA}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-mp 6 --res-mamba-mp 2 \
  --gcm-temporal-d-inner 128 --gcm-temporal-d-state 16 --gcm-temporal-d-conv 4 \
  --gcm-temporal-layers 1 --gcm-temporal-insert-count 1 \
  --rm-temporal-d-inner 128 --rm-temporal-d-state 16 --rm-temporal-d-conv 4 \
  --rm-temporal-layers 1 --rm-temporal-insert-count 2 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json ${OUT_DIR}/eval_vA_K2_step25000_vs_GCv1K3_matched.json

echo "eval V-A step25000 vs MATCHED GCv1 K=3 done"
