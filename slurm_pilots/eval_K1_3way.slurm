#!/usr/bin/env bash
# 3-way K=1 deployment eval: GCv1 K=3 baseline vs gc_mamba K=1 vs res_mamba K=1.
# All res=2 mesh=4 w=512.
# 32 anchors × (24 truth warmup + 40 closed-loop self-rollout).
#SBATCH --job-name=eval-K1-3way
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_3way_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_3way_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
GCMAMBA_K1=${ROOT}/results/gcmamba_v2/gcmamba_v2_GCv1K3_res2_K1_20k/ckpt_step20000.npz
RESMAMBA_K1=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K1_closed_20k/ckpt_step20000.npz

OUT_JSON=/home/lm8598/Weather_Global_experiments/results/2026-06-16-gcmamba_v2/eval_K1_3way.json

for p in "$GCV1_K3" "$GCMAMBA_K1" "$RESMAMBA_K1"; do
    [ -f "$p" ] && echo "✓ $p" || { echo "✗ MISSING: $p"; exit 1; }
done

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_3way.py \
  --gcv1-k3-ckpt "${GCV1_K3}" \
  --gc-mamba-ckpt "${GCMAMBA_K1}" \
  --res-mamba-ckpt "${RESMAMBA_K1}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-mp 6 --res-mamba-mp 2 \
  --gcm-temporal-d-inner 128 --gcm-temporal-d-state 16 --gcm-temporal-d-conv 4 \
  --gcm-temporal-layers 1 --gcm-temporal-insert-count 1 \
  --rm-temporal-d-inner 128 --rm-temporal-d-state 16 --rm-temporal-d-conv 4 \
  --rm-temporal-layers 1 --rm-temporal-insert-count 2 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "3-way eval done"
