#!/usr/bin/env bash
# Smoke test for eval_resmamba_rollout.py — verify script runs end-to-end on
# V-Bfix c=0.5 step40000 ckpt with tiny n_anchors=8 + warm-up eval protocol.
# Confirms: ckpt loading, build_residual_correction_predictor wiring, both
# predictor forward paths, closed-loop feedback computation, JSON output.
#SBATCH --job-name=eval-resmamba-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RM_CKPT=${ROOT}/results/resmamba_GCv1K3_vBfix/resmamba_GCv1K3_res2_K2_vBfix_clip05/ckpt_step40000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-23-vBfix-warmup-eval
mkdir -p ${OUT_DIR}

[ -f "${GCV1_K3}" ] || { echo "ERR: missing baseline ckpt"; exit 1; }
[ -f "${RM_CKPT}" ] || { echo "ERR: missing V-Bfix ckpt"; exit 1; }

$PYTHON -u scripts/training/eval_resmamba_rollout.py \
  --baseline-ckpt "${GCV1_K3}" \
  --residual-ckpt "${RM_CKPT}" \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --val-year 2022 \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-mp 6 --residual-mp 2 \
  --n-anchors 8 \
  --warmup-steps 24 \
  --target-steps 40 \
  --feedback-mode gated \
  --feedback-lambda 1.0 \
  --feedback-clip-mode tanh \
  --feedback-clip-c 0.5 \
  --seed 0 \
  --out-json ${OUT_DIR}/smoke_vBfix_c05_step40000.json

echo "smoke eval done — check ${OUT_DIR}/smoke_vBfix_c05_step40000.json"
