#!/usr/bin/env bash
# Diagnostic smoke: cold eval on K=2 open-loop step20000 ckpt.
# This is the cleanest sanity check: open-loop trained ckpt, open-loop eval,
# no closed-loop complications. If this gives sensible numbers (resmamba ~=
# baseline at lead 1 then growing slowly), pipeline is correct. If it produces
# garbage like V-Bfix smoke, the pipeline has a remaining bug.
#SBATCH --job-name=eval-resmamba-smoke-K2ol
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_smoke_K2ol_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_resmamba_smoke_K2ol_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RM_CKPT=${ROOT}/results/resmamba_GCv1K3_openloop_Kscan/resmamba_GCv1K3_res2_K2_openloop/ckpt_step20000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-23-K-curriculum-cold-eval
mkdir -p ${OUT_DIR}

[ -f "${RM_CKPT}" ] || { echo "ERR: missing K=2 ckpt"; exit 1; }

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
  --out-json ${OUT_DIR}/smoke_K2_openloop_step20000.json

echo "K=2 open-loop cold eval smoke done"
