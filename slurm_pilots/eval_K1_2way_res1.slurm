#!/usr/bin/env bash
# 2-way eval at res=1: DeepMind GraphCast small (baseline) vs gc_mamba K=1 (the postproc 20k run).
# 32 anchors × (24 truth warmup + 40 closed-loop rollout).
#SBATCH --job-name=eval-K1-2way-res1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_2way_res1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_2way_res1_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
BASELINE_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
GCMAMBA_CKPT="${ROOT}/results/gcmamba_v2/gcmamba_v2_postproc_K1_2k/ckpt_step20000.npz"
OUT_JSON=/home/lm8598/Weather_Global_experiments/results/2026-06-16-gcmamba_v2/eval_res1_2way.json

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_2way.py \
  --baseline-ckpt "${BASELINE_CKPT}" \
  --gc-mamba-ckpt "${GCMAMBA_CKPT}" \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-layers 1 --temporal-insert-count 1 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "2-way res=1 eval done"
