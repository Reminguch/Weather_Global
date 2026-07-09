#!/usr/bin/env bash
# Isolation Run 4: processor msg-passing remat'd ALONE; Mamba runs OUTSIDE
# the remat boundary. If Run 3 FAIL but Run 4 PASS, then "Mamba-inside-
# remat" is the bug (not stateful Mamba per se).
#SBATCH --job-name=gcm-v1-verify-r4
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_verify_r4_proc_temporal_outside_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_verify_r4_proc_temporal_outside_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

$PYTHON -u scripts/training/gcmamba_v1/verify_remat_equivalence.py \
  --remat-processor-steps \
  --temporal-hidden-size 128 --temporal-d-inner 64 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-layers 1 \
  --mamba-noise-scale 1e-2 --seed 0
echo "verify Run 4 (SAFE Mode-2: processor remat, Mamba outside) done"
