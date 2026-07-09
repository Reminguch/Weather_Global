#!/usr/bin/env bash
# v12 with v3-equivalent Mamba capacity:
#   d_inner=256 (vs v12's 128), dt_rank=256 (vs v12's auto=8), layers=2 (same)
# Total Mamba params: ~486K (4x v12's 117K)
# Tests whether v12's eval gap vs v3 is due to Mamba under-capacity.

#SBATCH --job-name=v12-bigmamba
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v12_bigmamba_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v12_bigmamba_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v12/train_mz_v12.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v11_setup_res1
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v12_bigmamba_20k \
  --resolution 1.0 --mesh-size 5 --hidden-size 128 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --temporal-d-inner 256 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank 256 \
  --temporal-layers 2
