#!/usr/bin/env bash
#SBATCH --job-name=v15v2-diag-tp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:15:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_diag_tp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_diag_tp_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2

${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py \
  --ckpt "${CKPT_BASE}/v13_residual_step20000.pkl" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --n-anchors 8 \
  --var total_precipitation_6hr \
  --out-npz "${CKPT_BASE}/eval_results/diag_tp6h_step20000.npz"
