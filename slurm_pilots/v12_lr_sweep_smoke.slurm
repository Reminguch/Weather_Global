#!/usr/bin/env bash
# Quick LR sweep smoke for v12: try lr=3e-4 and lr=1e-3 for 500 steps each.
# Watch for: loss decreases faster? grad_norm spikes? any NaN/Inf?

#SBATCH --job-name=v12-lr-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v12_lr_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v12_lr_smoke_%j.err

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
OUT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast

for LR in 3e-4 1e-3; do
  echo "=========================================="
  echo "v12 smoke at lr=${LR}"
  echo "=========================================="
  ${PYTHON} -u ${SCRIPT} \
    --prepared-root "${PREPARED_ROOT}" \
    --residual-root "${RESIDUAL_ROOT}" \
    --ckpt-in "${CKPT}" \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --out-dir "${OUT_BASE}" \
    --run-name v12_smoke_lr${LR} \
    --resolution 1.0 --mesh-size 5 --hidden-size 128 \
    --input-duration 12h --target-steps 1 \
    --batch-size 1 \
    --max-steps 500 --checkpoint-every 250 \
    --lr ${LR} --weight-decay 1e-4 --seed 0 --precision bf16 \
    --sequential-segment-steps 32 --bptt-steps 8 \
    --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
    --temporal-layers 2
  echo
done
