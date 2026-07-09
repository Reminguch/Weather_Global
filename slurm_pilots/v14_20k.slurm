#!/usr/bin/env bash
# v14 full 20K = v13 architecture + specialist heads.
# Smoke (7919010) confirmed: step 500 loss 0.747 (matches v13 0.748).
# Estimated wall clock: ~7h, same as v13.

#SBATCH --job-name=v14-20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v14_20k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v14_20k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v14/train_mz_v14.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v11_setup_res1
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v14_specialist_heads

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v14_20k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
