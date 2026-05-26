#!/usr/bin/env bash
# Phase 2: v18 K=10 with segment=64, chunk=16. Resume from K=1 chunk-16
# ckpt at step 23000. Finetune 3000 steps -> step 26000.
#SBATCH --job-name=v20-K10
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v20_K10_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v20_K10_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/train_mz_v20.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v20
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_v2/K1_from_v15v2_20k/v13_residual_step23000.pkl

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K10_from_v19K1_23k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 23001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 26000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 18 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 64 --bptt-steps 16 \
  --ar-tail-K 10 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
