#!/usr/bin/env bash
# v24-CLOSED K=22 with SMALL Mamba (v22-equivalent size):
#   hidden=128 (vs v24's 512), layers=2 (vs 3), d-conv=4 (vs 8).
# This matches v22 production Mamba size exactly. Direct apples-to-apples
# vs v22 K=22 — only difference is closed-loop AR feedback in training.
#
# Memory: v22 K=22 open-loop fits easily at this size; v24-closed K=22
# adds extra BPTT gradient memory but with 4× smaller Mamba should fit.
#
# After this finishes (~24h), compare RMSE + bias against v22 K=22 same Mamba
# size to isolate the closed-loop training effect.
#SBATCH --job-name=v24-closed-K22-small
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=30:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v24_closed_K22_small_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v24_closed_K22_small_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v24/train_mz_v24_closed.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v24_closed/K22_smallMamba_v22match_20k

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v24_closed_K22_smallMamba_v22match_20k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 24 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 500 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 22 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
