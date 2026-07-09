#!/usr/bin/env bash
# v25 K=1 fresh — same v24 config (h=256 L=2 freeze enc/dec) but d_conv=2 (vs v24's 8).
#
# This is the third point in the d_conv ablation:
#   v22: d_conv=4, full-trainable, Mamba 851k                          (baseline)
#   v23 fresh: d_conv=8, full-trainable, Mamba 851k                    (bigger conv)
#   v24: d_conv=8, freeze enc/dec, Mamba 1.71M                          (capacity reshuffle)
#   v25: d_conv=2, freeze enc/dec, Mamba 1.71M                          ← THIS RUN
#
# Hypothesis: smaller conv kernel forces more local averaging → potentially
# MORE over-smoothing → MSE @ long lead may go up (winning per MSE-optimal-in-chaos)
# while spectrum is more degraded.
#SBATCH --job-name=v25-K1-fresh
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=36:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_fresh_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_fresh_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v24/train_mz_v24.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v25

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K1_fresh_h256_L2_d2_23k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 23000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 25 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 24 --bptt-steps 24 \
  --ar-tail-K 0 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 \
  --temporal-d-state 16 --temporal-d-conv 2 --temporal-dt-rank auto \
  --temporal-layers 2
