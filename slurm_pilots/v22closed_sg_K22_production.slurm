#!/usr/bin/env bash
# v22closed _SG K=22 production: full-horizon closed-loop training with
# stop_gradient through AR feedback. 20k steps fresh from DeepMind init.
# v22-apples-to-apples Mamba size: hidden=128 layers=2 d_conv=4.
#
# Why _SG: full-BPTT closed-loop would backprop through residual_pred chain
# (22 steps), much higher memory + risk of unstable gradient. _SG preserves
# closed-loop FORWARD distribution (matches eval setup we'll run) but blocks
# the gradient through prior steps' residual. Stable per the short smoke
# (job 9164389, 500 steps: bias bounded, grad_norm bounded, no NaN).
#
# Expected: step_time ~4-5s after compile → ~24h walltime.
# After done: eval with eval_v24_closed.py (warmup-steps 0 = cold_full,
# warmup-steps 24 = warm_full) and compare RMSE + global mean bias to v22.
#SBATCH --job-name=v22closed-sg-K22-prod
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=30:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22closed_sg_K22_prod_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22closed_sg_K22_prod_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v22closed/train_mz_v22closed.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K22_sg_fresh_20k

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v22closed_sg_K22_fresh_20k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 500 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 22 \
  --feedback-stop-gradient \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
