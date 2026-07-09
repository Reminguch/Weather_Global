#!/usr/bin/env bash
# v24-CLOSED K=22 production train: bptt=24, ar-tail-K=22 (22 closed-loop
# AR anchors per BPTT chunk). 20k steps fresh from DeepMind init.
#
# Will train Mamba residual to MAINTAIN stable closed-loop AR rollouts —
# unlike v22 open-loop which only learned per-step refinement on a baseline
# self-rollout trajectory.
#
# Expected:
#   - step_time ~4-5s after JIT compile (= ~24h for 20k steps)
#   - bias_2mt, bias_u10, bias_mslp logged every step → watch for monotone
#     drift over training. If they stay near zero / don't grow, closed-loop
#     training is working as hypothesized.
#   - Checkpoint every 1000 steps so we can resume / probe mid-training.
#
# After this finishes, run:
#   $PYTHON scripts/training/full_mamba_v24/eval_v24_closed.py \
#     --ckpt $OUT/v13_residual_step20000.pkl \
#     --warmup-steps 0 \
#     --out-json results/v24_closed_eval/K22_cold_full.json
# and compare RMSE to v22's open-loop K=22 cold_bp / warm_bp.
#SBATCH --job-name=v24-closed-K22-prod
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=30:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v24_closed_K22_prod_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v24_closed_K22_prod_%j.err
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
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v24_closed/K22_fresh_20k

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v24_closed_K22_fresh_20k \
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
  --temporal-hidden-size 512 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 3
