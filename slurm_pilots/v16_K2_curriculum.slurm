#!/usr/bin/env bash
# K-curriculum K=4 stage: resume from v15 v2 step 20000 (truth-anchored bptt=8)
# and train v16 with --ar-tail-K 2. Anchors 1..4 use REAL ERA5 inputs (truth
# zone), anchors 5..8 use closed-loop AR. Mamba state grad still flows 8 steps.
# Train 3000 steps to step 23000.

#SBATCH --job-name=v16-K2-curric
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v16_K2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v16_K2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v16/train_mz_v16.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v16_K_curriculum
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2/v13_residual_step20000.pkl

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K2_from_v15v2_20k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 20001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 23000 --checkpoint-every 500 \
  --lr 5e-5 --weight-decay 1e-4 --seed 1 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --ar-tail-K 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
