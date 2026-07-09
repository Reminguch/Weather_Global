#!/usr/bin/env bash
# v24 K=22 finetune — from v24 K=1 fresh ckpt step 23000.
#
# Apples-to-apples vs v22 K=22 + v23 K=22 fresh:
#   v22 K=22: full-trainable, Mamba 851k, d_conv=4, 23k K=1 (v15-inherited)
#             + 3k K=22
#   v23 K=22 fresh: full-trainable, Mamba 851k, d_conv=8, 23k K=1 fresh +
#             3k K=22 (job 8862997 chain)
#   v24 K=22: FROZEN encoder/decoder, Mamba 1.71M (2×), d_conv=8, 23k K=1
#             fresh + 3k K=22  ← THIS RUN
#
# Submitted as afterok dependency on v24 K=1 resume:
#   sbatch --dependency=afterok:<RESUME_JOBID> v24_K22_from_K1_h256_L2.slurm
#SBATCH --job-name=v24-K22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v24_K22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v24_K22_%j.err
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
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v24
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v24/K1_fresh_h256_L2_d8_23k/v13_residual_step23000.pkl

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: missing v24 K=1 fresh ckpt: ${RESUME_CKPT}" 1>&2
  echo "Wait for v24 K=1 fresh + resume chain to finish step 23000." 1>&2
  exit 1
fi

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K22_from_K1_h256_L2_d8_23k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 23001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 26000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 24 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 22 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 2
