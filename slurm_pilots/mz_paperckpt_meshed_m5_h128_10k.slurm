#!/usr/bin/env bash
# Paper-ckpt + MZ-MESHED mesh=5 (10242 nodes), hidden=128, 10K-step training.
# Long run to see whether the +0.30% at 400 steps (per-grid h=128) translates
# into something much larger once the Mamba really has time to fit the
# paper-grade baseline's residual structure.
#
# Uses qos=gpu-medium (up to 72h). Checkpoints every 500 steps (20 total) so
# we can pick mid-training snapshots if the curve saturates or regresses.

#SBATCH --job-name=mz-p-m5h128-10k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=48:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_paperckpt_m5h128_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_paperckpt_m5h128_10k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_mz_residual_memory.py
cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_paperckpt_r1_in2_seg16_meshed_m5_h128_10k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --max-steps 10000 --eval-every 500 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 500 \
  --baseline-precision fp32 --precision bf16
