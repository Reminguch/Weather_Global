#!/usr/bin/env bash
# Resume 7318998 (Simplified+K=2) from step 4500 to step 5000 (only 500 more
# K=2 steps since it was almost finished before the disk crash).

#SBATCH --job-name=mz-K2-simpl-rsm
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_K2_simpl_rsm_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_K2_simpl_rsm_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mz/train_full_mz.py
cd /home/lm8598/Weather_Global_experiments

DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_full_paperckpt_r1_in2_seg16_meshed_m5_h128_allvars_K2_5k

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_full_paperckpt_r1_in2_seg16_meshed_m5_h128_allvars_K2_5k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --a-log-init -0.1 \
  --max-steps 5000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${DIR}/mz_residual_step4500.pkl" \
  --resume-step 4500
