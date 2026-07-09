#!/usr/bin/env bash
# Control: run eval_baseline_only.py with the 1.0 deg GC_small ckpt on the
# 1.0 deg version of the same 32 anchors. Should reproduce v15's baseline RMSE
# numbers — if it does, eval_baseline_only.py is correct, and the 0.25 deg
# result we already have is trustworthy.

#SBATCH --job-name=gc-1deg-ctrl
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gc_1deg_ctrl_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gc_1deg_ctrl_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/eval_baseline_only.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_1deg_eval32_2022.zarr
ANCHORS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/eval_anchors_2022_seed0_n32_T.npy
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gc_1deg_baseline_eval
mkdir -p "${OUT}"

${PYTHON} -u ${SCRIPT} \
  --ckpt-in "${CKPT}" \
  --data-path "${DATA}" \
  --stats-dir "${STATS}" \
  --resolution 1.0 \
  --mesh-size 5 \
  --baseline-msg-steps 16 \
  --anchor-times-npy "${ANCHORS}" \
  --n-samples 32 \
  --sparse-stride-deg 18.0 \
  --out-json "${OUT}/gc_small_1deg_eval32_2022.json"
