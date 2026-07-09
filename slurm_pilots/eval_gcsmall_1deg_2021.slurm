#!/usr/bin/env bash
# Control: GC_small (1.0 deg, 13 levels) on the conservative-coarsened 1.0 deg
# version of the 32 anchors of 2021. Same metrics as the headline eval.

#SBATCH --job-name=gc-small-1deg-2021
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gc_small_2021_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gc_small_2021_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/eval_baseline_only.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_1deg_eval32_2021.zarr
ANCHORS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/eval_anchors_2021_seed0_n32_T.npy
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gc_small_1deg_baseline_eval_2021
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
  --sparse-levels 50 100 150 200 250 300 400 500 600 700 850 925 1000 \
  --out-json "${OUT}/gc_small_1deg_eval32_2021.json"
