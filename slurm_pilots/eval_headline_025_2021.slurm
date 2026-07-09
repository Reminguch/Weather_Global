#!/usr/bin/env bash
# Eval headline GraphCast (0.25 deg, 37 levels, pure ERA5 1979-2017, precip in+out)
# on 32 anchors of 2021. With --regrid-to-deg 1.0 and --sparse-stride-deg 18.0
# to enable apples-to-apples comparison with GC_small at multiple metric grids.

#SBATCH --job-name=gc-headline-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gc_headline_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gc_headline_eval_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/eval_baseline_only.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_025_37lev_eval32_2021.zarr
ANCHORS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/eval_anchors_2021_seed0_n32_T.npy
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gc_headline_baseline_eval
mkdir -p "${OUT}"

${PYTHON} -u ${SCRIPT} \
  --ckpt-in "${CKPT}" \
  --data-path "${DATA}" \
  --stats-dir "${STATS}" \
  --resolution 0.25 \
  --mesh-size 6 \
  --baseline-msg-steps 16 \
  --anchor-times-npy "${ANCHORS}" \
  --n-samples 32 \
  --regrid-to-deg 1.0 \
  --sparse-stride-deg 18.0 \
  --sparse-levels 50 100 150 200 250 300 400 500 600 700 850 925 1000 \
  --out-json "${OUT}/gc_headline_025_eval32_2021.json"
