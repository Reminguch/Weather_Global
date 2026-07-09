#!/usr/bin/env bash
# Eval GraphCast_operational (0.25 deg, 13 levels, mesh 2to6) on 32 val anchors
# from 2022, same calendar dates as v15 v2 eval.

#SBATCH --job-name=gc-025-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gc_025_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gc_025_eval_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/eval_baseline_only.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_025_eval32_2022.zarr
ANCHORS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/eval_anchors_2022_seed0_n32_T.npy
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gc_025_baseline_eval

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
  --out-json "${OUT}/gc_operational_025_eval32_2022.json"
