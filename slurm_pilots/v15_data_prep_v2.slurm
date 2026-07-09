#!/usr/bin/env bash
# Data prep for v15 v2: WB2-cloud-6h ERA5 (paper-standard).
# Phase 1: prepare_streaming_store (zarr -> per-var .npy memmap)  ~10 min
# Phase 2: precompute_residual_targets (frozen GC1 forward per anchor)  ~30-60 min on GPU

#SBATCH --job-name=v15-data-prep-v2
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15_data_prep_v2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15_data_prep_v2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2

echo "============================================================"
echo "Phase 1: prepare_streaming_store on WB2-6h zarr"
echo "============================================================"
${PYTHON} -u -m src.data.prepare_streaming_store \
  --data-path "${DATA}" \
  --ckpt-in "${CKPT}" \
  --out-root "${PREPARED_ROOT}" \
  --resolutions 1.0 \
  --input-duration 12h \
  --overwrite

echo
echo "============================================================"
echo "Phase 2: precompute_residual_targets"
echo "  train: 2015-2021 (7 years), val: 2022"
echo "============================================================"
${PYTHON} -u -m src.data.precompute_residual_targets \
  --prepared-root "${PREPARED_ROOT}/res1" \
  --ckpt-in "${CKPT}" \
  --stats-dir "${STATS}" \
  --out-root "${RESIDUAL_ROOT}" \
  --resolution 1.0 --mesh-size 5 --baseline-msg-steps 16 \
  --input-duration 12h --target-steps 1 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --precision bf16 --seed 0 --print-every 50 --log-every 200 \
  --overwrite

echo "[done] data prep complete"
echo "  prepared store: ${PREPARED_ROOT}/res1"
echo "  residuals:      ${RESIDUAL_ROOT}"
ls -la "${RESIDUAL_ROOT}/residuals" || true
