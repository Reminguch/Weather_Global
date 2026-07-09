#!/usr/bin/env bash
# Phase 1: prepare streaming store (zarr -> per-variable .npy memmaps)
# Phase 2: precompute residual_target = truth - baseline_GC1_pred for every
#          valid (input_window, target) anchor in train+val years
#
# Output (on scratch):
#   /scratch/.../data/graphcast/graphcast/dataset/prepared_stream/res1/
#       coords/{time,lat,lon,level}.npy
#       vars/{var}.npy  (full multi-year arrays; memmap-friendly)
#       metadata.json
#   /scratch/.../data/graphcast/graphcast/dataset/precomputed_residuals/v11_setup_res1/
#       residuals/{var}.npy  (shape [N_anchors, target_steps, lat, lon, level?])
#       anchors/{anchor_indices, anchor_times, split_train, split_val}.npy
#       metadata.json
#
# Phase 1 is fast (just disk write, ~5-10 min).
# Phase 2 runs baseline GC1 once per anchor (~1 s/step), so for ~4K anchors
#         it takes ~1-2 hours on a single A100.

#SBATCH --job-name=v11-data-prep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v11_data_prep_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v11_data_prep_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
STATS="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats"
PREPARED_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream"
RESIDUAL_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v11_setup_res1"

echo "============================================================"
echo "Phase 1: prepare streaming store (zarr -> npy memmaps)"
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
echo "Phase 2: precompute residual_target = truth - baseline_GC1"
echo "============================================================"
${PYTHON} -u -m src.data.precompute_residual_targets \
  --prepared-root "${PREPARED_ROOT}/res1" \
  --ckpt-in "${CKPT}" \
  --stats-dir "${STATS}" \
  --out-root "${RESIDUAL_ROOT}" \
  --resolution 1.0 --mesh-size 5 --baseline-msg-steps 16 \
  --input-duration 12h --target-steps 1 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --precision bf16 --seed 0 --print-every 20 --log-every 100 \
  --overwrite

echo
echo "[done] data prep complete"
echo "  prepared store: ${PREPARED_ROOT}/res1"
echo "  residuals:      ${RESIDUAL_ROOT}"
ls -la "${RESIDUAL_ROOT}/residuals" || true
