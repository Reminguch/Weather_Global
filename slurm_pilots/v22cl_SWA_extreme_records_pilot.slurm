#!/usr/bin/env bash
# PILOT: v22cl K=22 SWA cold_full extreme records (K=40 rollout, ~120 anchors 2022).
# Verify pipeline before submitting full 11-K sweep.
#SBATCH --job-name=v22cl-extreme-pilot
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_extreme_pilot_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_extreme_pilot_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT=${ROOT}/results/v22cl_r1_EMA/K22_ema.pkl
OUT_DIR=${ROOT}/results/2026-07-03-v22cl-SWA-extreme
mkdir -p ${OUT_DIR}
OUT_NC=${OUT_DIR}/v22cl_K22_SWA_extreme_records_K40_2022.nc

echo "=== v22cl K=22 SWA cold_full extreme records (pilot) ==="
$PYTHON -u scripts/training/full_mamba_v23/save_extreme_records_data_v22cl.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 --anchor-stride-days 3 \
  --feedback-mode cold_full \
  --residual-state-init zero \
  --out-nc "${OUT_NC}"
echo "Done pilot"
