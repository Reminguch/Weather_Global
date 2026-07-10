#!/usr/bin/env bash
# TCBench smoke: v22cl K=22 SWA cold_full at 3 chosen 2023 inits
# (DOKSURI 2023-07-24T12, multi-storm 2023-08-30T00, HALE dateline 2023-01-09T12).
# Produces both pure GC-baseline self-rollout AND SWA closed-loop trajectory
# in a single NC file, ready for TempestExtremes tracking.
#SBATCH --job-name=tcbench-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/tcbench_smoke_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/tcbench_smoke_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/era5_2023_tcbench_smoke_1deg.zarr
CKPT=${ROOT}/results/v22cl_r1_EMA/K22_ema.pkl
OUT_DIR=${ROOT}/results/2026-07-04-tcbench-smoke
mkdir -p ${OUT_DIR}
OUT_NC=${OUT_DIR}/v22cl_K22_SWA_smoke_2023_K40.nc

echo "=== TCBench smoke: v22cl K=22 SWA cold_full, 3 inits, K=40 ==="
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
  --target-steps 40 \
  --feedback-mode cold_full \
  --residual-state-init zero \
  --anchor-times "2023-07-24T12:00:00,2023-08-30T00:00:00,2023-01-09T12:00:00" \
  --out-nc "${OUT_NC}"
echo "Done"
