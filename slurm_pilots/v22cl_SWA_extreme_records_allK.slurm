#!/usr/bin/env bash
# v22cl SWA cold_full extreme records for K=2..20 (K=22 already done via pilot).
# 10 K values, 1 task each. Each ~40min.
#SBATCH --job-name=v22cl-extreme-allK
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_extreme_allK_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_extreme_allK_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
OUT_DIR=${ROOT}/results/2026-07-03-v22cl-SWA-extreme
mkdir -p ${OUT_DIR}

K_LABELS=(2 4 6 8 10 12 14 16 18 20)
K=${K_LABELS[$SLURM_ARRAY_TASK_ID]}
CKPT=${ROOT}/results/v22cl_r1_EMA/K${K}_ema.pkl
OUT_NC=${OUT_DIR}/v22cl_K${K}_SWA_extreme_records_K40_2022.nc

[ -f "${CKPT}" ] || { echo "ERR missing ${CKPT}"; exit 1; }

echo "=== v22cl K=${K} SWA cold_full extreme records ==="
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
echo "Done K=${K}"
