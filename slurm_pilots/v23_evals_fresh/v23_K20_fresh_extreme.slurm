#!/usr/bin/env bash
# v23 K=20 fresh — extreme records dense .nc. afterok on training job 9000478.
#SBATCH --job-name=v23-K20-fresh-ext
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=05:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_K20_fresh_ext_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_K20_fresh_ext_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23/K20_from_K1_fresh_dconv8_23k/v13_residual_step26000.pkl
[ ! -f "${CKPT}" ] && { echo "missing ${CKPT}"; exit 1; }
OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_extreme
mkdir -p "${OUT_DIR}"
$PYTHON -u scripts/training/full_mamba_v20/save_extreme_records_data_v20.py \
  --ckpt "${CKPT}" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 8 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 --anchor-stride-days 3 \
  --out-nc ${OUT_DIR}/v23_K20_fresh_extreme_records_K40_2022.nc
