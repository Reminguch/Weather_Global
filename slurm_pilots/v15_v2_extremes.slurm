#!/usr/bin/env bash
# (1) Compute historical records (CPU, parallel with model eval).
# (2) Dump model rollout fields for 2m_t and 10m winds for 100 val anchors.

#SBATCH --job-name=v15v2-extremes
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_extremes_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_extremes_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
OUT=${CKPT_BASE}/eval_results

# Phase 1: historical records 2015-2021
RECORDS="${OUT}/historical_records_2015_2021.npz"
if [[ ! -f "${RECORDS}" ]]; then
  echo "===== Compute historical records 2015-2021 ====="
  ${PYTHON} -u scripts/training/full_mamba_v9/compute_historical_records.py \
    --data-path ${DATA} --start-year 2015 --end-year 2021 \
    --out-npz "${RECORDS}"
else
  echo "[skip] records already computed"
fi

# Phase 2: 100-anchor rollout dumps for 2m_t, 10m_u, 10m_v
COMMON="--ckpt ${CKPT_BASE}/v13_residual_step20000.pkl \
  --data-path ${DATA} \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --n-anchors 100"

for V in 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind; do
  OUT_NPZ="${OUT}/diag_${V}_n100_step20000.npz"
  [[ -f "${OUT_NPZ}" ]] && { echo "[skip] ${V}"; continue; }
  echo "===== rollout dump 100 anchors: ${V} ====="
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --var ${V} --out-npz "${OUT_NPZ}"
done
echo "[done]"
