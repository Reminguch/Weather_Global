#!/usr/bin/env bash
# Dump raw truth/baseline/full fields for many variables, K=6 rollout.

#SBATCH --job-name=v15v2-diag-vars
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_diag_vars_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_diag_vars_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
OUT=${CKPT_BASE}/eval_results
COMMON="--ckpt ${CKPT_BASE}/v13_residual_step20000.pkl \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --n-anchors 8"

# Surface
for V in 10m_u_component_of_wind 10m_v_component_of_wind 2m_temperature mean_sea_level_pressure; do
  OUT_NPZ="${OUT}/diag_${V}_step20000.npz"
  [[ -f "${OUT_NPZ}" ]] && { echo "[skip] ${V}"; continue; }
  echo "===== ${V} ====="
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --var ${V} --out-npz "${OUT_NPZ}"
done

# Atmospheric — pick representative level per variable
for ENTRY in "geopotential:500" "specific_humidity:700" "u_component_of_wind:500" \
             "v_component_of_wind:500" "vertical_velocity:500"; do
  V="${ENTRY%:*}"; L="${ENTRY#*:}"
  OUT_NPZ="${OUT}/diag_${V}_level${L}_step20000.npz"
  [[ -f "${OUT_NPZ}" ]] && { echo "[skip] ${V}@${L}"; continue; }
  echo "===== ${V} @ ${L} hPa ====="
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --var ${V} --level ${L} --out-npz "${OUT_NPZ}"
done
echo "[all done]"
