#!/usr/bin/env bash
# All step 14000 (BEST K=1 ckpt) evals + diagnostics in one slurm.
# Output: eval_results_step14000/ folder.

#SBATCH --job-name=v15v2-s14k-all
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_s14k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_s14k_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
CKPT="${CKPT_BASE}/v13_residual_step14000.pkl"
OUT=${CKPT_BASE}/eval_results_step14000
mkdir -p "${OUT}"

COMMON="--ckpt ${CKPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2"

echo "===== 1. K=1 single-step eval ====="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9.py ${COMMON} \
  --n-samples 32 --out-json "${OUT}/eval_step14000.json"

echo "===== 2. K=6 rollout (RMSE + MAE) ====="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9_rollout.py ${COMMON} \
  --target-steps 6 --n-samples 32 --out-json "${OUT}/eval_step14000_rollout_K6.json"

echo "===== 3. diag-ts 200 anchors (per-anchor MAE/RMSE + spatial means) ====="
${PYTHON} -u scripts/training/full_mamba_v9/diag_var_timeseries.py ${COMMON} \
  --target-steps 6 --n-anchors 200 --var total_precipitation_6hr \
  --out-npz "${OUT}/diag_ts_tp6h_step14000.npz"
${PYTHON} -u scripts/training/full_mamba_v9/diag_var_timeseries.py ${COMMON} \
  --target-steps 6 --n-anchors 200 --var temperature --level 850 \
  --out-npz "${OUT}/diag_ts_T850_step14000.npz"

echo "===== 4. 8-anchor field dumps for spectra (11 variables) ====="
# Surface
for V in 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind \
         mean_sea_level_pressure total_precipitation_6hr; do
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --target-steps 6 --n-anchors 8 --var ${V} \
    --out-npz "${OUT}/diag_${V}_step14000.npz"
done
# Atmospheric
for ENTRY in "temperature:850" "geopotential:500" "specific_humidity:700" \
             "u_component_of_wind:500" "v_component_of_wind:500" "vertical_velocity:500"; do
  V="${ENTRY%:*}"; L="${ENTRY#*:}"
  TAG="${V}_level${L}"
  [[ "${V}" == "temperature" ]] && TAG="T${L}"
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --target-steps 6 --n-anchors 8 --var ${V} --level ${L} \
    --out-npz "${OUT}/diag_${TAG}_step14000.npz"
done

echo "===== 5. 100-anchor 2m_t + 10m winds for extreme records ====="
for V in 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind; do
  ${PYTHON} -u scripts/training/full_mamba_v9/diag_tp6h_errors.py ${COMMON} \
    --target-steps 6 --n-anchors 100 --var ${V} \
    --out-npz "${OUT}/diag_${V}_n100_step14000.npz"
done

echo "[done]"
