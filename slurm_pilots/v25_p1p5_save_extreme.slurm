#!/usr/bin/env bash
# Save 2022 extreme-records full-grid predictions for v25 phase 1.5 step 5000.
# Output goes alongside v22 K=18 baseline for direct WITH/WITHOUT comparison.
#SBATCH --job-name=v25-p1p5-saveext
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_p1p5_saveext_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_p1p5_saveext_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
STATS=${ROOT}/data/graphcast/graphcast/stats
OUT_DIR=${ROOT}/results/2026-6-27-v25-extreme

CKPT=${ROOT}/results/v25_gated_extreme/K18_gext_phase1p5_smoke/v13_residual_step5000.pkl
OUT_NC=${OUT_DIR}/v25_phase1p5_step5000_res2_extreme_records_K40_2022.nc

mkdir -p ${OUT_DIR}
[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

echo "=== Saving 2022 extreme records for v25 phase 1.5 step 5000 (gated extreme, gate_bias=-1) ==="

$PYTHON -u scripts/training/full_mamba_v20/save_extreme_records_data_v20.py \
  --ckpt "${CKPT}" \
  --ckpt-in "${GCV1_K3}" \
  --data-path "${DATA}" \
  --stats-dir "${STATS}" \
  --resolution 2.0 --mesh-size 4 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 \
  --anchor-stride-days 6 \
  --gated-extreme-head --gate-bias-init -1.0 \
  --out-nc "${OUT_NC}"

echo "Done: ${OUT_NC}"
