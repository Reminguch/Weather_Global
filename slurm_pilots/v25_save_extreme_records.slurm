#!/usr/bin/env bash
# Save full-grid 2022 extreme-records predictions at res=2 for:
#   task 0: v25 step3000 (gated extreme head)
#   task 1: v22 K=18 step20000 (open-loop baseline)
# anchor stride 6 days → ~60 anchors, K=40 leads, 3 surface vars (2m_T, u10, v10).
#SBATCH --job-name=v25-saveextreme
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_saveextreme_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_saveextreme_t%a_%A.err
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
mkdir -p ${OUT_DIR}

CKPTS=(
  "${ROOT}/results/v25_gated_extreme/K18_gext_phase1_from_step20k/v13_residual_step3000.pkl"
  "${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl"
)
TAGS=(v25_step3000_res2 v22_K18_step20000_res2)
GATED=(--gated-extreme-head "")

CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
GATED_FLAG=${GATED[$SLURM_ARRAY_TASK_ID]}
OUT_NC=${OUT_DIR}/${TAG}_extreme_records_K40_2022.nc

[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

echo "================================================"
echo "=== TASK $SLURM_ARRAY_TASK_ID: ${TAG}"
echo "=== ckpt=${CKPT}"
echo "=== gated_flag='${GATED_FLAG}'"
echo "================================================"

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
  ${GATED_FLAG} \
  --out-nc "${OUT_NC}"

echo "Done: ${OUT_NC}"
