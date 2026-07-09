#!/usr/bin/env bash
# TEST: verify the frozen snapshot reproduces the drifted tree's eval exactly.
# Runs K=4 H=256 SWA cold_full eval from the FROZEN tree (relative script path!)
# Expected (from drifted-tree job 10894039 task 0):
#   24h +1.23% / 72h -1.92% / 120h -3.78% / 168h -4.09% / 240h -3.19%, chan+ 7/82
#SBATCH --job-name=test-h256frozen-K4
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/test_h256frozen_K4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/test_h256frozen_K4_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_h256frozen   # FROZEN tree; script path below is RELATIVE

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

K=4
CKPT_DIR=${ROOT}/results/v22closed/K${K}_H256_fresh_20k/v22closed_H256_K${K}_fresh_20k
CKPT=${CKPT_DIR}/v13_residual_SWA_step6k-16k_zero_state.pkl
[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-09-h256frozen-test
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/test_K${K}_H256_SWA_frozen_tree.json

echo "=== TEST: frozen-tree K=${K} H256 SWA cold_full eval ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 --temporal-d-inner 256 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "=== TEST done ==="
