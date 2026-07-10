#!/usr/bin/env bash
# City traces in cold_full feedback mode.
# 4 ckpts × 4 anchors = 16 tasks
#   ckpts: v22cl K=8/16/22 SWA + v22 paper K=22 (for direct open-vs-closed comparison)
#SBATCH --job-name=v22cl-city-cf
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-15
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_city_cf_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_city_cf_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
EMA_DIR=${ROOT}/results/v22cl_r1_EMA
V22_PAPER_CKPT=${ROOT}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22cl-r1-city-traces-coldfull
mkdir -p ${OUT_DIR}

# 4 ckpts × 4 anchors
CKPT_TAGS=("v22cl_K8_SWA" "v22cl_K16_SWA" "v22cl_K22_SWA" "v22_K22_open")
CKPT_PATHS=("${EMA_DIR}/K8_ema.pkl"
            "${EMA_DIR}/K16_ema.pkl"
            "${EMA_DIR}/K22_ema.pkl"
            "${V22_PAPER_CKPT}")
ANCHOR_DATES=("2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00")
ANCHOR_TAGS=("20220115" "20220415" "20220715" "20221015")

CKPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
ANC_IDX=$((SLURM_ARRAY_TASK_ID % 4))
TAG=${CKPT_TAGS[$CKPT_IDX]}
CKPT=${CKPT_PATHS[$CKPT_IDX]}
ANCHOR=${ANCHOR_DATES[$ANC_IDX]}
ATAG=${ANCHOR_TAGS[$ANC_IDX]}
OUT_JSON=${OUT_DIR}/${TAG}_cf_cities_${ATAG}.json

[ -f "${CKPT}" ] || { echo "ERR missing ${CKPT}"; exit 1; }

echo "=== city trace cold_full  ckpt=${TAG}  anchor=${ANCHOR} ==="
$PYTHON -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v23/eval_city_trace.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --target-steps 40 \
  --anchor-date ${ANCHOR} \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --feedback-mode cold_full \
  --out-json "${OUT_JSON}"
echo "Done ${TAG} ${ATAG}"
