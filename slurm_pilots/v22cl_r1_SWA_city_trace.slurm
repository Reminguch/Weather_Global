#!/usr/bin/env bash
# City traces for v22cl res=1 SWA ckpts (K=8, K=16, K=22).
# 3 K × 4 anchors (Jan/Apr/Jul/Oct 2022) = 12 tasks.
#SBATCH --job-name=v22cl-r1-SWA-city
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-11
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_SWA_city_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_SWA_city_t%a_%A.err
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
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22cl-r1-SWA-city-traces
mkdir -p ${OUT_DIR}

# 3 K × 4 anchors
KS=(8 8 8 8   16 16 16 16   22 22 22 22)
ANCHORS=("2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00"
         "2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00"
         "2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00")
TAGS=("20220115" "20220415" "20220715" "20221015"
      "20220115" "20220415" "20220715" "20221015"
      "20220115" "20220415" "20220715" "20221015")

K=${KS[$SLURM_ARRAY_TASK_ID]}
ANCHOR=${ANCHORS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

CKPT=${EMA_DIR}/K${K}_ema.pkl
OUT_JSON=${OUT_DIR}/v22cl_r1_K${K}_SWA_cities_${TAG}.json

[ -f "${CKPT}" ] || { echo "ERR missing ${CKPT}"; exit 1; }

echo "=== v22cl K=${K} SWA city trace anchor=${ANCHOR} ==="
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
  --out-json "${OUT_JSON}"
echo "Done K=${K} anchor=${ANCHOR}"
