#!/usr/bin/env bash
# v22 paper K=2..20 cold_bp city traces via V22CLEAN tree (K=22 already done).
# 10 K × 4 anchors = 40 tasks.
#SBATCH --job-name=v22paper-city-allK
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22paper_city_allK_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22paper_city_allK_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
V22_ROOT=${ROOT}/results/v22
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22paper-city-fixed-v22cleantree
mkdir -p ${OUT_DIR}

K_LABELS=(18 20)   # K=22 already done
ANCHOR_DATES=("2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00")
ANCHOR_TAGS=("20220115" "20220415" "20220715" "20221015")

K_IDX=$((SLURM_ARRAY_TASK_ID / 4))
ANC_IDX=$((SLURM_ARRAY_TASK_ID % 4))
K=${K_LABELS[$K_IDX]}
ANCHOR=${ANCHOR_DATES[$ANC_IDX]}
ATAG=${ANCHOR_TAGS[$ANC_IDX]}

CKPT=${V22_ROOT}/K${K}_from_v22K1_23k/v13_residual_step26000.pkl
OUT_JSON=${OUT_DIR}/v22_K${K}_cities_${ATAG}.json

[ -f "${CKPT}" ] || { echo "ERR missing ${CKPT}"; exit 1; }

echo "=== v22 paper K=${K} cold_bp via v22clean tree  anchor=${ANCHOR} ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_city_trace.py \
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
  --feedback-mode cold_bp \
  --out-json "${OUT_JSON}"
echo "Done K=${K} ${ATAG}"
