#!/usr/bin/env bash
# City traces: all 11 K × 4 seasons × 2 model types = 88 tasks
#   model 0 (v22cl_SWA):     cold_full  ← closed-loop's natural eval
#   model 1 (v22_open_paper):cold_bp    ← open-loop's natural eval
#SBATCH --job-name=citytrace-allK
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-87
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/citytrace_allK_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/citytrace_allK_t%a_%A.err
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
V22_ROOT=${ROOT}/results/v22
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-03-city-traces-allK
mkdir -p ${OUT_DIR}

K_LABELS=(2 4 6 8 10 12 14 16 18 20 22)
ANCHOR_DATES=("2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00")
ANCHOR_TAGS=("20220115" "20220415" "20220715" "20221015")

# Indexing: 88 = 2 models × 11 K × 4 seasons
# model_idx = SLURM_ARRAY_TASK_ID / 44 ; rest / 4 = K_idx ; rest % 4 = season_idx
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 44))
REM=$((SLURM_ARRAY_TASK_ID % 44))
K_IDX=$((REM / 4))
ANC_IDX=$((REM % 4))
K=${K_LABELS[$K_IDX]}
ANCHOR=${ANCHOR_DATES[$ANC_IDX]}
ATAG=${ANCHOR_TAGS[$ANC_IDX]}

if [ "$MODEL_IDX" = "0" ]; then
  # v22cl SWA cold_full
  CKPT=${EMA_DIR}/K${K}_ema.pkl
  MODE=cold_full
  TAG="v22cl_K${K}_SWA_cf"
else
  # v22 paper open-loop cold_bp
  CKPT=${V22_ROOT}/K${K}_from_v22K1_23k/v13_residual_step26000.pkl
  MODE=cold_bp
  TAG="v22_K${K}_open_cbp"
fi

OUT_JSON=${OUT_DIR}/${TAG}_cities_${ATAG}.json

[ -f "${CKPT}" ] || { echo "ERR missing ${CKPT}"; exit 1; }

echo "=== ${TAG}  K=${K}  anchor=${ANCHOR}  mode=${MODE} ==="
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
  --feedback-mode ${MODE} \
  --out-json "${OUT_JSON}"
echo "Done ${TAG} ${ATAG}"
