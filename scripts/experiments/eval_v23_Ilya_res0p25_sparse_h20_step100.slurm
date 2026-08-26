#!/usr/bin/env bash
# Matched cold closed-loop evaluation of the v23_Ilya res0.25 sparse-horizon
# checkpoint at step 100.  One 0.25-degree 64-step sample is much larger than
# the res1 evaluation shape; 240G leaves headroom over the 82.4G peak observed
# for the related streaming training-objective calibration (job 12452147).
#SBATCH --job-name=v23-r025-s100-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v23_r025_sparse_step100_eval_%j.out
#SBATCH --error=logs/v23_r025_sparse_step100_eval_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_2k_20260817/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_lr3em6_2k"
CKPT="${RUN_DIR}/checkpoints/checkpoint_step00000100.pkl"
OUT_JSON="${RUN_DIR}/eval/cold_full_zero/step100.json"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

for required in "${CKPT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-s100-eval-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "checkpoint=${CKPT} output=${OUT_JSON}"
python -u scripts/analyze_models/eval_v23_Ilya.py \
  --ckpt "${CKPT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2019 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups 1 --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${OUT_JSON}"

echo "Completed v23_Ilya res0.25 checkpoint-100 evaluation: ${OUT_JSON}"
