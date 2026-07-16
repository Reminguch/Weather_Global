#!/usr/bin/env bash
# Evaluate every saved checkpoint from the legacy-v20 res2 d_conv/feedback sweep.
#
# Array layout: six runs x eight saved checkpoints (steps 2k through 16k).
# Open-loop runs use cold_bp; closed-loop runs use cold_full.  Both force a
# zero-initialized residual state, matching the original v22 evaluation pilots.
# Each result is kept with its source checkpoint:
#   <run>/eval/<mode>/stepNNNNN.json
#
# Budget evidence: completed residual evaluation 10250667 used 95 GiB and
# 1:27:46 at res1; this res2 / d_inner16 sweep therefore uses 160 GiB and
# 2:30 of walltime for headroom.
#SBATCH --job-name=resm-r2-v20-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --array=0-47%4
#SBATCH --output=logs/resm_r2_v20_eval_%A_%a.out
#SBATCH --error=logs/resm_r2_v20_eval_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

source scripts/graphcast_env.sh

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
SWEEP_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k"

RUNS=(
  "resmamba_res2_v20_k14_di16_l2_dc1_open_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc2_open_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc4_open_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc1_closed_sg_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc2_closed_sg_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_20000steps"
)
STEPS=(2000 4000 6000 8000 10000 12000 14000 16000)

I="${SLURM_ARRAY_TASK_ID}"
RUN_IDX=$((I / ${#STEPS[@]}))
STEP_IDX=$((I % ${#STEPS[@]}))
RUN_NAME="${RUNS[$RUN_IDX]}"
STEP="${STEPS[$STEP_IDX]}"
RUN_DIR="${SWEEP_ROOT}/${RUN_NAME}"
CKPT="${RUN_DIR}/v13_residual_step${STEP}.pkl"

MODE="cold_bp"
[[ "${RUN_NAME}" == *"_closed_sg_"* ]] && MODE="cold_full"
DCONV="${RUN_NAME#*_dc}"
DCONV="${DCONV%%_*}"
OUT_JSON="${RUN_DIR}/eval/${MODE}/step${STEP}.json"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}"; exit 1; }
[[ -d "${EVAL_DATA}" ]] || { echo "Missing evaluation data: ${EVAL_DATA}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
if [[ -s "${OUT_JSON}" ]]; then
  echo "Already evaluated: ${OUT_JSON}"
  exit 0
fi
mkdir -p "$(dirname "${OUT_JSON}")"

echo "run=${RUN_NAME} step=${STEP} d_conv=${DCONV} mode=${MODE}"
python -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "${CKPT}" \
  --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" \
  --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv "${DCONV}" --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode "${MODE}" --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"

echo "Completed: ${OUT_JSON}"
