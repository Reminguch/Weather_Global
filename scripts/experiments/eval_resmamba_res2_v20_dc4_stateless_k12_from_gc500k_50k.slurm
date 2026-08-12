#!/usr/bin/env bash
# Evaluate the final 50k stateless K=12 residual-Mamba checkpoints on the
# same cold rollout protocol used by the existing d_conv=4 curves.
#SBATCH --job-name=resm-r2-gc500k-k12-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/resm_r2_gc500k_k12_eval_%A_%a.out
#SBATCH --error=logs/resm_r2_gc500k_k12_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
RUN_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
RUNS=(
  "resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps"
  "resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps"
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUNS[$TASK_ID]}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
MODE="cold_bp"
[[ "${RUN_NAME}" == *"_closed_sg_"* ]] && MODE="cold_full"
CKPT="${RUN_DIR}/v13_residual_step50000.pkl"
OUT_JSON="${RUN_DIR}/eval/${MODE}_zero_from_gc500k/step50000.json"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}" >&2; exit 1; }
[[ -d "${EVAL_DATA}" ]] || { echo "Missing evaluation data: ${EVAL_DATA}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing final residual checkpoint: ${CKPT}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

python -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "${CKPT}" --ckpt-in "${BASELINE_CKPT}" --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 --input-duration 12h \
  --temporal-location mesh_processor_interleaved --temporal-hidden-size 128 \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --no-temporal-stateful --eval-mode "${MODE}" --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 --out-json "${OUT_JSON}"

echo "Completed: ${OUT_JSON}"
