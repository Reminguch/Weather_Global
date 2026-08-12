#!/usr/bin/env bash
# Train stateless residual-Mamba controls from the res2 500k vanilla GraphCast
# checkpoint.  K=12 is the legacy trainer's AR-tail length: each inner rollout
# call is one 6-hour step, and the final 12 anchors use autoregressive inputs.
#
# Array mapping:
#   0: baseline-only feedback (open)
#   1: baseline + stop-gradient residual feedback (closed_sg)
#
# This is deliberately separate from the historical 20k / K=14 ablation.
# It initializes a fresh residual head/Mamba on top of the specified frozen
# 500k GraphCast checkpoint; it never resumes or writes into that checkpoint.
#
# The completed 20k stateless predecessor (11317907) sustained about 0.89 s
# per optimizer update.  Fifty thousand updates, compile, and periodic
# checkpoints fit within 24 h.  32G is conservative relative to its 24G run
# and leaves room for the different 500k baseline checkpoint.
#SBATCH --job-name=resm-r2-gc500k-k12-stateless
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/resm_r2_gc500k_k12_stateless_%A_%a.out
#SBATCH --error=logs/resm_r2_gc500k_k12_stateless_%A_%a.err

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

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
PREPARED_ROOT="data/graphcast/graphcast/dataset/prepared_stream/res2"
STATS_DIR="data/graphcast/graphcast/stats"
RESIDUAL_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
AR_TAIL_K="${AR_TAIL_K:-12}"

FEEDBACK_MODES=(baseline closed_loop_sg)
LABELS=(open closed_sg)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#FEEDBACK_MODES[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#FEEDBACK_MODES[@]} - 1 ))" >&2
  exit 1
fi
FEEDBACK_MODE="${FEEDBACK_MODES[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
RUN_NAME="resmamba_res2_v20_k${AR_TAIL_K}_di16_l2_dc4_${LABEL}_stateless_from_gc500k_${MAX_STEPS}steps"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}" >&2; exit 1; }
[[ -d "${PREPARED_ROOT}" ]] || { echo "Missing prepared store: ${PREPARED_ROOT}" >&2; exit 1; }
[[ -f "${RESIDUAL_ROOT}/metadata.json" ]] || { echo "Missing residual cache: ${RESIDUAL_ROOT}" >&2; exit 1; }

CMD=(
  python -u scripts/training/full_mamba_v20/train_mz_v20.py
  --prepared-root "${PREPARED_ROOT}"
  --residual-root "${RESIDUAL_ROOT}"
  --ckpt-in "${BASELINE_CKPT}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --resolution 2.0 --mesh-size 4 --width 512
  --baseline-msg-steps 6 --residual-msg-steps 2
  --input-duration 12h --target-steps 1 --batch-size 1
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr 1e-4 --weight-decay 1e-4 --seed 18 --precision bf16
  --grad-clip 1.0 --warmup-steps 200
  --sequential-segment-steps 64 --bptt-steps 16 --ar-tail-K "${AR_TAIL_K}"
  --feedback-mode "${FEEDBACK_MODE}"
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 16 --temporal-d-state 16
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2
  --no-temporal-stateful
)

echo "task=${TASK_ID} feedback=${FEEDBACK_MODE} d_conv=4 stateful=false ar_tail_K=${AR_TAIL_K} run=${RUN_NAME}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
"${CMD[@]}"
