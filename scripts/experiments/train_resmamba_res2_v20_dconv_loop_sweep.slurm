#!/usr/bin/env bash
# Legacy v20 res2 residual-Mamba sweep, using the original long-AR trainer.
#
# Array mapping:
#   0/1/2: v20 baseline-only feedback, d_conv=1/2/4
#   3/4/5: baseline+stop-gradient residual feedback, d_conv=1/2/4
#
# All runs start from the same frozen local vanilla GraphCast checkpoint and
# train fresh residual heads for 20,000 optimizer updates.
#
# ``RESIDUAL_ROOT`` is the v22 structural cache for this exact prepared res2
# stream.  Its arrays are intentionally one-element placeholders: v20 forms
# truth - frozen_baseline live inside every AR step, and only uses this cache
# for its validated anchor/split metadata.
#SBATCH --job-name=resm-r2-v20-loop
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
# 2-step legacy-v20 smoke 11222546_0 peaked at 7.0 GiB CPU RAM (including
# compilation) and measured 0.87 s/steady-state update.  16 GiB and 8 h leave
# headroom for the d_conv=4/closed-loop variants and ~5 h of 20k updates.
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/resm_r2_v20_loop_dconv_%A_%a.out
#SBATCH --error=logs/resm_r2_v20_loop_dconv_%A_%a.err

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
PREPARED_ROOT="data/graphcast/graphcast/dataset/prepared_stream/res2"
STATS_DIR="data/graphcast/graphcast/stats"
RESIDUAL_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k}"
MAX_STEPS="${MAX_STEPS:-20000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"

DCONVS=(1 2 4 1 2 4)
LOOPS=(baseline baseline baseline closed_loop_sg closed_loop_sg closed_loop_sg)
I="${SLURM_ARRAY_TASK_ID}"
DCONV="${DCONVS[$I]}"
FEEDBACK_MODE="${LOOPS[$I]}"
LABEL="open"
[[ "${FEEDBACK_MODE}" == "closed_loop_sg" ]] && LABEL="closed_sg"
RUN_NAME="resmamba_res2_v20_k14_di16_l2_dc${DCONV}_${LABEL}_${MAX_STEPS}steps"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}"; exit 1; }
[[ -d "${PREPARED_ROOT}" ]] || { echo "Missing prepared res2 store: ${PREPARED_ROOT}"; exit 1; }
[[ -f "${RESIDUAL_ROOT}/metadata.json" ]] || { echo "Missing v20 anchor cache: ${RESIDUAL_ROOT}"; exit 1; }

echo "task=${I} feedback=${FEEDBACK_MODE} d_conv=${DCONV} run=${RUN_NAME}"
python -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${BASELINE_CKPT}" \
  --stats-dir "${STATS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 --batch-size 1 \
  --max-steps "${MAX_STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" \
  --lr 1e-4 --weight-decay 1e-4 --seed 18 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 64 --bptt-steps 16 --ar-tail-K 14 \
  --feedback-mode "${FEEDBACK_MODE}" \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv "${DCONV}" --temporal-dt-rank auto --temporal-layers 2
