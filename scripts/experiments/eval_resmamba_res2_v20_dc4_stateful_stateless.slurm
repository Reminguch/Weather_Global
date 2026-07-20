#!/usr/bin/env bash
# Matched final-checkpoint rollout evaluation for the d_conv=4 state-carry
# ablation.  Tasks 0/1 are the stateful predecessors (which reached 16k);
# tasks 2/3 are the stateless variants (which reached 20k).
#
# All tasks use the same 32 seed-0 validation anchors and a zero residual
# state.  The evaluator records the frozen vanilla-GC rollout beside every
# residual-Mamba rollout, making this a matched vanilla-GC comparison.
#
# Budget evidence: analogous res2 evaluations (11269707) took 28--34 min and
# peaked at 6.7 GiB CPU RAM.  12 GiB and 1:02 provide 1.8x RAM and time margin.
#SBATCH --job-name=resm-r2-dc4-state
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/resm_r2_dc4_state_%A_%a.out
#SBATCH --error=logs/resm_r2_dc4_state_%A_%a.err

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
STATEFUL_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k"
STATELESS_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dconv4_stateless_ablation_20k"

RUNS=(
  "resmamba_res2_v20_k14_di16_l2_dc4_open_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc4_open_stateless_20000steps"
  "resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_stateless_20000steps"
)
ROOTS=("${STATEFUL_ROOT}" "${STATEFUL_ROOT}" "${STATELESS_ROOT}" "${STATELESS_ROOT}")
STEPS=(16000 16000 20000 20000)
MODES=(cold_bp cold_full cold_bp cold_full)
STATEFUL=(true true false false)

I="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="${RUNS[$I]}"
RUN_ROOT="${ROOTS[$I]}"
STEP="${STEPS[$I]}"
MODE="${MODES[$I]}"
IS_STATEFUL="${STATEFUL[$I]}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
CKPT="${RUN_DIR}/v13_residual_step${STEP}.pkl"
# Keep these matched, zero-state evaluations separate from historical logs.
OUT_JSON="${RUN_DIR}/eval/${MODE}_zero_matched/step${STEP}.json"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}"; exit 1; }
[[ -d "${EVAL_DATA}" ]] || { echo "Missing evaluation data: ${EVAL_DATA}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
if [[ -s "${OUT_JSON}" ]]; then
  echo "Already evaluated: ${OUT_JSON}"
  exit 0
fi
mkdir -p "$(dirname "${OUT_JSON}")"

echo "run=${RUN_NAME} step=${STEP} d_conv=4 mode=${MODE} stateful=${IS_STATEFUL}"
STATEFUL_FLAG=(--temporal-stateful)
if [[ "${IS_STATEFUL}" != true ]]; then
  STATEFUL_FLAG=(--no-temporal-stateful)
fi

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
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  "${STATEFUL_FLAG[@]}" \
  --eval-mode "${MODE}" --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"

echo "Completed: ${OUT_JSON}"
