#!/usr/bin/env bash
# Streamed H1--H40 evaluation of raw step 200 after four truth-fed calls.
# Task 0 retains the resulting residual state; task 1 resets it once before H1.
# Both tasks save exact original GraphCast loss plus the full per-variable,
# per-channel, RMSE, MAE, RMS-bias, and residual-diagnostic schema.
#
# The completed cold 32-anchor selection jobs used about 206.8 GiB CPU RAM and
# 6.6--7.0 hours. Four warmup calls add modest work, so retain 256G/10h.
# The rollout is one-step streamed and requires one A100-80GB.
#SBATCH --job-name=v23-r025-h40-state
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/v23_r025_h40_state_%A_%a.out
#SBATCH --error=logs/v23_r025_h40_state_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_hybrid_2k_20260818/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_hybrid_lr3em6_2k"
CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_step00000200.pkl"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
N_SAMPLES="${N_SAMPLES:-32}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid N_SAMPLES=${N_SAMPLES}" >&2
  exit 2
}
if [[ -n "${OUTPUT_SUFFIX}" && ! "${OUTPUT_SUFFIX}" =~ ^_[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_SUFFIX must be empty or begin with '_' and contain path-safe characters" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODE="keep_warm_state"
    EVAL_MODE="warm_full"
    ;;
  1)
    MODE="reset_before_h1"
    EVAL_MODE="warm_full_reset_state"
    ;;
  *)
    echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT_DIR="${RUN_DIR}/eval/warm4_state_ablation"
OUT_JSON="${OUT_DIR}/step200_${MODE}${OUTPUT_SUFFIX}.json"
for required in "${CHECKPOINT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || {
    echo "Missing input: ${required}" >&2
    exit 1
  }
done
[[ ! -e "${OUT_JSON}" ]] || {
  echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2
  exit 1
}
mkdir -p "${OUT_DIR}"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-h40-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "mode=${MODE} eval_mode=${EVAL_MODE} samples=${N_SAMPLES}"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUT_JSON}"
echo "git_commit=$(git rev-parse HEAD)"

python -u scripts/analyze_models/eval_v23_Ilya.py \
  --ckpt "${CHECKPOINT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2019 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 --temporal-bc-groups 1 \
  --temporal-stateful --eval-mode "${EVAL_MODE}" \
  --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 4 \
  --n-samples "${N_SAMPLES}" --seed 0 --out-json "${OUT_JSON}"

echo "Completed streamed v23_Ilya H40 state ablation: ${OUT_JSON}"
