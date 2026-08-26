#!/usr/bin/env bash
# Exact H1--H40 warm4 state ablation split into four matched anchor shards.
# Tasks 0--3 keep warm state; tasks 4--7 reset state before H1.
# Each task evaluates 8 of the same 32 seed-0 anchors and writes the full shard
# JSON plus a temporary sufficient-statistics state for exact RMS-bias merging.
#
# One shard is estimated at ~74 min first-anchor compile/run + 7 * ~13 min,
# with extra time for finalization and the ~70 GiB merge-state write.  Retain
# the proven 256G/A100-80GB memory shape and allow five hours.
#SBATCH --job-name=v23-r025-h40-sh4
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=logs/v23_r025_h40_sh4_%A_%a.out
#SBATCH --error=logs/v23_r025_h40_sh4_%A_%a.err

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
SHARD_COUNT=4

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid N_SAMPLES=${N_SAMPLES}" >&2
  exit 2
}
(( N_SAMPLES % SHARD_COUNT == 0 )) || {
  echo "N_SAMPLES=${N_SAMPLES} must be divisible by ${SHARD_COUNT}" >&2
  exit 2
}
if [[ -n "${OUTPUT_SUFFIX}" && ! "${OUTPUT_SUFFIX}" =~ ^_[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_SUFFIX must be empty or begin with '_' and contain path-safe characters" >&2
  exit 2
fi

MODE_INDEX=$((SLURM_ARRAY_TASK_ID / SHARD_COUNT))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % SHARD_COUNT))
case "${MODE_INDEX}" in
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

OUT_DIR="${RUN_DIR}/eval/warm4_state_ablation/sharded4"
STEM="step200_${MODE}${OUTPUT_SUFFIX}_shard${SHARD_INDEX}-of${SHARD_COUNT}"
OUT_JSON="${OUT_DIR}/${STEM}.json"
PARTIAL_JSON="${OUT_DIR}/${STEM}.partial.json"
MERGE_STATE="${OUT_DIR}/${STEM}.merge_state.pkl"
for required in "${CHECKPOINT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || {
    echo "Missing input: ${required}" >&2
    exit 1
  }
done
for output in "${OUT_JSON}" "${PARTIAL_JSON}" "${MERGE_STATE}"; do
  [[ ! -e "${output}" ]] || {
    echo "Refusing to overwrite existing output: ${output}" >&2
    exit 1
  }
done
mkdir -p "${OUT_DIR}"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-h40-sh4-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "mode=${MODE} eval_mode=${EVAL_MODE} shard=${SHARD_INDEX}/${SHARD_COUNT} samples=${N_SAMPLES}"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUT_JSON}"
echo "merge_state=${MERGE_STATE}"
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
  --n-samples "${N_SAMPLES}" --seed 0 \
  --anchor-shard-count "${SHARD_COUNT}" \
  --anchor-shard-index "${SHARD_INDEX}" \
  --merge-state-out "${MERGE_STATE}" \
  --out-json "${OUT_JSON}"

echo "Completed v23_Ilya H40 shard: ${OUT_JSON}"
