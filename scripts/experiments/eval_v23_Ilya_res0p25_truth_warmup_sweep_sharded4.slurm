#!/usr/bin/env bash
# Matched-date truth-fed warmup sweep: 20/40/60 warm steps, then 40 scored leads.
# Tasks 0--3 use warm20, 4--7 warm40, and 8--11 warm60. Each task evaluates
# one round-robin shard of the same seed-0 scored H1 anchors.
#
# Truth/forcing data are loaded in four-step blocks, only the current Mamba
# state is carried, and RMS-bias fields are omitted. The exact GraphCast loss
# plus per-variable/per-level RMSE and MAE (including 2m temperature) remain.
# Completed no-RMS-bias tasks peaked near 79 GiB of CPU RAM and took up to
# 4:49. Device metric reduction removes per-lead prediction copies; request
# 96G with modest walltime headroom until the optimized pilot is measured.
#SBATCH --job-name=v23-r025-warm-sweep
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --output=logs/v23_r025_warm_sweep_%A_%a.out
#SBATCH --error=logs/v23_r025_warm_sweep_%A_%a.err

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
METRIC_BACKEND="${METRIC_BACKEND:-auto}"
SHARD_COUNT=4
HISTORY_STEPS=60
STREAM_BLOCK_STEPS=4
WARMUPS=(20 40 60)

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
case "${METRIC_BACKEND}" in
  auto|cpu|device) ;;
  *)
    echo "METRIC_BACKEND must be auto, cpu, or device; got ${METRIC_BACKEND}" >&2
    exit 2
    ;;
esac

WARMUP_INDEX=$((SLURM_ARRAY_TASK_ID / SHARD_COUNT))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % SHARD_COUNT))
if (( WARMUP_INDEX < 0 || WARMUP_INDEX >= ${#WARMUPS[@]} )); then
  echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
WARMUP_STEPS="${WARMUPS[WARMUP_INDEX]}"

OUT_DIR="${RUN_DIR}/eval/truth_warmup_sweep_matched60/sharded4"
STEM="step200_warm${WARMUP_STEPS}${OUTPUT_SUFFIX}_shard${SHARD_INDEX}-of${SHARD_COUNT}"
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

export MPLCONFIGDIR="/tmp/mpl-v23-r025-warm-sweep-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export JAX_COMPILATION_CACHE_DIR="${ROOT}/artifacts/jax_compilation_cache/v23_r025_eval_a100"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
source scripts/graphcast_env.sh

echo "warmup=${WARMUP_STEPS} history=${HISTORY_STEPS} shard=${SHARD_INDEX}/${SHARD_COUNT} samples=${N_SAMPLES} metric_backend=${METRIC_BACKEND}"
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
  --temporal-stateful --eval-mode warm_full \
  --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps "${WARMUP_STEPS}" \
  --anchor-history-steps "${HISTORY_STEPS}" \
  --stream-block-steps "${STREAM_BLOCK_STEPS}" --omit-rms-bias \
  --metric-backend "${METRIC_BACKEND}" \
  --n-samples "${N_SAMPLES}" --seed 0 \
  --anchor-shard-count "${SHARD_COUNT}" \
  --anchor-shard-index "${SHARD_INDEX}" \
  --merge-state-out "${MERGE_STATE}" \
  --out-json "${OUT_JSON}"

echo "Completed v23_Ilya truth-warmup shard: ${OUT_JSON}"
