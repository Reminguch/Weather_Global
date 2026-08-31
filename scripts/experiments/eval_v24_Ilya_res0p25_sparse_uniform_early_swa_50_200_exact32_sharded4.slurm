#!/usr/bin/env bash
# Four-GPU exact 32-anchor / 10-day evaluation of uniform-sparse early
# SWA(50,100,150,200). Each task evaluates one eight-anchor shard.
#SBATCH --job-name=v24-r025-spunswev
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:30:00
#SBATCH --output=logs/v24_r025_spuni_early_swa_shard4_%A_%a.out
#SBATCH --error=logs/v24_r025_spuni_early_swa_shard4_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_sparse_uniform_weight_ablation_20260829/legacy_di16_bcg1_sparse_uniform_h20_no_tzero_fp32_lr3em6_sg500"
SWA_TAG="swa_step00050-00200"
CHECKPOINT="${RUN_DIR}/swa/${SWA_TAG}.pkl"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
SHARD_COUNT=4
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if ((SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT)); then
  echo "Shard ${SHARD_INDEX} is outside 0-$((SHARD_COUNT - 1))" >&2
  exit 2
fi

OUT_DIR="${RUN_DIR}/eval/cold_full_zero_exact_gc/reference32_sharded4"
STEM="${SWA_TAG}_shard${SHARD_INDEX}-of${SHARD_COUNT}"
OUT_JSON="${OUT_DIR}/${STEM}.json"
PARTIAL_JSON="${OUT_DIR}/${STEM}.partial.json"
MERGE_STATE="${OUT_DIR}/${STEM}.merge_state.pkl"

for required in "${CHECKPOINT}" "${RUN_DIR}/run_config.json" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
for output in "${OUT_JSON}" "${PARTIAL_JSON}" "${MERGE_STATE}"; do
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite existing output: ${output}" >&2; exit 1; }
done
mkdir -p "${OUT_DIR}"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-spuni-early-swa-${SLURM_ARRAY_JOB_ID:-local}-${SHARD_INDEX}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "artifact=${SWA_TAG} shard=${SHARD_INDEX}/${SHARD_COUNT} anchors=32 leads=40"
echo "checkpoint=${CHECKPOINT}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${CHECKPOINT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2019 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme legacy_haiku \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups 1 --temporal-stateful \
  --no-zero-init-out --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --omit-rms-bias --seed 0 \
  --anchor-shard-count "${SHARD_COUNT}" --anchor-shard-index "${SHARD_INDEX}" \
  --merge-state-out "${MERGE_STATE}" --out-json "${OUT_JSON}"

echo "Completed ${SWA_TAG} shard=${SHARD_INDEX}/${SHARD_COUNT}"
