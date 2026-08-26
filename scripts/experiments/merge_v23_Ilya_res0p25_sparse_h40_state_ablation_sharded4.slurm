#!/usr/bin/env bash
# Merge four exact accumulator states for each warm4 state-ablation mode.
# The merge retains one ~70 GiB accumulator plus one incoming state at a time;
# 256G leaves headroom for pickle loading and spatial RMS-bias finalization.
#SBATCH --job-name=v23-r025-h40-mrg
#SBATCH --account=dabanin
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v23_r025_h40_merge_%A_%a.out
#SBATCH --error=logs/v23_r025_h40_merge_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_hybrid_2k_20260818/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_hybrid_lr3em6_2k"
OUT_DIR="${RUN_DIR}/eval/warm4_state_ablation/sharded4"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
SHARD_COUNT=4

if [[ -n "${OUTPUT_SUFFIX}" && ! "${OUTPUT_SUFFIX}" =~ ^_[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_SUFFIX must be empty or begin with '_' and contain path-safe characters" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODE="keep_warm_state"
    ;;
  1)
    MODE="reset_before_h1"
    ;;
  *)
    echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT_JSON="${OUT_DIR}/step200_${MODE}${OUTPUT_SUFFIX}_sharded4.json"
[[ ! -e "${OUT_JSON}" ]] || {
  echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2
  exit 1
}

ARGS=()
for SHARD_INDEX in 0 1 2 3; do
  STEM="step200_${MODE}${OUTPUT_SUFFIX}_shard${SHARD_INDEX}-of${SHARD_COUNT}"
  SHARD_JSON="${OUT_DIR}/${STEM}.json"
  MERGE_STATE="${OUT_DIR}/${STEM}.merge_state.pkl"
  [[ -f "${SHARD_JSON}" ]] || {
    echo "Missing shard JSON: ${SHARD_JSON}" >&2
    exit 1
  }
  [[ -f "${MERGE_STATE}" ]] || {
    echo "Missing merge state: ${MERGE_STATE}" >&2
    exit 1
  }
  ARGS+=(--shard-json "${SHARD_JSON}" --merge-state "${MERGE_STATE}")
done

source scripts/graphcast_env.sh

echo "mode=${MODE} output=${OUT_JSON}"
python -u scripts/analyze_models/merge_v23_Ilya_eval_shards.py \
  "${ARGS[@]}" --out-json "${OUT_JSON}" --delete-merge-states

echo "Completed exact v23_Ilya H40 shard merge: ${OUT_JSON}"
