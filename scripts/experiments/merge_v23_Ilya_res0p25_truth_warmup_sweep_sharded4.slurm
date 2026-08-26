#!/usr/bin/env bash
# Merge the four lightweight metric shards for warm20, warm40, or warm60.
#SBATCH --job-name=v23-r025-warm-merge
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v23_r025_warm_merge_%A_%a.out
#SBATCH --error=logs/v23_r025_warm_merge_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_hybrid_2k_20260818/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_hybrid_lr3em6_2k"
OUT_DIR="${RUN_DIR}/eval/truth_warmup_sweep_matched60/sharded4"
WARMUPS=(20 40 60)
WARMUP_STEPS="${WARMUPS[SLURM_ARRAY_TASK_ID]}"
OUT_JSON="${OUT_DIR}/step200_warm${WARMUP_STEPS}_sharded4.json"

args=()
for shard in 0 1 2 3; do
  stem="${OUT_DIR}/step200_warm${WARMUP_STEPS}_shard${shard}-of4"
  args+=(--shard-json "${stem}.json" --merge-state "${stem}.merge_state.pkl")
done

[[ ! -e "${OUT_JSON}" ]] || {
  echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2
  exit 1
}

source scripts/graphcast_env.sh
python -u scripts/analyze_models/merge_v23_Ilya_eval_shards.py \
  "${args[@]}" \
  --out-json "${OUT_JSON}" \
  --delete-merge-states

echo "Completed truth-warmup merge: ${OUT_JSON}"
