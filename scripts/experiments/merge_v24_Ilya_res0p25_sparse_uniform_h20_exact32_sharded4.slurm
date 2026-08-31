#!/usr/bin/env bash
# Merge four native v24 res0.25 uniform-sparse anchor shards exactly.
# Merge states contain sufficient statistics, so the resulting GraphCast loss is
# exactly the 32-anchor aggregate—not an average of per-shard percentages.
#SBATCH --job-name=v24-r025-spunmrg
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v24_r025_spuni_shard4_merge_%j.out
#SBATCH --error=logs/v24_r025_spuni_shard4_merge_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_sparse_uniform_weight_ablation_20260829/legacy_di16_bcg1_sparse_uniform_h20_no_tzero_fp32_lr3em6_sg500"
STEP="${STEP:?Set STEP to the checkpoint step merged by this job}"
SHARD_COUNT=4
case "${STEP}" in
  50|100|150|200|250) ;;
  *) echo "Unsupported STEP=${STEP}; expected one of 50,100,150,200,250" >&2; exit 2 ;;
esac

OUT_DIR="${RUN_DIR}/eval/cold_full_zero_exact_gc/reference32_sharded4"
OUT_JSON="${OUT_DIR}/step${STEP}.json"
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }

ARGS=()
for shard in 0 1 2 3; do
  stem="${OUT_DIR}/step${STEP}_shard${shard}-of${SHARD_COUNT}"
  shard_json="${stem}.json"
  merge_state="${stem}.merge_state.pkl"
  [[ -f "${shard_json}" ]] || { echo "Missing shard JSON: ${shard_json}" >&2; exit 1; }
  [[ -f "${merge_state}" ]] || { echo "Missing merge state: ${merge_state}" >&2; exit 1; }
  ARGS+=(--shard-json "${shard_json}" --merge-state "${merge_state}")
done

source scripts/graphcast_env.sh
python -u scripts/analyze_models/merge_v24_Ilya_eval_shards.py \
  "${ARGS[@]}" --out-json "${OUT_JSON}" --delete-merge-states

python - "${OUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

metric = json.loads(Path(sys.argv[1]).read_text())["original_graphcast_loss"]
print(
    f"rollout_improvement_pct={metric['improvement_pct_rollout']:.6f} "
    f"day10_improvement_pct={metric['improvement_pct_per_step'][39]:.6f}"
)
PY
