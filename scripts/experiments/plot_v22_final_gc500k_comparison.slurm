#!/usr/bin/env bash
# Render the gc500k/AR-k12 comparison from the v22_final rerun JSON files.
# Prior plot job 11843739 used 148 MiB and 16 seconds; 2G and 5 minutes leave
# ample startup and filesystem margin.
#SBATCH --job-name=plot-v22f-r2-gc500k
#SBATCH --account=dabanin
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/plot_v22f_r2_gc500k_%j.out
#SBATCH --error=logs/plot_v22f_r2_gc500k_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
source scripts/graphcast_env.sh

DATA_ROOT="plots/analyze_models/data/resolution_eval/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
JSON_ROOT="${DATA_ROOT}/v22_final_eval_json"
IMAGE_ROOT="plots/analyze_models/images/resolution_eval/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"

REQUIRED=(
  current_open_step50k
  current_closed_step50k
  current_open_swa42k_50k
  current_closed_swa42k_50k
  earlier_open_step16k
  earlier_closed_step16k
  vanilla_ar_k12_best_step24k
  vanilla_ar_k12_swa18k_26k
  source_vanilla_step500k
)
for label in "${REQUIRED[@]}"; do
  [[ -s "${JSON_ROOT}/${label}.json" ]] || { echo "Missing evaluation JSON: ${label}" >&2; exit 1; }
done

python -u scripts/analyze_models/plot_gc500k_resmamba_vs_partial_vanilla.py \
  --open-json "${JSON_ROOT}/current_open_step50k.json" \
  --closed-json "${JSON_ROOT}/current_closed_step50k.json" \
  --earlier-open-json "${JSON_ROOT}/earlier_open_step16k.json" \
  --earlier-closed-json "${JSON_ROOT}/earlier_closed_step16k.json" \
  --ar-json "${JSON_ROOT}/vanilla_ar_k12_best_step24k.json" \
  --vanilla-json "${JSON_ROOT}/source_vanilla_step500k.json" \
  --swa-open-json "${JSON_ROOT}/current_open_swa42k_50k.json" \
  --swa-closed-json "${JSON_ROOT}/current_closed_swa42k_50k.json" \
  --swa-ar-json "${JSON_ROOT}/vanilla_ar_k12_swa18k_26k.json" \
  --deepmind-csv "plots/analyze_models/data/resolution_eval/gcsmall_full_year_2022_res1grid_leads1_10d/resolution_eval.csv" \
  --output "${IMAGE_ROOT}/cold_rmse_2m_temperature_all_with_current_swa_v22_final.png" \
  --summary-csv "${DATA_ROOT}/cold_rmse_2m_temperature_all_with_current_swa_v22_final.csv"

echo "Rendered v22_final comparison plot."
