#!/usr/bin/env bash
# Render the all-references matched cold-rollout plot after SWA evaluations.
#SBATCH --job-name=plot-swa-r2-gc500k
#SBATCH --account=dabanin
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/plot_swa_r2_gc500k_%j.out
#SBATCH --error=logs/plot_swa_r2_gc500k_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
AR_DIR="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps"
IMAGE_DIR="plots/analyze_models/images/resolution_eval/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
DATA_DIR="plots/analyze_models/data/resolution_eval/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"

python -u scripts/analyze_models/plot_gc500k_resmamba_vs_partial_vanilla.py \
  --open-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps/eval/cold_bp_zero_from_gc500k/step50000.json" \
  --closed-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps/eval/cold_full_zero_from_gc500k/step50000.json" \
  --earlier-open-json "artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k/resmamba_res2_v20_k14_di16_l2_dc4_open_20000steps/eval/cold_bp_zero_matched/step16000.json" \
  --earlier-closed-json "artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k/resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_20000steps/eval/cold_full_zero_matched/step16000.json" \
  --ar-json "${AR_DIR}/eval/cold_bp_matched_resmamba_samples/step24000.json" \
  --vanilla-json "artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/eval/cold_bp_matched_resmamba_samples/step500000.json" \
  --swa-open-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps/eval/cold_bp_zero_from_gc500k/swa_step42k-50k.json" \
  --swa-closed-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps/eval/cold_full_zero_from_gc500k/swa_step42k-50k.json" \
  --swa-ar-json "${AR_DIR}/eval/cold_bp_matched_resmamba_samples/swa_step18k-26k.json" \
  --deepmind-csv "plots/analyze_models/data/resolution_eval/gcsmall_full_year_2022_res1grid_leads1_10d/resolution_eval.csv" \
  --output "${IMAGE_DIR}/cold_rmse_2m_temperature_all_with_current_swa.png" \
  --summary-csv "${DATA_DIR}/cold_rmse_2m_temperature_all_with_current_swa.csv"
