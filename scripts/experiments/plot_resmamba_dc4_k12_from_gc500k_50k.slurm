#!/usr/bin/env bash
# Render the existing absolute-RMSE overlay with the completed K=12 controls.
#SBATCH --job-name=plot-resm-k12-gc500k
#SBATCH --account=dabanin
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/plot_resm_k12_gc500k_%j.out
#SBATCH --error=logs/plot_resm_k12_gc500k_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
GC_CSV="plots/analyze_models/data/resolution_eval/res2_vanilla_gc_ar_k12_from_gc500k_50k_warm_leads1_9d/resolution_eval_graphcast_res2_ar_k12_from_gc500k.csv"
GC_VARIANT="vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps"

python -u scripts/analyze_models/plot_resmamba_dc4_swa_absolute_error.py \
  --comparison-csv plots/analyze_models/data/resolution_eval/res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d/resolution_eval.csv \
  --comparison-variant vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di16_ds16_20k_target_step12_bptt16 \
  --comparison-label "Frozen GC-Mamba di16/ds16 k=12 (res2, warm)" \
  --stateless-open-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps/eval/cold_bp_zero_from_gc500k/step50000.json" \
  --stateless-closed-json "${RUN_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps/eval/cold_full_zero_from_gc500k/step50000.json" \
  --ar-gc-csv "${GC_CSV}" --ar-gc-variant "${GC_VARIANT}" \
  --ar-gc-label "Vanilla GC AR k=12 from 500k (res2, warm)"
