#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

source scripts/graphcast_env.sh

OUT_DIR="artifacts/checkpoints/graphcast_2020_2021_val2022"
PLOT_DIR="plots/analyze_models"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

mkdir -p "${PLOT_DIR}"

run_one() {
  local run_name="$1"
  local input_duration="$2"
  local n_input_steps="$3"
  local temporal_backbone="$4"

  python -u scripts/training/train_graphcast.py \
    --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_2020_2022.zarr \
    --val-year 2022 \
    --train-start-year 2020 \
    --train-end-year 2021 \
    --resolution 2.0 \
    --mesh-size 4 \
    --width 128 \
    --processor-msg-steps 1 \
    --batch-size 1 \
    --max-steps 2000 \
    --eval-every 500 \
    --eval-batch-size 32 \
    --checkpoint-every 500 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed 0 \
    --precision bf16 \
    --input-duration "${input_duration}" \
    --temporal-backbone "${temporal_backbone}" \
    --out-dir "${OUT_DIR}" \
    --run-name "${run_name}"

  python scripts/analyze_models/mae_vs_lead.py \
    --model-group baseline \
    --baseline-ckpt "${OUT_DIR}/${run_name}/ckpt_step2000.npz" \
    --dataset-dir "${EVAL_DATA}" \
    --stats-dir "${STATS_DIR}" \
    --n-eval-days 10 \
    --hours-per-step 6 \
    --n-input-steps "${n_input_steps}" \
    --max-lead-steps 24 \
    --window-batch-size 8 \
    --output-dir "${PLOT_DIR}"

  cp "${PLOT_DIR}/nyc_mae_vs_lead_baseline.csv" "${PLOT_DIR}/nyc_mae_vs_lead_${run_name}.csv"
  cp "${PLOT_DIR}/nyc_mae_vs_lead_baseline.png" "${PLOT_DIR}/nyc_mae_vs_lead_${run_name}.png"
}

run_one "cmp2000_hist2_none" "12h" "2" "none"
run_one "cmp2000_hist2_mamba" "12h" "2" "mamba"
run_one "cmp2000_hist4_none" "24h" "4" "none"
run_one "cmp2000_hist4_mamba" "24h" "4" "mamba"
