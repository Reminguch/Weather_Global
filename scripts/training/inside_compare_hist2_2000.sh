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
  local temporal_backbone="$2"

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
    --eval-batch-size 8 \
    --checkpoint-every 500 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed 0 \
    --precision bf16 \
    --input-duration 12h \
    --temporal-backbone "${temporal_backbone}" \
    --temporal-location mesh_processor_interleaved \
    --out-dir "${OUT_DIR}" \
    --run-name "${run_name}"

  for step in 500 1000 2000; do
    python scripts/analyze_models/mae_vs_lead.py \
      --model-group baseline \
      --baseline-ckpt "${OUT_DIR}/${run_name}/ckpt_step${step}.npz" \
      --dataset-dir "${EVAL_DATA}" \
      --stats-dir "${STATS_DIR}" \
      --n-eval-days 10 \
      --hours-per-step 6 \
      --n-input-steps 2 \
      --max-lead-steps 24 \
      --window-batch-size 8 \
      --output-dir "${PLOT_DIR}"

    cp "${PLOT_DIR}/nyc_mae_vs_lead_baseline.csv" \
      "${PLOT_DIR}/nyc_mae_vs_lead_${run_name}_step${step}.csv"
    cp "${PLOT_DIR}/nyc_mae_vs_lead_baseline.png" \
      "${PLOT_DIR}/nyc_mae_vs_lead_${run_name}_step${step}.png"
  done
}

run_one "cmp2000_inside_hist2_none" "none"
run_one "cmp2000_inside_hist2_mamba" "mamba"
