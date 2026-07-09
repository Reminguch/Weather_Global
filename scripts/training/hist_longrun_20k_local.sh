#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

source scripts/graphcast_env.sh

BASE_OUT="artifacts/checkpoints/graphcast_2020_2021_val2022"
DATA_PATH="data/graphcast/graphcast/dataset/wb2_res1_levels13_2020_2022.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr"
OUT_DIR="plots/analyze_models"
mkdir -p "${OUT_DIR}" logs

run_one() {
  local run_name="$1"
  local input_duration="$2"
  local n_input_steps="$3"
  echo "==== TRAIN ${run_name} (${input_duration}) from scratch -> 20000 ===="
  python -u scripts/training/train_graphcast.py \
    --data-path "${DATA_PATH}" \
    --val-year 2022 \
    --train-start-year 2020 \
    --train-end-year 2021 \
    --resolution 2.0 \
    --mesh-size 4 \
    --width 128 \
    --processor-msg-steps 1 \
    --batch-size 1 \
    --max-steps 20000 \
    --eval-every 2000 \
    --eval-batch-size 32 \
    --checkpoint-every 1000 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed 0 \
    --precision bf16 \
    --input-duration "${input_duration}" \
    --out-dir "${BASE_OUT}" \
    --run-name "${run_name}" \
    --stats-dir "${STATS_DIR}"

  for step in 10000 20000; do
    local ckpt="${BASE_OUT}/${run_name}/ckpt_step${step}.npz"
    if [[ ! -f "${ckpt}" ]]; then
      echo "Missing checkpoint for eval: ${ckpt}"
      exit 1
    fi

    echo "==== EVAL ${run_name} step${step} ===="
    python scripts/analyze_models/mae_vs_lead.py \
      --model-group baseline \
      --baseline-ckpt "${ckpt}" \
      --dataset-dir "${EVAL_DATA}" \
      --stats-dir "${STATS_DIR}" \
      --n-eval-days 10 \
      --hours-per-step 6 \
      --n-input-steps "${n_input_steps}" \
      --max-lead-steps 24 \
      --window-batch-size 8 \
      --output-dir "${OUT_DIR}"

    cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.csv" "${OUT_DIR}/nyc_mae_vs_lead_${run_name}_step${step}.csv"
    cp "${OUT_DIR}/nyc_mae_vs_lead_baseline.png" "${OUT_DIR}/nyc_mae_vs_lead_${run_name}_step${step}.png"
  done
}

run_one "gpu_20000steps_scratch_hist2" "12h" "2"
run_one "gpu_20000steps_scratch_hist4" "24h" "4"
run_one "gpu_20000steps_scratch_hist6" "36h" "6"
run_one "gpu_20000steps_scratch_hist8" "48h" "8"

echo "All long-run training/eval finished."
