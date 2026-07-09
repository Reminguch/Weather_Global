#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

source scripts/graphcast_env.sh

run_one() {
  local run_name="$1"
  local input_duration="$2"
  local temporal_backbone="$3"

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
    --max-steps 100 \
    --eval-every 100 \
    --eval-batch-size 32 \
    --checkpoint-every 100 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed 0 \
    --precision bf16 \
    --input-duration "${input_duration}" \
    --temporal-backbone "${temporal_backbone}" \
    --out-dir artifacts/checkpoints/graphcast_2020_2021_val2022 \
    --run-name "${run_name}"
}

run_one "cmp100_hist2_none" "12h" "none"
run_one "cmp100_hist2_mamba" "12h" "mamba"
run_one "cmp100_hist4_none" "24h" "none"
run_one "cmp100_hist4_mamba" "24h" "mamba"
