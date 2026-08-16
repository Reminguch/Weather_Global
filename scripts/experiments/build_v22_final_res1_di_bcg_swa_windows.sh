#!/usr/bin/env bash
# Build rolling four-checkpoint SWAs for the carry-trained v22_final Res1 sweep.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813"
RUN_NAMES=(
  di16_bcg1_closed_sg_stateful_20k
  di16_bcg4_closed_sg_stateful_20k
  di32_bcg1_closed_sg_stateful_20k
  di32_bcg4_closed_sg_stateful_20k
)
WINDOW_STARTS=(2000 4000 6000 8000 10000)

for run_name in "${RUN_NAMES[@]}"; do
  run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}/swa"
  for start in "${WINDOW_STARTS[@]}"; do
    end=$((start + 6000))
    steps=("${start}" "$((start + 2000))" "$((start + 4000))" "${end}")
    inputs=()
    for step in "${steps[@]}"; do
      inputs+=("${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl")
    done
    output="${run_dir}/swa/swa_step$(printf '%05d' "${start}")-$(printf '%05d' "${end}").pkl"
    if [[ -e "${output}" ]]; then
      echo "Already exists, skipping: ${output}"
      continue
    fi
    python -u scripts/training/build_v22_final_swa.py \
      --inputs "${inputs[@]}" \
      --source-steps "${steps[@]}" \
      --output "${output}"
  done
done
