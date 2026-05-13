#!/bin/bash
set -euo pipefail

SCRIPT="scripts/experiments/7y_mp6_mamba_frozen_sweep_20k.slurm"
COMBOS_PER_BASE=18
START_BASE=${START_BASE:-0}
END_BASE=${END_BASE:-8}

submit_base() {
  local base_idx="$1"
  local time_limit="$2"
  local mem_gb="$3"
  if (( base_idx < START_BASE || base_idx > END_BASE )); then
    return
  fi
  local start=$((base_idx * COMBOS_PER_BASE))
  local end=$((start + COMBOS_PER_BASE - 1))
  sbatch --array="${start}-${end}" --time="${time_limit}" --mem="${mem_gb}G" "${SCRIPT}"
}

# Sorted mp6 baselines:
# 0 res15_m2, 1 res18_m2, 2 res2_m4, 3 res3_m4, 4 res4_m3,
# 5 res4_m4, 6 res6_m3, 7 res9_m2, 8 res9_m3.
#
# Walltime is scaled from vanilla step-time logs plus margin for Mamba/residual
# overhead. --mem is CPU RAM, not A100 GPU memory; vanilla logs showed GPU use
# below an 80G A100, while CPU use is expected to stay modest because these
# runs inherit data_cache_mode=never from the vanilla configs.
submit_base 0 "01:02:00" 24
submit_base 1 "01:02:00" 24
submit_base 2 "04:00:00" 64
submit_base 3 "03:00:00" 48
submit_base 4 "02:00:00" 32
submit_base 5 "03:00:00" 48
submit_base 6 "02:00:00" 32
submit_base 7 "01:30:00" 24
submit_base 8 "01:30:00" 24
