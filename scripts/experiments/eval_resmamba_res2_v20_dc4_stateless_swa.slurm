#!/usr/bin/env bash
# Matched open-loop evaluation for the stateless d_conv=4 SWA(4k--12k).
#SBATCH --job-name=resm-r2-dc4-stat-swa
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --output=logs/resm_r2_dc4_stateless_swa_%j.out
#SBATCH --error=logs/resm_r2_dc4_stateless_swa_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

source scripts/graphcast_env.sh

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
RUN_DIR="artifacts/checkpoints/7_years/resmamba_res2_v20_dconv4_stateless_ablation_20k/resmamba_res2_v20_k14_di16_l2_dc4_open_stateless_20000steps"
CKPT="${RUN_DIR}/v13_residual_SWA_step4k-12k.pkl"
OUT_JSON="${RUN_DIR}/eval/cold_bp_zero_matched/swa_step4k-12k.json"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}"; exit 1; }
[[ -d "${EVAL_DATA}" ]] || { echo "Missing evaluation data: ${EVAL_DATA}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing SWA checkpoint: ${CKPT}"; exit 1; }
if [[ -s "${OUT_JSON}" ]]; then
  echo "Already evaluated: ${OUT_JSON}"
  exit 0
fi
mkdir -p "$(dirname "${OUT_JSON}")"

python -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "${CKPT}" \
  --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" \
  --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --no-temporal-stateful \
  --eval-mode cold_bp --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"

echo "Completed: ${OUT_JSON}"
