#!/usr/bin/env bash
# Matched cold-rollout evaluation of the best available partial vanilla AR-k12
# checkpoint on the same 32 anchors and 40 six-hour leads as residual-Mamba.
#SBATCH --job-name=gc-r2-ar-k12-24k-match
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=logs/gc_r2_ar_k12_24k_matched_%j.out
#SBATCH --error=logs/gc_r2_ar_k12_24k_matched_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

VANILLA_CKPT="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps/ckpt_step24000.npz"
AUX_RESIDUAL_CKPT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps/v13_residual_step50000.pkl"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
OUT_JSON="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps/eval/cold_bp_matched_resmamba_samples/step24000.json"

for required in "$VANILLA_CKPT" "$AUX_RESIDUAL_CKPT" "$EVAL_DATA"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing matched-eval input: $required" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$OUT_JSON")"

python -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "$AUX_RESIDUAL_CKPT" --ckpt-in "$VANILLA_CKPT" \
  --data-path "$EVAL_DATA" --stats-dir data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 --input-duration 12h \
  --temporal-location mesh_processor_interleaved --temporal-hidden-size 128 \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --no-temporal-stateful --eval-mode cold_bp --residual-state-init zero --residual-alpha 0 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 --out-json "$OUT_JSON"

echo "Completed matched vanilla evaluation: $OUT_JSON"
