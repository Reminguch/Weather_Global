#!/usr/bin/env bash
# Build and evaluate late-checkpoint SWA models on the matched cold rollout.
# Tasks: open residual-Mamba, closed-SG residual-Mamba, vanilla AR-k12.
#SBATCH --job-name=swa-r2-gc500k-ar-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/swa_r2_gc500k_ar_eval_%A_%a.out
#SBATCH --error=logs/swa_r2_gc500k_ar_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RESIDUAL_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
AR_DIR="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps"
BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
AUX_RESIDUAL_CKPT="${RESIDUAL_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps/v13_residual_step50000.pkl"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
SWA_STEPS=(42000 44000 46000 48000 50000)
AR_SWA_STEPS=(18000 20000 22000 24000 26000)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODE="cold_bp"
RESIDUAL_ALPHA_ARGS=()

case "$TASK_ID" in
  0)
    RUN_DIR="${RESIDUAL_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps"
    SWA_CKPT="${RUN_DIR}/v13_residual_SWA_step42k-50k.pkl"
    OUT_JSON="${RUN_DIR}/eval/cold_bp_zero_from_gc500k/swa_step42k-50k.json"
    INPUTS=()
    for step in "${SWA_STEPS[@]}"; do INPUTS+=("${RUN_DIR}/v13_residual_step${step}.pkl"); done
    python -u scripts/training/full_mamba_v23/build_swa_checkpoints.py \
      --checkpoint-type residual --inputs "${INPUTS[@]}" --output "$SWA_CKPT" \
      --source-steps "${SWA_STEPS[@]}"
    CKPT="$SWA_CKPT"
    CKPT_IN="$BASELINE_CKPT"
    ;;
  1)
    RUN_DIR="${RESIDUAL_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps"
    SWA_CKPT="${RUN_DIR}/v13_residual_SWA_step42k-50k.pkl"
    OUT_JSON="${RUN_DIR}/eval/cold_full_zero_from_gc500k/swa_step42k-50k.json"
    MODE="cold_full"
    INPUTS=()
    for step in "${SWA_STEPS[@]}"; do INPUTS+=("${RUN_DIR}/v13_residual_step${step}.pkl"); done
    python -u scripts/training/full_mamba_v23/build_swa_checkpoints.py \
      --checkpoint-type residual --inputs "${INPUTS[@]}" --output "$SWA_CKPT" \
      --source-steps "${SWA_STEPS[@]}"
    CKPT="$SWA_CKPT"
    CKPT_IN="$BASELINE_CKPT"
    ;;
  2)
    SWA_CKPT="${AR_DIR}/ckpt_SWA_step18k-26k.npz"
    OUT_JSON="${AR_DIR}/eval/cold_bp_matched_resmamba_samples/swa_step18k-26k.json"
    INPUTS=()
    for step in "${AR_SWA_STEPS[@]}"; do INPUTS+=("${AR_DIR}/ckpt_step${step}.npz"); done
    python -u scripts/training/full_mamba_v23/build_swa_checkpoints.py \
      --checkpoint-type graphcast --inputs "${INPUTS[@]}" --output "$SWA_CKPT" \
      --source-steps "${AR_SWA_STEPS[@]}"
    CKPT="$AUX_RESIDUAL_CKPT"
    CKPT_IN="$SWA_CKPT"
    RESIDUAL_ALPHA_ARGS=(--residual-alpha 0)
    ;;
  *)
    echo "Unsupported array task: $TASK_ID" >&2
    exit 2
    ;;
esac

for required in "$CKPT" "$CKPT_IN" "$EVAL_DATA"; do
  [[ -e "$required" ]] || { echo "Missing input: $required" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUT_JSON")"

python -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "$CKPT" --ckpt-in "$CKPT_IN" --data-path "$EVAL_DATA" --stats-dir data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 --input-duration 12h \
  --temporal-location mesh_processor_interleaved --temporal-hidden-size 128 \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --no-temporal-stateful --eval-mode "$MODE" --residual-state-init zero "${RESIDUAL_ALPHA_ARGS[@]}" \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 --out-json "$OUT_JSON"

echo "Completed SWA evaluation: $OUT_JSON"
