#!/bin/bash
#SBATCH --job-name=res1-gcm-train-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=280G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/res1_gc_mamba_training_eval_compare_%A_%a.out
#SBATCH --error=logs/res1_gc_mamba_training_eval_compare_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

TARGET_STEPS_SET=(1 4 8)
MODES=(identity trained)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODE_IDX=$((TASK_ID / ${#TARGET_STEPS_SET[@]}))
TARGET_IDX=$((TASK_ID % ${#TARGET_STEPS_SET[@]}))

if (( MODE_IDX < 0 || MODE_IDX >= ${#MODES[@]} )); then
  echo "Task ${TASK_ID} outside expected range 0-5" >&2
  exit 1
fi

MODE="${MODES[$MODE_IDX]}"
TARGET_STEPS="${TARGET_STEPS_SET[$TARGET_IDX]}"

OFFICIAL_CKPT=${OFFICIAL_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
SOURCE_ROOT=${SOURCE_ROOT:-artifacts/checkpoints/res1_official/gc_mamba_bptt16_k_sweep}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt16_k_sweep_training_eval_compare}

case "${TARGET_STEPS}" in
  1)
    TRAINED_RUN="official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k1_bptt16_bs1_optimal"
    ;;
  4)
    TRAINED_RUN="official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k4_bptt16_bs1_optimal"
    ;;
  8)
    TRAINED_RUN="official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k8_bptt16_bs1_optimal"
    ;;
  *)
    echo "Unsupported TARGET_STEPS=${TARGET_STEPS}" >&2
    exit 1
    ;;
esac

RUN_NAME="official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k${TARGET_STEPS}_bptt16_bs1_${MODE}_training_eval"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr
  --data-source prepared_array
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream
  --stats-dir data/graphcast/graphcast/stats
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year 2022
  --train-start-year 2015
  --train-end-year 2021
  --resolution 1.0
  --mesh-size 5
  --width 512
  --processor-msg-steps 16
  --input-duration 12h
  --target-steps "${TARGET_STEPS}"
  --len-segment 32
  --bptt-steps 16
  --chunk-load-workers 6
  --batch-size 1
  --eval-batch-size 1
  --eval-num-segments 16
  --final-eval-num-segments 16
  --no-eval-rotating-diagnostics
  --data-cache-mode never
  --data-cache-max-gib 48
  --batch-builder prepared_array
  --max-steps 0
  --eval-every 999999
  --checkpoint-every 999999
  --lr 1e-4
  --weight-decay 1e-4
  --precision bf16
  --seed 0
  --memory-mode optimal
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner 128
  --temporal-d-state 64
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers 1
  --temporal-dropout 0.0
  --temporal-insert-count 2
  --trainable-part mamba
  --zero-init-temporal-out
)

if [[ "${MODE}" == "identity" ]]; then
  CMD+=(--ckpt-in "${OFFICIAL_CKPT}" --init-from-graphcast-ckpt "${OFFICIAL_CKPT}")
else
  TRAINED_CKPT="${SOURCE_ROOT}/${TRAINED_RUN}/ckpt_best.npz"
  if [[ ! -f "${TRAINED_CKPT}" ]]; then
    echo "Missing trained checkpoint: ${TRAINED_CKPT}" >&2
    exit 1
  fi
  CMD+=(--ckpt-in "${TRAINED_CKPT}" --resume-step 0)
fi

echo "Mode=${MODE} target_steps=${TARGET_STEPS}"
echo "Run name=${RUN_NAME}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
