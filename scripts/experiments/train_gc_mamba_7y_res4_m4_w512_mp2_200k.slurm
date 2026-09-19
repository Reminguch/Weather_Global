#!/usr/bin/env bash
# Fresh, all-parameter segment-training comparison at the same
# res4/mesh4/width512/mp2 spatial shape as the vanilla baseline. Array 0 is a
# no-temporal segment-trained GC control; array 1 is di16; array 2 is di64.
# Both run 200k optimizer updates with batch8 x BPTT6 = 48 supervised forecast
# losses/update. Update 33,334 is the approximate exposure match to the vanilla
# baseline's 200k x batch8 = 1.6M supervised forecast losses.
# The closest res4/mesh3 di64 pilot averaged 0.374 s/update and ~19.8 GiB RSS;
# 36h/40G allows for the larger mesh4 graph, full evaluations, and I/O.
#SBATCH --job-name=gcm-r4-m4-w512-mp2-200k
#SBATCH --account=dabanin
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/gcm_r4_m4_w512_mp2_200k_%A_%a.out
#SBATCH --error=logs/gcm_r4_m4_w512_mp2_200k_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-gcm-r4-mp2-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

source scripts/graphcast_env.sh

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" != "0" && "${TASK_ID}" != "1" && "${TASK_ID}" != "2" ]]; then
  echo "SLURM_ARRAY_TASK_ID must be 0, 1, or 2, got ${TASK_ID}" >&2
  exit 1
fi

SOURCE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/7_years/gc_mamba_res4_m4_w512_mp2_200k_20260916}"
MAX_STEPS="${MAX_STEPS:-200000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
RUN_SUFFIX="${RUN_SUFFIX:-200k}"

if [[ "${TASK_ID}" == "0" ]]; then
  RUN_NAME="segment_gc_7y_res4_m4_w512_mp2_ls30_bptt6_${RUN_SUFFIX}"
  TEMPORAL_ARGS=(--temporal-backbone none)
else
  D_INNERS=(unused 16 64)
  D_INNER="${D_INNERS[${TASK_ID}]}"
  RUN_NAME="gc_mamba_7y_res4_m4_w512_mp2_di${D_INNER}_ds16_l1_ls30_bptt6_${RUN_SUFFIX}"
  TEMPORAL_ARGS=(
    --temporal-backbone mamba
    --temporal-location mesh_processor_interleaved
    --temporal-d-inner "${D_INNER}"
    --temporal-d-state 16
    --temporal-d-conv 4
    --temporal-dt-rank auto
    --temporal-layers 1
    --temporal-dropout 0.0
    --temporal-stateful
  )
fi

for required in "${SOURCE_CKPT}" \
  data/graphcast/graphcast/dataset/prepared_stream/res4/metadata.json; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

if [[ -e "${OUT_DIR}/${RUN_NAME}/ckpt_step${MAX_STEPS}.npz" ]]; then
  echo "Refusing to overwrite completed run: ${OUT_DIR}/${RUN_NAME}" >&2
  exit 1
fi

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr
  --data-source prepared_array
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream
  --stats-dir data/graphcast/graphcast/stats
  --ckpt-in "${SOURCE_CKPT}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year 2022
  --train-start-year 2015
  --train-end-year 2021
  --resolution 4.0
  --mesh-size 4
  --width 512
  --processor-msg-steps 2
  --input-duration 12h
  --target-steps 1
  --batch-size 8
  --eval-batch-size 4
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr 1e-4
  --weight-decay 1e-4
  --precision bf16
  --seed 0
  --len-segment 30
  --bptt-steps 6
  --chunk-load-workers 6
  --segment-prefetch-depth 2
  --data-cache-mode auto
  --data-cache-max-gib 48
  --batch-builder prepared_array
  --trainable-part all
  "${TEMPORAL_ARGS[@]}"
)

echo "Fresh segment run: run=${RUN_NAME} max_steps=${MAX_STEPS}."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
