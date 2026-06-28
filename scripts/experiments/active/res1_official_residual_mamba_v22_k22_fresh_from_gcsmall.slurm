#!/bin/bash
#SBATCH --job-name=my-v22-k22-fresh
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/my_v22_k22_fresh_%j.out
#SBATCH --error=logs/my_v22_k22_fresh_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"

source scripts/graphcast_env.sh

BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT_DIR="artifacts/checkpoints/res1_official/residual_mamba_v22setup_k1_then_k22"
RUN_NAME="my_v22setup_k22_fresh_from_gcsmall"

echo "Fresh residual Mamba v22 k22 run"
echo "Run directory: ${OUT_DIR}/${RUN_NAME}"
echo "Frozen baseline checkpoint: ${BASELINE_CKPT}"
echo "Trainable init: fresh"

python -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr \
  --data-source prepared_array \
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir data/graphcast/graphcast/stats \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --val-year 2022 \
  --train-start-year 2015 \
  --train-end-year 2021 \
  --resolution 1.0 \
  --mesh-size 5 \
  --width 512 \
  --processor-msg-steps 2 \
  --input-duration 12h \
  --target-steps 22 \
  --len-segment 96 \
  --bptt-steps 24 \
  --autoregressive-loss-mode tail_uniform \
  --chunk-load-workers 6 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --eval-num-segments 16 \
  --final-eval-num-segments all \
  --eval-subset-policy stratified_fixed \
  --checkpoint-every 500 \
  --max-steps 26000 \
  --eval-every 2000 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 22 \
  --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 \
  --temporal-d-state 16 \
  --temporal-d-conv 4 \
  --temporal-dt-rank auto \
  --temporal-layers 2 \
  --temporal-dropout 0.0 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --memory-mode optimal \
  --residual-output-head enabled \
  --data-cache-mode never \
  --data-cache-max-gib 48 \
  --batch-builder prepared_array \
  --training-target residual \
  --residual-ar-feedback baseline_plus_residual \
  --baseline-ckpt "${BASELINE_CKPT}"
