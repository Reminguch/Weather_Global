#!/usr/bin/env bash
# GCmamba_v2 with the new "mesh_post_processor" insertion (Mamba AFTER the
# full 16-step GraphCast processor, before mesh2grid/decoder).
#
# Rationale (vs the interleaved variant we're running in 9635421):
#   - GraphCast spatial reasoning runs to completion before any temporal
#     correction is applied. Frozen processor sees clean encoder latent.
#   - Mamba only sees the post-processor mesh latent and outputs a
#     temporal correction; mesh2grid + decoder see (post-processor + Mamba).
#   - Less risk of Mamba's early-training distribution shift propagating
#     through the frozen processor.
#
# Same Mamba block sizing as the interleaved sanity:
#   layers=1, d_inner=128, d_state=16, d_conv=4, stateful, insert_count=1.
#
# Sanity scope: max_steps=5, K=1, bptt=24, GC overlay via
# --init-from-graphcast-ckpt, prepared_stream_2015_2022_v2 (full 7yr data).
#SBATCH --job-name=gcmv2-postproc-K1-sanity
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_postproc_K1_sanity_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_postproc_K1_sanity_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DATA_PATH=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
PREPARED=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2
STATS=${ROOT}/data/graphcast/graphcast/stats
CKPT_IN="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"

OUT_DIR=${ROOT}/results/gcmamba_v2
mkdir -p ${OUT_DIR}

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --ckpt-in "${CKPT_IN}" \
  --init-from-graphcast-ckpt "${CKPT_IN}" \
  --out-dir "${OUT_DIR}" \
  --run-name gcmamba_v2_postproc_K1_sanity_5step \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 1 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 1 --final-eval-num-segments 1 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 5 --eval-every 5 --checkpoint-every 5 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 1 \
  --zero-init-temporal-out \
  --trainable-part mamba \
  --memory-mode optimal \
  --seed 22

echo "gcmamba_v2 post-processor K=1 sanity (5 steps) done"
