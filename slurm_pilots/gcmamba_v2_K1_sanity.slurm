#!/usr/bin/env bash
# GCmamba_v2 minimal sanity (max_steps=5).
# Fixes 2 bugs in the prior K=1 attempt (job 9634854):
#   A) Missing --init-from-graphcast-ckpt → GC-Mamba was training from
#      RANDOM init instead of overlaying pretrained GraphCast small.
#      Confirmed: log lacked "Initialized GC-Mamba from vanilla GraphCast
#      checkpoint" line.  Ilya's official k-sweep slurm always passes BOTH
#      --ckpt-in (for config) AND --init-from-graphcast-ckpt (for params).
#   B) Wrong prepared_data_root: prepared_stream = 2020-2022 only (90 GB).
#      Switched to prepared_stream_2015_2022_v2 = 2015-2022 full (239 GB).
#
# Purpose: verify compile + 1 optimizer step + checkpoint save before
# committing to a longer run. K=1, bptt=24, layers=1, d_state=16.
#SBATCH --job-name=gcmamba-v2-K1-sanity
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_sanity_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_sanity_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DATA_PATH=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
PREPARED=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2     # fix B
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
  --run-name gcmamba_v2_K1_sanity_5step \
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
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --zero-init-temporal-out \
  --trainable-part mamba \
  --memory-mode optimal \
  --seed 22

echo "gcmamba_v2 K=1 sanity (5 steps, with GC overlay + full data) done"
