#!/usr/bin/env bash
# GCmamba_v2 smoke run.
# Same code path as Ilya's segments_training.py (--model gc_mamba) WITH our
# full-trunk-loss patch (loss accumulates over every bptt step instead of
# only the AR tail).
#
# Memory-safe params: matches Ilya's PROVEN gc_mamba production except:
#   - bptt_steps = 24 (v22-style, +50% time activations vs Ilya's 16)
#   - d_state    = 16 (v22-style; smaller than Ilya's 64 — net Mamba state shrinks)
#   - layers     = 1  (match Ilya, NOT v22's 2 — biggest OOM risk reduction)
#   - len_segment= 48 (compromise: 2 chunks/segment vs v22's 4)
# Net Mamba state: roughly Ilya's, time axis +50%. Should fit on 80 GiB.
#SBATCH --job-name=gcmamba-v2-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DATA_PATH=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
PREPARED=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream
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
  --out-dir "${OUT_DIR}" \
  --run-name gcmamba_v2_bptt24_K4_L1_ds16_smoke \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 4 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 100 --eval-every 50 --checkpoint-every 100 \
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

echo "gcmamba_v2 smoke (bptt=24, K=4, layers=1, d_state=16, full-trunk loss) done"
