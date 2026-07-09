#!/usr/bin/env bash
# Resume from step 2000 ckpt and continue to step 20000.
# 18k more steps × 2.68 sec/step ≈ 13.4 h training + ~3 h evals = ~16-17 h.
#
# Key changes from 2k run:
#   - --resume-step 2000  (resume from prev ckpt)
#   - --ckpt-in <step2000>  (load prev params)
#   - REMOVED --init-from-graphcast-ckpt  (mutually exclusive with resume-step)
#   - --max-steps 20000
#   - --checkpoint-every 1000  (so we have 18 intermediate ckpts for ablation)
#   - GCMAMBA_PRUNE_CKPTS env not set (patched prune disabled, all kept)
#SBATCH --job-name=gcmamba-v2-K1-20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=23:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_20k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_20k_%j.err
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
RUN_DIR=${ROOT}/results/gcmamba_v2/gcmamba_v2_postproc_K1_2k
RESUME_CKPT=${RUN_DIR}/ckpt_step2000.npz

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: resume ckpt not found at ${RESUME_CKPT}" >&2
    exit 1
fi

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --ckpt-in "${RESUME_CKPT}" \
  --resume-step 2000 \
  --out-dir "${ROOT}/results/gcmamba_v2" \
  --run-name gcmamba_v2_postproc_K1_2k \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 1 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 20000 --eval-every 1000 --checkpoint-every 1000 \
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

echo "gcmamba_v2 K=1 resumed to 20000 step done"
