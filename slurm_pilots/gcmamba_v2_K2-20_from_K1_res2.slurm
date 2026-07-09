#!/usr/bin/env bash
# gc_mamba @ res=2 K-curriculum (fan-out from K=1 best ckpt @ step 19000).
# K=2, 3, ..., 20 each trained for 20k MORE steps (max_step=39000).
# Backbone: GCv1 K=3 frozen (res=2, mp=6, mesh=4, w=512).
# Mamba: post_processor insert=1 layers=1 d_state=16 d_inner=128 — matches K=1.
#SBATCH --job-name=gcmv2-res2-K-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --array=2-20
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmv2_res2_K-sweep_K%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmv2_res2_K-sweep_K%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

K=${SLURM_ARRAY_TASK_ID}
ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
K1_BEST=${ROOT}/results/gcmamba_v2/gcmamba_v2_GCv1K3_res2_K1_20k/ckpt_step19000.npz
RUN_NAME=gcmamba_v2_res2_K${K}_from_K1step19k
OUT_DIR=${ROOT}/results/gcmamba_v2

[ -f "${K1_BEST}" ] || { echo "ERR: K=1 best ckpt not found"; exit 1; }
echo "=== res=2 gc_mamba K=${K} starting from K=1 best @ step 19000 ==="

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --ckpt-in "${K1_BEST}" \
  --resume-step 19000 \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 6 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps ${K} --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 39000 --eval-every 1000 --checkpoint-every 1000 \
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

echo "res=2 gc_mamba K=${K} (20k from K1step19k) done"
