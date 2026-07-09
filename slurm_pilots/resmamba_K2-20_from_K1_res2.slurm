#!/usr/bin/env bash
# res_mamba @ res=2 K-curriculum (fan-out from K=1 best @ step 20000).
# K=2..20 each trained for 20k MORE steps (max_step=40000).
# Setup: frozen GCv1 K=3 baseline + GC2 residual head (g2m + m2g frozen, proc + Mamba trainable).
# closed-loop via --residual-ar-feedback baseline_plus_residual.
#SBATCH --job-name=resmamba-res2-K-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=14:00:00
#SBATCH --array=2-20
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_res2_K-sweep_K%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_res2_K-sweep_K%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

K=${SLURM_ARRAY_TASK_ID}
ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RESMAMBA_K1_BEST=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K1_closed_20k/ckpt_step20000.npz
RUN_NAME=resmamba_GCv1K3_res2_K${K}_from_K1step20k
OUT_DIR=${ROOT}/results/resmamba_GCv1K3

[ -f "${GCV1_K3}" ] || { echo "ERR: GCv1 K=3 baseline ckpt not found"; exit 1; }
[ -f "${RESMAMBA_K1_BEST}" ] || { echo "ERR: res_mamba K=1 best ckpt not found"; exit 1; }
echo "=== res=2 res_mamba K=${K} resume from K=1 best @ step 20000 ==="

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --baseline-ckpt "${GCV1_K3}" \
  --resume-ckpt "${RESMAMBA_K1_BEST}" \
  --resume-step 20000 \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 2 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps ${K} --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 40000 --eval-every 1000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --residual-ar-feedback baseline_plus_residual \
  --residual-output-head enabled \
  --trainable-part proc_and_mamba \
  --memory-mode optimal \
  --seed 22

echo "res=2 res_mamba K=${K} closed-loop (20k from K1step20k) done"
