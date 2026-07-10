#!/usr/bin/env bash
# v22 res=2 K-scan with inner1024 config (SWA-winner from K=18 arch-screen).
# 7 K values × 10000 steps each. K=18 already trained.
#
# Config (from K=18 arch-screen winner):
#   d_inner=1024, d_state=16, layers=2, d_conv=4  (Mamba S6 with 8x width)
#
# Same base ckpt: GCv1 K=3 cascade res=2 mesh=4 width=512 mp=6 @ step 52000.
#SBATCH --job-name=v22cl-res2-inner1024-kscan
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_i1024_kscan_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_i1024_kscan_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
[ -f "${GCV1_K3}" ] || { echo "ERR: missing GCv1 K3 ckpt"; exit 1; }

# K values (excluding K=18 which is already done)
KS=(2 4 8 12 16 20 22)

I=$SLURM_ARRAY_TASK_ID
K=${KS[$I]}

OUT_PARENT=${ROOT}/results/v22_res2_closedloop
OUT=${OUT_PARENT}/K${K}_res2_inner1024
mkdir -p ${OUT}

# nvidia-smi peak-mem tracker
(while true; do
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv >> ${OUT}/nvsmi.log 2>&1 || true
  sleep 60
done) &
NVSMI_PID=$!

echo "=== v22cl res=2 inner1024 K-scan  K=${K}  d_inner=1024 d_state=16 layers=2 d_conv=4 ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT_PARENT}" \
  --run-name K${K}_res2_inner1024 \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 10000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${K} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 1024 \
  --temporal-d-inner 1024 \
  --temporal-d-state 16 \
  --temporal-d-conv 4 \
  --temporal-dt-rank auto \
  --temporal-layers 2

kill $NVSMI_PID 2>/dev/null || true
echo "=== inner1024 K=${K} done ==="
