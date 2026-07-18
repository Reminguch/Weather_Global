#!/usr/bin/env bash
# REGRESSION (res=2): train K18 FRESH-from-GCv1K3 for 10 steps in the frozen-clean
# tree and compare steps 1..10 to the original K18_res2_closedloop_sg_fresh train_log.
# res=2 runs start at step 1 (no resume), so this is a direct fresh-start check.
# Match => (up to bf16 nondeterminism) the res=2 data pipeline + trainer reproduce.
#SBATCH --job-name=res2-regcheck
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res2_regcheck_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res2_regcheck_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_res2/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
OUT=${ROOT}/results/res2_sync/regcheck
mkdir -p ${OUT}

echo "=== REGRESSION res=2: K18 fresh 1..10 (seed 22) ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" --run-name K18_res2_regcheck \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 --batch-size 1 \
  --max-steps 10 --checkpoint-every 10 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2
echo "=== res=2 regcheck done ==="
