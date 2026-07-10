#!/usr/bin/env bash
# res=1 d_inner=16 K-scan RETRAIN with the CORRECT (clean, Design-A) code.
# Replaces the void drifted-tree d_inner=16 series (bad input normalization — see
# incident 2026-07-09). Protocol mirrors the H=128 res=1 K-scan exactly:
# fresh from DM GC-small, closed_loop_sg, 20k steps, seed 22; the ONLY change
# vs H=128 is temporal-hidden-size = 128, d-inner = 16 (downward ablation). Fair d_inner ablation.
#SBATCH --job-name=v22cl-r1-inner16-CLEAN
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=44:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/v22cl_r1_inner16_CLEAN_t%a_%A.out
#SBATCH --error=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/v22cl_r1_inner16_CLEAN_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen   # frozen CLEAN tree; RELATIVE script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
[ -f "${DM_CKPT}" ] || { echo "ERR: missing DM ckpt"; exit 1; }

OUT=${ROOT}/results/v22_res1_closedloop_inner16_clean
mkdir -p ${OUT}

K_LABELS=(4 8 12 16 20 22)
K=${K_LABELS[$SLURM_ARRAY_TASK_ID]}

echo "=== v22cl res=1 d_inner=16 CLEAN K=${K} fresh-from-DM 0->20000 seed 22 ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K${K}_inner16_clean_fresh_dmckpt \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${K} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 16 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "=== K=${K} inner16 clean done ==="
