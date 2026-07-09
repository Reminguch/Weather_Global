#!/usr/bin/env bash
# v23 K=4 K-scan branch (from step20000) — EARLY VARIANT: resumes from v23 K=1 step20000 ckpt
# (3k less K=1 training than the canonical "from step23000" version).
#
# Purpose: get K=22 results sooner without waiting for K=1 extension to 23k.
# The canonical apples-to-apples-vs-v22 version (v23_K22.slurm) still launches
# separately from K=1 step 23000 once that ckpt appears.
#
# Output dir is distinct: K22_from_v23K1_20k (vs K22_from_v23K1_23k).
#
#SBATCH --job-name=v23-K4-20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_K22_20k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_K22_20k_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR

# Avoid OOM during JIT compile for resume runs — see chain.slurm for the full story.
# K=4 graph has bigger compile peak than K=1 (22 AR steps × bptt=24 activations);
# Triton-disable alone was not enough (job 8780755 OOMed at 35.83 GiB on della-l04g12).
# Plan D: also switch to platform allocator (more fragmentation-tolerant than BFC).
# If this still OOMs at ~36 GiB, escalate to --exclusive (single one-shot, cost ~15-35 GPU-hours).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
echo "[env] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "[env] XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR}"
echo "[env] XLA_FLAGS=${XLA_FLAGS}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/train_mz_v20.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23/K1_fresh_20k/v13_residual_step20000.pkl

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: missing resume ckpt: ${RESUME_CKPT}" 1>&2
  exit 1
fi
echo "[v23 K=4 from 20k] resume ckpt verified: ${RESUME_CKPT}"

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K4_from_v23K1_20k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 20001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 23000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 23 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 4 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 2
