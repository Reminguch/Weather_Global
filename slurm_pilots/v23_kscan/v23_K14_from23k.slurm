#!/usr/bin/env bash
# v23 K=14 K-scan branch (from step23000) — resumes from v23 K=1 step23000 ckpt, trains 3000 steps with ar-tail-K=14.
# Single-knob ablation vs v22 K=22: --temporal-d-conv 8 (was 4). All other args identical.
# Total K=22 budget = step 23001 → 26000 (same as v22).
#
# REQUIRES: v23 K=1 chain has reached step 23000 (TARGET_STEPS=23000 in chain.slurm).
#
#SBATCH --job-name=v23-K14
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR

# Avoid OOM during JIT compile for resume runs — see slurm_pilots/v23_K1_chain.slurm for the full story.
# K=14 graph has bigger compile peak than K=1; also use platform allocator (Plan D).
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
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23/K1_fresh_20k/v13_residual_step23000.pkl

# Guard: require v23 K=1 step 23000 ckpt before starting K=14 K-scan branch (from step23000).
if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: required resume checkpoint does not exist: ${RESUME_CKPT}" 1>&2
  echo "K=1 chain has not yet reached step 23000. Check v23_K1_chain.slurm progress." 1>&2
  exit 1
fi
echo "[v23 K=14] resume ckpt verified: ${RESUME_CKPT}"

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K14_from_v23K1_23k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 23001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 26000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 23 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 14 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 2
