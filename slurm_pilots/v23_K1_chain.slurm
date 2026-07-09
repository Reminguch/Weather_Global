#!/usr/bin/env bash
# v23 K=1 from-scratch chain — self-resubmits until step 20000 reached.
#
# Each invocation:
#   1. Detects the latest ckpt in results/v23/K1_fresh/.
#   2. If none -> start fresh (no --resume-from), --start-step 1.
#      If found -> resume from latest, --start-step = ckpt_step + 1.
#   3. Trains for up to ~7.5h (slurm time limit 8h, with 30min buffer).
#   4. After training: if last ckpt step >= 20000, exit. Otherwise resubmit self.
#
# Safety: aborts the chain if a run produces no NEW ckpt (i.e. crashed at/before
# the first ckpt-every interval). Prevents runaway resubmits on a broken model.
#
#SBATCH --job-name=v23-K1-chain
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_chain_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_chain_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR

# Anti-OOM-during-JIT-compile: disable JAX's eager preallocation of GPU memory.
# Default behavior (preallocate=true) makes JAX grab ~75% of GPU mem upfront in
# one contiguous block; if the node's GPU has fragmented residual memory from
# a previous user, JAX's grab succeeds but XLA's later compile-scratch alloc
# (~36 GiB peak in our case) fails. With preallocate=false, JAX allocates
# on-demand in smaller blocks, which fits better into fragmented memory.
# (Won't help if a SINGLE allocation truly needs 36 GiB contiguous.)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Anti-OOM-during-Triton-compile: force XLA to use cuBLAS GEMM instead of
# Triton nested-GEMM fusion. The Triton path peaks at ~36 GiB scratch buffer
# during compilation (observed 3x in jobs 8691699/8694234/8702678), which
# doesn't fit on cluster A100 80GB nodes after driver/system overhead.
# cuBLAS GEMM uses a small fixed scratch; expected ~5-10% slower training.
# Only triggered by `d_conv=8 + resume-from-ckpt`; fresh-init runs were fine.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

echo "[env] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "[env] XLA_FLAGS=${XLA_FLAGS}"
echo "[env] nvidia-smi at job start:"
nvidia-smi || true

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/train_mz_v20.py
THIS_SLURM=/home/lm8598/Weather_Global_experiments/slurm_pilots/v23_K1_chain.slurm
cd /home/lm8598/Weather_Global_experiments

# ----- run config (matches v22 K=1 except no --resume-from and d_conv=8) -----
CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23
RUN_NAME=K1_fresh_20k
# Extended target: 23000 steps to match v22's effective K=0 training budget.
# v22 inherited 20k from v15 + did 3k more in v22 K=1 = 23k total K=0 steps.
# v23 was originally 20k from scratch; now extended to 23k for apples-to-apples.
TARGET_STEPS=23000

RUN_DIR="${OUT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

# ----- detect latest ckpt (None on first invocation) -----
get_latest_step() {
  local latest_ckpt
  latest_ckpt=$(ls -1 "${RUN_DIR}"/v13_residual_step*.pkl 2>/dev/null | sort -V | tail -1)
  if [ -z "${latest_ckpt}" ]; then
    echo 0
  else
    echo "${latest_ckpt}" | grep -oE 'step[0-9]+' | grep -oE '[0-9]+' | tail -1
  fi
}

PREV_STEP=$(get_latest_step)
if [ "${PREV_STEP}" -eq 0 ]; then
  RESUME_ARGS=""
  START_STEP=1
  echo "[chain] fresh start (no ckpt found in ${RUN_DIR})"
else
  RESUME_CKPT="${RUN_DIR}/v13_residual_step${PREV_STEP}.pkl"
  RESUME_ARGS="--resume-from ${RESUME_CKPT}"
  START_STEP=$((PREV_STEP + 1))
  echo "[chain] resuming from ${RESUME_CKPT} at step ${START_STEP}"
fi

if [ "${START_STEP}" -gt "${TARGET_STEPS}" ]; then
  echo "[chain] already past target ${TARGET_STEPS} (prev_step=${PREV_STEP}). Exiting chain."
  exit 0
fi

# ----- run training -----
${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name "${RUN_NAME}" \
  ${RESUME_ARGS} \
  --start-step ${START_STEP} \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps ${TARGET_STEPS} --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 23 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 0 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 2 || true
# `|| true` so we can inspect progress even if training hit slurm time limit (SIGTERM -> non-zero).

# ----- decide whether to resubmit -----
NEW_STEP=$(get_latest_step)
echo "[chain] this run: prev_step=${PREV_STEP} -> new_step=${NEW_STEP}, target=${TARGET_STEPS}"

if [ "${NEW_STEP}" -ge "${TARGET_STEPS}" ]; then
  echo "[chain] reached target ${TARGET_STEPS}. Chain complete."
  exit 0
fi

if [ "${NEW_STEP}" -le "${PREV_STEP}" ]; then
  echo "[chain] no progress made (${PREV_STEP} -> ${NEW_STEP}). Crashed early."
  exit 1
fi

echo "[chain] partial progress to step ${NEW_STEP}. (Watcher will resubmit if needed; self-resubmit doesn't survive slurm TIMEOUT.)"
# Note: the prior version called `sbatch ${THIS_SLURM}` here. But slurm sends
# SIGTERM at the time limit, killing this bash before it can sbatch. Instead,
# an external watcher (in the user's session) detects when this job ends and
# resubmits if step < TARGET_STEPS.
exit 0
