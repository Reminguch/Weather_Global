#!/usr/bin/env bash
# Periodic-SWA restart experiment, res2 / GC500k / K12.
#
# Each array task starts from the best matched SWA(4k--12k), trains 8k updates,
# averages its four 2k-spaced checkpoints, warm-starts from that average for a
# further 8k, repeats the averaging, then trains the final 4k (20k new updates
# total).  These are intentional SWA warm restarts: optimizer, RNG, and cursor
# reset at phase boundaries.
#
# Budget basis: matching di16/di64 full-Mamba 20k jobs 12332071/12336275 took
# 9:47--10:12 at 40G.  48G gives memory headroom; 14h covers phase overhead.
#SBATCH --job-name=v22f-r2-swa-stab
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v22f_r2_swa_stabilized_%A_%a.out
#SBATCH --error=logs/v22f_r2_swa_stabilized_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIGS=(
  configs/experiments/v22_final/res2_gc500k_k12_di16_closed_sg_stateful_20k.json
  configs/experiments/v22_final/res2_gc500k_k12_di64_bcg4_closed_sg_stateful_20k.json
)
VARIANTS=(di16 di64_bcg4)
INITIAL_SWAS=(
  artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/swa/swa_step04000-12000.pkl
  artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di64_bcg4_closed_sg_stateful_20k/swa/swa_step04000-12000.pkl
)

INDEX="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${INDEX}]}"
VARIANT="${VARIANTS[${INDEX}]}"
INITIAL_SWA="${INITIAL_SWAS[${INDEX}]}"
OUTPUT_ROOT="artifacts/checkpoints/v22_final/res2_gc500k_k12_swa_stabilized_20260814"
PHASE1="${VARIANT}_swa_restart_phase1_8k"
PHASE2="${VARIANT}_swa_restart_phase2_8k"
PHASE3="${VARIANT}_swa_restart_phase3_4k"

export MPLCONFIGDIR="/tmp/mpl-v22f-r2-swa-stab-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

[[ -f "${INITIAL_SWA}" ]] || { echo "Missing initial SWA: ${INITIAL_SWA}" >&2; exit 1; }

echo "variant=${VARIANT} config=${CONFIG} initial_swa=${INITIAL_SWA}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum "${CONFIG}" scripts/training/train_v22_final.py \
  scripts/training/build_v22_final_swa.py \
  src/models/mamba/v22_final/training/runner.py \
  src/models/mamba/v22_final/checkpoint.py

run_phase() {
  local run_name="$1"
  local max_steps="$2"
  local init_from="$3"
  python -u scripts/training/train_v22_final.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    --max-steps "${max_steps}" \
    --checkpoint-every 2000 \
    --init-from "${init_from}"
}

build_phase_swa() {
  local run_name="$1"
  local output="$2"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  python -u scripts/training/build_v22_final_swa.py \
    --inputs \
      "${run_dir}/checkpoints/checkpoint_step00002000.pkl" \
      "${run_dir}/checkpoints/checkpoint_step00004000.pkl" \
      "${run_dir}/checkpoints/checkpoint_step00006000.pkl" \
      "${run_dir}/checkpoints/checkpoint_step00008000.pkl" \
    --source-steps 2000 4000 6000 8000 \
    --output "${output}"
}

run_phase "${PHASE1}" 8000 "${INITIAL_SWA}"
SWA1="${OUTPUT_ROOT}/${PHASE1}/swa/swa_step02000-08000.pkl"
build_phase_swa "${PHASE1}" "${SWA1}"

run_phase "${PHASE2}" 8000 "${SWA1}"
SWA2="${OUTPUT_ROOT}/${PHASE2}/swa/swa_step02000-08000.pkl"
build_phase_swa "${PHASE2}" "${SWA2}"

run_phase "${PHASE3}" 4000 "${SWA2}"

echo "Completed ${VARIANT}: final=${OUTPUT_ROOT}/${PHASE3}/checkpoints/checkpoint_step00004000.pkl"
