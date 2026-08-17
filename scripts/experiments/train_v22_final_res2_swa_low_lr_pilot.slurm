#!/usr/bin/env bash
# Res2 counterpart of the conservative SWA continuation pilot. Two tasks start
# from the best carry-state SWA(4k--12k), zero recurrent state, and train 2k
# updates at lower learning rates before averaging four 500-step checkpoints.
#
# Comparable res2 jobs used the full 40--48 GiB request and about 0.5 hours per
# 1k updates. 48G/2h leaves useful compilation and SWA-building headroom.
#SBATCH --job-name=v22f-r2-swa-low-lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v22f_r2_swa_low_lr_%A_%a.out
#SBATCH --error=logs/v22f_r2_swa_low_lr_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r2-swa-low-lr-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

CONFIG="configs/experiments/v22_final/res2_gc500k_k12_di16_closed_sg_stateful_20k.json"
INITIAL_SWA="artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/swa/swa_step04000-12000.pkl"
OUTPUT_ROOT="artifacts/checkpoints/v22_final/res2_swa_low_lr_continuation_20260816"
LEARNING_RATES=(1e-5 3e-5)
LR_TAGS=(lr1em5 lr3em5)

task_id="${SLURM_ARRAY_TASK_ID}"
learning_rate="${LEARNING_RATES[$task_id]}"
lr_tag="${LR_TAGS[$task_id]}"
run_name="di16_swa4k12k_${lr_tag}_2k"
run_dir="${OUTPUT_ROOT}/${run_name}"

for required in "${CONFIG}" "${INITIAL_SWA}"; do
  [[ -f "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse existing run: ${run_dir}" >&2; exit 1; }

echo "resolution=2 learning_rate=${learning_rate} init=${INITIAL_SWA} run=${run_dir}"
python -u scripts/training/train_v22_final.py \
  --config "${CONFIG}" \
  --init-from "${INITIAL_SWA}" \
  --zero-state-on-init \
  --learning-rate "${learning_rate}" \
  --max-steps 2000 \
  --checkpoint-every 500 \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${run_name}"

python -u scripts/training/build_v22_final_swa.py \
  --inputs \
    "${run_dir}/checkpoints/checkpoint_step00000500.pkl" \
    "${run_dir}/checkpoints/checkpoint_step00001000.pkl" \
    "${run_dir}/checkpoints/checkpoint_step00001500.pkl" \
    "${run_dir}/checkpoints/checkpoint_step00002000.pkl" \
  --source-steps 500 1000 1500 2000 \
  --output "${run_dir}/swa/swa_step00500-02000.pkl"

echo "Completed res2 low-LR SWA continuation: ${run_dir}"
