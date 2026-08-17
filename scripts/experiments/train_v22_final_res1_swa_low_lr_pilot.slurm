#!/usr/bin/env bash
# Res1 conservative SWA continuation pilot. Two tasks warm-start from the best
# carry-state SWA(2k--8k), zero recurrent state, and train 2k updates at lower
# learning rates. Four 500-step checkpoints are averaged after training.
#
# Res1 jobs 12349225 used about 232 GiB host RAM. Their roughly 37.3 hours for
# about 16k updates implies about 4.7 hours for 2k; 240G/7h preserves the known
# successful memory allocation and adds time for compilation and SWA building.
#SBATCH --job-name=v22f-r1-swa-low-lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=07:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v22f_r1_swa_low_lr_%A_%a.out
#SBATCH --error=logs/v22f_r1_swa_low_lr_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-swa-low-lr-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

CONFIG="configs/experiments/v22_final/res1_dm_7yr_k20_di16_bcg1_closed_sg_stateful_20k.json"
INITIAL_SWA="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/swa/swa_step02000-08000.pkl"
OUTPUT_ROOT="artifacts/checkpoints/v22_final/res1_swa_low_lr_continuation_20260816"
LEARNING_RATES=(1e-5 3e-5)
LR_TAGS=(lr1em5 lr3em5)

task_id="${SLURM_ARRAY_TASK_ID}"
learning_rate="${LEARNING_RATES[$task_id]}"
lr_tag="${LR_TAGS[$task_id]}"
run_name="di16_bcg1_swa2k8k_${lr_tag}_2k"
run_dir="${OUTPUT_ROOT}/${run_name}"

for required in "${CONFIG}" "${INITIAL_SWA}"; do
  [[ -f "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse existing run: ${run_dir}" >&2; exit 1; }

echo "resolution=1 learning_rate=${learning_rate} init=${INITIAL_SWA} run=${run_dir}"
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

echo "Completed res1 low-LR SWA continuation: ${run_dir}"
