#!/usr/bin/env bash
# Matched res1 held-cosine pilot: di16/BCG1 control versus di64/BCG4.
# Comparable res1 jobs used 232 GiB host RAM and 6.3--6.5 s/update. Eight
# thousand updates plus compilation and validation should take about 15 h;
# 240G/20h retains the proven memory request and useful time headroom.
#SBATCH --job-name=v22f-r1-cos-di
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v22f_r1_cos_di_%A_%a.out
#SBATCH --error=logs/v22f_r1_cos_di_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-cos-di-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

CONFIGS=(
  configs/experiments/v22_final/res1_dm_7yr_k20_di16_bcg1_cosine_8k.json
  configs/experiments/v22_final/res1_dm_7yr_k20_di64_bcg4_cosine_8k.json
)
RUN_NAMES=(di16_bcg1_cosine_8k di64_bcg4_cosine_8k)
OUTPUT_ROOT="artifacts/checkpoints/v22_final/res1_cosine_di16_di64_8k_20260817"

task_id="${SLURM_ARRAY_TASK_ID}"
config="${CONFIGS[$task_id]}"
run_dir="${OUTPUT_ROOT}/${RUN_NAMES[$task_id]}"
[[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
[[ ! -e "${run_dir}" ]] || { echo "Refusing to reuse existing run: ${run_dir}" >&2; exit 1; }

python -u scripts/training/train_v22_final.py --config "${config}"

build_swa() {
  local start="$1"
  local end="$2"
  local step output
  local -a inputs=()
  local -a source_steps=()
  for ((step=start; step<=end; step+=500)); do
    inputs+=("${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl")
    source_steps+=("${step}")
  done
  output="${run_dir}/swa/swa_step$(printf '%05d' "${start}")-$(printf '%05d' "${end}").pkl"
  python -u scripts/training/build_v22_final_swa.py \
    --inputs "${inputs[@]}" --source-steps "${source_steps[@]}" --output "${output}"
}

build_swa 2000 4000
build_swa 2000 6000
build_swa 4000 8000
build_swa 2000 8000

echo "Completed res1 held-cosine pilot: ${run_dir}"
