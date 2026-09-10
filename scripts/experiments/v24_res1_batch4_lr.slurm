#!/usr/bin/env bash
# Resources are overridden per stage after the batch-4 GPU smoke measurement.
#SBATCH --job-name=v24-r1-b4lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_r1_b4lr_%A_%a.out
#SBATCH --error=logs/v24_r1_b4lr_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
STAGE="${1:?stage required}"
EXPERIMENT="artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908"
export MPLCONFIGDIR="/tmp/mpl-v24-b4lr-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh
sha256sum --quiet -c "${EXPERIMENT}/source/SHA256SUMS"
GPU_LOG="${EXPERIMENT}/logs/${STAGE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpu.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
echo "stage=${STAGE} task=${SLURM_ARRAY_TASK_ID} job=${SLURM_JOB_ID}"
git rev-parse HEAD
python -u scripts/experiments/run_v24_res1_batch4_lr.py "${STAGE}" --task "${SLURM_ARRAY_TASK_ID}"
