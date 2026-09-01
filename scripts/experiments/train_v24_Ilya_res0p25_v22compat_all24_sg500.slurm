#!/usr/bin/env bash
# Single-A100 res0.25 run with 24 equally weighted supervised steps.
# The wrapper submits an independent exact 32-anchor evaluation as soon as each
# atomically written 50-step checkpoint becomes visible.
#SBATCH --job-name=v24-r025-f24
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=288G
#SBATCH --time=30:00:00
#SBATCH --output=logs/v24_r025_full24_%j.out
#SBATCH --error=logs/v24_r025_full24_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res0p25_v22compat_all24_legacy_di16_bcg1_fp32_sg500.json"
OUTPUT_ROOT="artifacts/checkpoints/v24_Ilya/res0p25_v22compat_all24_fp32_20260901"
RUN_NAME="legacy_di16_bcg1_all24_no_tzero_fp32_lr3em6_sg500"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
EVAL_SCRIPT="scripts/experiments/eval_v24_Ilya_res0p25_v22compat_all24_step_exact32.slurm"
GPU_TELEMETRY="logs/v24_r025_full24_${SLURM_JOB_ID}_gpu.csv"
STEPS=(50 100 150 200 250 300 350 400 450 500)

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "Missing evaluator: ${EVAL_SCRIPT}" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing output root: ${OUTPUT_ROOT}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r025-full24-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

python - <<'PY'
import jax

devices = jax.local_devices()
print(f"jax_process_count={jax.process_count()}")
print(f"jax_local_devices={devices}")
if jax.process_count() != 1 or len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit("expected one JAX process with exactly one GPU")
PY

echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "objective=all_steps weights=24x1/24 ar_tail_k=20 fp32_target_tape=reused"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  "${EVAL_SCRIPT}" \
  src/models/mamba/v24_Ilya/checkpoint.py \
  src/models/mamba/v24_Ilya/training/config.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/frame_data.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/runner.py

monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 5
  done
}

monitor_gpu > "${GPU_TELEMETRY}" &
GPU_MONITOR_PID=$!
TRAIN_PID=""
cleanup() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

submit_eval() {
  local step="$1"
  local checkpoint marker_dir marker temporary job_id
  checkpoint="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
  [[ -f "${checkpoint}" ]] || return 0
  marker_dir="${RUN_DIR}/eval_submissions"
  marker="${marker_dir}/step${step}.jobid"
  [[ ! -e "${marker}" ]] || return 0
  mkdir -p "${marker_dir}"
  echo "submitting exact evaluation for completed checkpoint step=${step}"
  if job_id="$(sbatch --parsable \
      --job-name="v24-r025-f24e${step}" \
      --export="ALL,STEP=${step},RUN_DIR=${RUN_DIR}" \
      "${EVAL_SCRIPT}")"; then
    temporary="${marker}.tmp.${SLURM_JOB_ID}"
    printf '%s\n' "${job_id}" > "${temporary}"
    mv "${temporary}" "${marker}"
    echo "evaluation step=${step} job_id=${job_id}"
  else
    echo "evaluation submission failed for step=${step}; will retry" >&2
  fi
}

submit_ready_evals() {
  local step
  for step in "${STEPS[@]}"; do
    submit_eval "${step}"
  done
}

python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}" &
TRAIN_PID=$!
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  submit_ready_evals
  sleep 30
done

TRAIN_STATUS=0
wait "${TRAIN_PID}" || TRAIN_STATUS=$?
TRAIN_PID=""
submit_ready_evals
if (( TRAIN_STATUS != 0 )); then
  echo "training exited with status ${TRAIN_STATUS}" >&2
  exit "${TRAIN_STATUS}"
fi

# Slurm submission can fail transiently even after successful training. Retry
# any missing submissions without coupling one checkpoint to another.
for attempt in 1 2 3 4 5; do
  missing=0
  submit_ready_evals
  for step in "${STEPS[@]}"; do
    checkpoint="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    marker="${RUN_DIR}/eval_submissions/step${step}.jobid"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
    [[ -f "${marker}" ]] || missing=1
  done
  (( missing == 0 )) && break
  echo "waiting to retry missing evaluation submissions attempt=${attempt}" >&2
  sleep 60
done

for step in "${STEPS[@]}"; do
  marker="${RUN_DIR}/eval_submissions/step${step}.jobid"
  [[ -s "${marker}" ]] || { echo "Missing evaluation job record: ${marker}" >&2; exit 1; }
done

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
