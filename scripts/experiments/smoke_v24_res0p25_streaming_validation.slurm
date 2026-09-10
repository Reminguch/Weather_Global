#!/usr/bin/env bash
# One full-resolution validation segment, baseline then checkpoint 100.
# Comparable res0.25 validation used about 110 GiB host RAM and 35 minutes.
# Allow compilation headroom for the newly streamed full-24 objective.
#SBATCH --job-name=v24-r025-val-smoke
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:20:00
#SBATCH --output=logs/v24_r025_val_smoke_%j.out
#SBATCH --error=logs/v24_r025_val_smoke_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Weather_global repository}"
source scripts/graphcast_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export MPLCONFIGDIR="/tmp/mpl-v24-val-smoke-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" logs

SOURCE_CONFIG="configs/experiments/v24_Ilya/res0p25_uniform_full24_legacy_di64_bcg2_bf16_dp4_lr1em5_sg500.json"
CHECKPOINT="artifacts/checkpoints/v24_Ilya/res0p25_uniform_full24_di64_bcg2_dp4_20260905/legacy_di64_bcg2_uniform_full24_bf16_dp4_lr1em5_sg500/checkpoints/checkpoint_step00000100.pkl"
SMOKE_CONFIG="logs/v24_r025_val_smoke_${SLURM_JOB_ID}_config.json"
[[ -f "${CHECKPOINT}" ]]
python - "${SOURCE_CONFIG}" "${SMOKE_CONFIG}" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path
source, destination, job = sys.argv[1:]
config = json.loads(Path(source).read_text())
config["distributed"].update(mode="single", num_devices=1, per_device_batch_size=1)
config["validation"]["num_segments"] = 1
config["output"] = {
    "output_root": "artifacts/checkpoints/v24_Ilya/streaming_validation_smoke",
    "run_name": f"checkpoint100_{job}",
}
Path(destination).write_text(json.dumps(config, indent=2) + "\n")
PY

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits --loop=5 \
    > "logs/v24_r025_val_smoke_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT
sha256sum src/models/mamba/v24_Ilya/training/endpoint_step.py "${SMOKE_CONFIG}"
python -u scripts/analyze_models/smoke_v24_streaming_validation.py \
    --config "${SMOKE_CONFIG}" --validation-compare "${CHECKPOINT}"
