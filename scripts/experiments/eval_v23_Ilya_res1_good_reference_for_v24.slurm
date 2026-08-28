#!/usr/bin/env bash
# Canonical 32-anchor exact-GC re-evaluation of the good v23 res1 checkpoint
# and its four-member SWA(2k,4k,6k,8k), for comparison with v24 FP32 runs.
#SBATCH --job-name=v23-r1-v24-ref
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v23_r1_v24_ref_%A_%a.out
#SBATCH --error=logs/v23_r1_v24_ref_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res1_sparse_h20_no_temporal_zero_10k_20260817/di16_bcg1_sparse_h20_closed_sg_stateful_no_tzero_lr3em6_10k"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-v24-ref-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    artifact_tag="step10000"
    ckpt="${RUN_DIR}/checkpoints/checkpoint_step00010000.pkl"
    ;;
  1)
    artifact_tag="swa_step02000-08000"
    ckpt="${RUN_DIR}/swa/${artifact_tag}.pkl"
    ;;
  *) exit 2 ;;
esac

out_json="${RUN_DIR}/eval/cold_full_zero_exact_gc/v24_reference32/${artifact_tag}.json"
for required in "${ckpt}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

python -u scripts/analyze_models/eval_v23_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 --temporal-bc-groups 1 \
  --temporal-stateful --no-zero-init-out \
  --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${out_json}"

python - "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

metric = json.loads(Path(sys.argv[1]).read_text())["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
if len(baseline) != 40 or len(full) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
day10 = 100.0 * (1.0 - full[39] / baseline[39])
through_day10 = 100.0 * (1.0 - sum(full[:40]) / sum(baseline[:40]))
print(
    f"day10_improvement_pct={day10:.6f} "
    f"through_day10_improvement_pct={through_day10:.6f}"
)
PY
