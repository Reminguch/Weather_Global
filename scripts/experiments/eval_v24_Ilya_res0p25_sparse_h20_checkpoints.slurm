#!/usr/bin/env bash
# Canonical cold full-feedback evaluation of v24 res0.25 checkpoints
# 50/100/150/200 using exact original GraphCast loss.
# The evaluator streams each completed lead into metrics and retains no full
# prediction history. The legacy two-branch spatial-bias accumulators require
# about 70.2 GiB at this shape; 256G leaves room for data and working arrays.
# Pilot timings (3994.6s compile/first sample, 681.6s steady sample) imply
# 1.35 * (3994.6 + 31 * 681.6) = 9.42h, rounded to a 10h production limit.
#SBATCH --job-name=v24-r025-ckpt-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v24_r025_checkpoint_eval_%A_%a.out
#SBATCH --error=logs/v24_r025_checkpoint_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="${RUN_DIR:-artifacts/checkpoints/v24_Ilya/res0p25_fp32_causal_200_20260826/di16_bcg1_sparse_h20_legacy_zero_head_dp2_fp32_lr3em6_200}"
RUN_CONFIG="${RUN_DIR}/run_config.json"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-checkpoint-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
n_samples="${N_SAMPLES:-32}"
output_suffix="${OUTPUT_SUFFIX:-}"
[[ "${n_samples}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid N_SAMPLES=${n_samples}" >&2; exit 2; }
if [[ -n "${output_suffix}" && ! "${output_suffix}" =~ ^_[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_SUFFIX must be empty or begin with '_' and contain path-safe characters" >&2
  exit 2
fi

case "${task_id}" in
  0|1|2|3)
    steps=(50 100 150 200)
    step="${steps[${task_id}]}"
    artifact_tag="step${step}"
    ckpt="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    ;;
  *)
    echo "Task ${task_id} is outside 0-3" >&2
    exit 2
    ;;
esac

output_tag="${artifact_tag}${output_suffix}"
out_json="${RUN_DIR}/eval/cold_full_zero/${output_tag}.json"
for required in "${ckpt}" "${RUN_CONFIG}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "artifact=${artifact_tag} checkpoint=${ckpt} samples=${n_samples} output=${out_json}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2019 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 --temporal-bc-groups 1 \
  --temporal-stateful --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples "${n_samples}" --seed 0 --out-json "${out_json}"

echo "Completed v24_Ilya res0.25 checkpoint evaluation: ${out_json}"
