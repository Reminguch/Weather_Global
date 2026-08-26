#!/usr/bin/env bash
# Matched cold full-feedback selection evaluation of the v23_Ilya res0.25
# sparse-H20 run: raw checkpoints 150/200/250 and SWA over 200--400.
# The evaluator streams each completed lead into metrics and retains no full
# prediction history. The legacy two-branch spatial-bias accumulators require
# about 70.2 GiB at this shape; 256G leaves room for data and working arrays.
# Pilot timings (3994.6s compile/first sample, 681.6s steady sample) imply
# 1.35 * (3994.6 + 31 * 681.6) = 9.42h, rounded to a 10h production limit.
#SBATCH --job-name=v23-r025-sel-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v23_r025_selection_eval_%A_%a.out
#SBATCH --error=logs/v23_r025_selection_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_hybrid_2k_20260818/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_hybrid_lr3em6_2k"
RUN_CONFIG="${RUN_DIR}/run_config.json"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-selection-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
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
  0|1|2)
    steps=(150 200 250)
    step="${steps[${task_id}]}"
    artifact_tag="step${step}"
    ckpt="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    ;;
  3)
    artifact_tag="swa_step00200-00400"
    ckpt="${RUN_DIR}/swa/${artifact_tag}.pkl"
    inputs=()
    source_steps=()
    for step in 200 250 300 350 400; do
      input="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
      [[ -f "${input}" ]] || { echo "Missing SWA input: ${input}" >&2; exit 1; }
      inputs+=("${input}")
      source_steps+=("${step}")
    done
    if [[ ! -e "${ckpt}" ]]; then
      mkdir -p "$(dirname "${ckpt}")"
      python -u scripts/training/build_v23_Ilya_swa.py \
        --inputs "${inputs[@]}" --source-steps "${source_steps[@]}" --output "${ckpt}"
    fi
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
python -u scripts/analyze_models/eval_v23_Ilya.py \
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

echo "Completed v23_Ilya res0.25 selection evaluation: ${out_json}"
