#!/usr/bin/env bash
# Immediate canonical V22-matched evaluation of already saved res1 artifacts.
# Uses 32 cold full-feedback anchors and zero residual state.  The 24-step
# anchor reservation matches V22 but does not warm up a cold rollout.
# This script deliberately has no training-job dependency.
#SBATCH --job-name=v24-r1-eager32
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-10
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:20:00
#SBATCH --output=logs/v24_r1_eager32_%A_%a.out
#SBATCH --error=logs/v24_r1_eager32_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-artifacts/checkpoints/v24_Ilya/res1_v22compat_init_ablation_20260828}"
MANIFEST="${MANIFEST:-configs/experiments/v24_Ilya/res1_v22compat_eager_exact32_artifacts.json}"
EVAL_DIR_TAG="${EVAL_DIR_TAG:-matched_v22_reference32}"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
V22_REFERENCE="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json"

export MPLCONFIGDIR="/tmp/mpl-v24-r1-eager32-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
task_id="${SLURM_ARRAY_TASK_ID:-0}"
task_count="$(jq 'length' "${MANIFEST}")"
if ((task_id < 0 || task_id >= task_count)); then
  echo "Task ${task_id} is outside 0-$((task_count - 1))" >&2
  exit 2
fi

spec="$(jq -cer ".[$task_id]" "${MANIFEST}")"
run_name="$(jq -r '.run_name' <<<"${spec}")"
artifact_tag="$(jq -r '.artifact_tag' <<<"${spec}")"
init_scheme="$(jq -r '.temporal_init_scheme' <<<"${spec}")"
temporal_zero="$(jq -r '.temporal_zero_init_out' <<<"${spec}")"

case "${artifact_tag}" in
  step[0-9]*)
    step="${artifact_tag#step}"
    [[ "${step}" =~ ^[0-9]+$ ]] || { echo "Invalid step artifact: ${artifact_tag}" >&2; exit 2; }
    ckpt="${EXPERIMENT_ROOT}/${run_name}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    ;;
  swa_step02000-08000)
    ckpt="${EXPERIMENT_ROOT}/${run_name}/swa/${artifact_tag}.pkl"
    ;;
  *)
    echo "Unsupported artifact tag: ${artifact_tag}" >&2
    exit 2
    ;;
esac

run_dir="${EXPERIMENT_ROOT}/${run_name}"
run_config="${run_dir}/run_config.json"
[[ "${EVAL_DIR_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid EVAL_DIR_TAG: ${EVAL_DIR_TAG}" >&2; exit 2; }
out_json="${run_dir}/eval/cold_full_zero_exact_gc/${EVAL_DIR_TAG}/${artifact_tag}.json"
for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}" "${V22_REFERENCE}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

zero_flag=()
case "${temporal_zero}" in
  true) ;;
  false) zero_flag=(--no-zero-init-out) ;;
  *) echo "Invalid temporal_zero_init_out: ${temporal_zero}" >&2; exit 2 ;;
esac

echo "run=${run_name} artifact=${artifact_tag} init=${init_scheme} temporal_zero=${temporal_zero} samples=32 output=${out_json}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme "${init_scheme}" \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups 1 --temporal-stateful "${zero_flag[@]}" \
  --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${out_json}"

python - "${out_json}" "${V22_REFERENCE}" <<'PY'
import json
import sys
from pathlib import Path

candidate = json.loads(Path(sys.argv[1]).read_text())
reference = json.loads(Path(sys.argv[2]).read_text())
if candidate["chosen_idx"] != reference["chosen_idx"]:
    raise SystemExit("chosen_idx differs from the good V22 exact reference")
metric = candidate["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
if len(baseline) != 40 or len(full) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
day10 = 100.0 * (1.0 - full[39] / baseline[39])
through_day10 = 100.0 * (1.0 - sum(full) / sum(baseline))
print(f"day10_improvement_pct={day10:.6f} through_day10_improvement_pct={through_day10:.6f}")
PY
