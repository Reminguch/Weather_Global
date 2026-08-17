#!/usr/bin/env bash
# Re-evaluate LH's clean 15-year res1/K20/S128 lineage with the maintained
# v22_final evaluator. Seven artifacts (six SWA members plus their SWA) are
# evaluated with state carried and forcibly reset: 7 * 2 = 14 array tasks.
#
# A prior matching res1 evaluation task used ~12 GiB CPU RAM and ~37 minutes.
# This d_inner=128 model is larger than those di16/di32 variants, so use 48 GiB
# and two hours while retaining one A100-80GB.
#SBATCH --job-name=v22-lh-r1-swa-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-13%2
#SBATCH --output=logs/v22_lh_r1_swa_eval_%A_%a.out
#SBATCH --error=logs/v22_lh_r1_swa_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22-lh-r1-swa-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

LH_RUN="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res1_sizeladder_15yr/ladder_15yr_S128_h128_di128_ds16"
OUTPUT_ROOT="artifacts/evaluations/v22_legacy_lh/res1_15yr_k20_s128_swa6k_11k"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

ARTIFACT_TAGS=(swa_step6000-11000 step6000 step7000 step8000 step9000 step10000 step11000)
ARTIFACT_FILES=(
  v13_residual_SWA_step6k-11k.pkl
  v13_residual_step6000.pkl
  v13_residual_step7000.pkl
  v13_residual_step8000.pkl
  v13_residual_step9000.pkl
  v13_residual_step10000.pkl
  v13_residual_step11000.pkl
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
artifact_id=$((task_id / 2))
policy_id=$((task_id % 2))
if ((artifact_id < 0 || artifact_id >= ${#ARTIFACT_TAGS[@]})); then
  echo "Task ${task_id} is outside 0-13" >&2
  exit 2
fi

tag="${ARTIFACT_TAGS[$artifact_id]}"
ckpt="${LH_RUN}/${ARTIFACT_FILES[$artifact_id]}"
reset_flag=()
if ((policy_id == 0)); then
  policy="carry"
  eval_dir="cold_full_zero"
else
  policy="reset_every_step"
  eval_dir="cold_full_reset_every_step_zero"
  reset_flag=(--reset-state-every-step)
fi
out_json="${OUTPUT_ROOT}/${eval_dir}/${tag}.json"

for required in "${ckpt}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "artifact=${tag} eval_state_policy=${policy} checkpoint=${ckpt} output=${out_json}"
sha256sum "${ckpt}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2007 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups 1 --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero "${reset_flag[@]}" \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

echo "Completed LH v22 res1 evaluation: ${out_json}"
