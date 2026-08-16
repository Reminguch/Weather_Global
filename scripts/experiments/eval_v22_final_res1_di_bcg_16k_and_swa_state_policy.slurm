#!/usr/bin/env bash
# Evaluate the four carry-trained v22_final Res1 variants at their last saved
# checkpoint (16k) and across rolling SWA windows. Every artifact is evaluated
# once with temporal state carried and once with state reset before every step.
#
# Artifact ordering within each variant:
#   16k, SWA 2-8k, 4-10k, 6-12k, 8-14k, 10-16k
# Policy ordering within each artifact:
#   carry, reset-every-step
#
# Res1 custom-v22 evaluation has no exact completed accounting precedent. The
# 80G/4h request follows the repository's 72-80G guidance for the heaviest Res1
# evaluation shards. Submit tasks 0-1 as a pilot, then gate tasks 2-47 afterok.
#SBATCH --job-name=v22f-r1-state-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-47%2
#SBATCH --output=logs/v22f_r1_state_eval_%A_%a.out
#SBATCH --error=logs/v22f_r1_state_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-state-eval-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813"
RUN_NAMES=(
  di16_bcg1_closed_sg_stateful_20k
  di16_bcg4_closed_sg_stateful_20k
  di32_bcg1_closed_sg_stateful_20k
  di32_bcg4_closed_sg_stateful_20k
)
D_INNERS=(16 16 32 32)
BC_GROUPS=(1 4 1 4)
WINDOW_STARTS=(2000 4000 6000 8000 10000)

BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
artifacts_per_variant=6
policies_per_artifact=2
artifact_id=$((task_id / policies_per_artifact))
policy_id=$((task_id % policies_per_artifact))
variant_id=$((artifact_id / artifacts_per_variant))
artifact_kind=$((artifact_id % artifacts_per_variant))

if ((variant_id < 0 || variant_id >= ${#RUN_NAMES[@]})); then
  echo "Task ${task_id} is outside 0-47" >&2
  exit 2
fi

run_dir="${RUN_ROOT}/${RUN_NAMES[$variant_id]}"
d_inner="${D_INNERS[$variant_id]}"
bc_group="${BC_GROUPS[$variant_id]}"

if ((artifact_kind == 0)); then
  checkpoint_tag="step16000"
  ckpt="${run_dir}/checkpoints/checkpoint_step00016000.pkl"
else
  window_id=$((artifact_kind - 1))
  start="${WINDOW_STARTS[$window_id]}"
  end=$((start + 6000))
  checkpoint_tag="swa_step$(printf '%05d' "${start}")-$(printf '%05d' "${end}")"
  ckpt="${run_dir}/swa/${checkpoint_tag}.pkl"
fi

reset_flag=()
if ((policy_id == 0)); then
  policy="carry"
  eval_dir="cold_full_zero"
else
  policy="reset_every_step"
  eval_dir="cold_full_reset_every_step_zero"
  reset_flag=(--reset-state-every-step)
fi
out_json="${run_dir}/eval/${eval_dir}/${checkpoint_tag}.json"

for required in "${ckpt}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "variant=${RUN_NAMES[$variant_id]} artifact=${checkpoint_tag} eval_state_policy=${policy} output=${out_json}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner "${d_inner}" --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups "${bc_group}" --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero "${reset_flag[@]}" \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

echo "Completed v22_final Res1 state-policy evaluation: ${out_json}"
