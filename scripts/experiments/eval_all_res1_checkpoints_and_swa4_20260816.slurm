#!/usr/bin/env bash
# Evaluate the 2026-08-16 snapshot of all saved Res1 training checkpoints and
# rolling four-checkpoint SWAs with each run's matching temporal-state policy.
#
# Existing matching-policy results cover the four carry-trained v22_final 16k
# checkpoints and their 20 SWAs. This array fills the 47 missing results:
#   tasks  0-27: four carry-trained v22_final runs, raw checkpoints 2k--14k
#   tasks 28-39: two reset-trained v22_final runs, raw checkpoints 2k--12k
#   tasks 40-45: two reset-trained v22_final runs, SWA4 windows 2k--8k,
#                4k--10k, and 6k--12k
#   task      46: carry-trained v23_Ilya endpoint run, raw checkpoint 2k
#
# Exact Res1 evaluations (jobs 12448854/12448856) took 36-38 minutes and used
# 11.5-11.8 GiB CPU RAM. Each array element runs up to two evaluations, so
# 20G/1h45 leaves useful time and memory headroom. Packing 47 evaluations into
# 24 elements stays below the per-user QOS submission limit.
#SBATCH --job-name=res1-all-ckpt-swa4
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:45:00
#SBATCH --array=0-23
#SBATCH --output=logs/res1_all_ckpt_swa4_%A_%a.out
#SBATCH --error=logs/res1_all_ckpt_swa4_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-res1-all-ckpt-swa4-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

CARRY_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813"
RESET_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_reset_every_anchor_20k_20260815"
V23_ROOT="artifacts/checkpoints/v23_Ilya/res1_endpoint_k24_20k_20260814"
CARRY_RUNS=(
  di16_bcg1_closed_sg_stateful_20k
  di16_bcg4_closed_sg_stateful_20k
  di32_bcg1_closed_sg_stateful_20k
  di32_bcg4_closed_sg_stateful_20k
)
RESET_RUNS=(
  di16_bcg1_closed_sg_reset_every_anchor_20k
  di16_bcg4_closed_sg_reset_every_anchor_20k
)
D_INNERS=(16 16 32 32)
BC_GROUPS=(1 4 1 4)

BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

evaluate_artifact() {
  local artifact_id="$1"
  local local_id variant_id step start end run_dir d_inner bc_group
  local checkpoint_tag ckpt eval_dir evaluator out_json required
  local -a reset_flag=()

  if ((artifact_id < 28)); then
    variant_id=$((artifact_id / 7))
    step=$((2000 * (artifact_id % 7 + 1)))
    run_dir="${CARRY_ROOT}/${CARRY_RUNS[$variant_id]}"
    d_inner="${D_INNERS[$variant_id]}"
    bc_group="${BC_GROUPS[$variant_id]}"
    checkpoint_tag="step${step}"
    ckpt="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    eval_dir="cold_full_zero"
    evaluator="scripts/analyze_models/eval_v22_final.py"
  elif ((artifact_id < 40)); then
    local_id=$((artifact_id - 28))
    variant_id=$((local_id / 6))
    step=$((2000 * (local_id % 6 + 1)))
    run_dir="${RESET_ROOT}/${RESET_RUNS[$variant_id]}"
    d_inner=16
    bc_group=$((variant_id == 0 ? 1 : 4))
    checkpoint_tag="step${step}"
    ckpt="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    eval_dir="cold_full_reset_every_step_zero"
    evaluator="scripts/analyze_models/eval_v22_final.py"
    reset_flag=(--reset-state-every-step)
  elif ((artifact_id < 46)); then
    local_id=$((artifact_id - 40))
    variant_id=$((local_id / 3))
    start=$((2000 * (local_id % 3 + 1)))
    end=$((start + 6000))
    run_dir="${RESET_ROOT}/${RESET_RUNS[$variant_id]}"
    d_inner=16
    bc_group=$((variant_id == 0 ? 1 : 4))
    checkpoint_tag="swa_step$(printf '%05d' "${start}")-$(printf '%05d' "${end}")"
    ckpt="${run_dir}/swa/${checkpoint_tag}.pkl"
    eval_dir="cold_full_reset_every_step_zero"
    evaluator="scripts/analyze_models/eval_v22_final.py"
    reset_flag=(--reset-state-every-step)
  else
    run_dir="${V23_ROOT}/di16_bcg1_last_step_closed_sg_stateful_20k"
    d_inner=16
    bc_group=1
    checkpoint_tag="step2000"
    ckpt="${run_dir}/checkpoints/checkpoint_step00002000.pkl"
    eval_dir="cold_full_zero"
    evaluator="scripts/analyze_models/eval_v23_Ilya.py"
  fi

  out_json="${run_dir}/eval/${eval_dir}/${checkpoint_tag}.json"
  for required in "${ckpt}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
    [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
  done
  [[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
  mkdir -p "$(dirname "${out_json}")"

  echo "artifact_id=${artifact_id} run=${run_dir} artifact=${checkpoint_tag} eval_dir=${eval_dir} output=${out_json}"
  python -u "${evaluator}" \
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

  echo "Completed Res1 evaluation: ${out_json}"
}

array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((array_task_id < 0 || array_task_id > 23)); then
  echo "Array task ${array_task_id} is outside 0-23" >&2
  exit 2
fi

first_artifact=$((array_task_id * 2))
evaluate_artifact "${first_artifact}"
if ((first_artifact + 1 <= 46)); then
  evaluate_artifact "$((first_artifact + 1))"
fi
