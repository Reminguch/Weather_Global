#!/usr/bin/env bash
# Match training and evaluation state policies for the v22_final full-Mamba
# carry/reset pair, then extend the carry-state checkpoint curve beyond 6k.
#
# Tasks 0--6: reset-trained checkpoints evaluated with reset every rollout
# step (2k, 4k, 6k, 8k, 10k, 12k, 20k).
# Tasks 7--12: carry-trained checkpoints evaluated with carried state
# (8k, 10k, 12k, 14k, 16k, 18k).  The carry 20k evaluation already exists.
#
# Resource evidence: matching res2 v22 40-step/32-anchor evaluations used
# 5.7--6.0 GiB CPU RAM and 26--29 minutes.  12 GiB, 62 minutes, and at most
# three concurrent Zarr readers preserve the verified safety margin.
#SBATCH --job-name=v22f-state-policy-ckpt
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-12%3
#SBATCH --output=logs/v22f_state_policy_ckpt_%A_%a.out
#SBATCH --error=logs/v22f_state_policy_ckpt_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res2_gc500k_k12_full_mamba_state_ablation_20k"
RESET_RUN="${RUN_ROOT}/di16_closed_sg_reset_every_anchor_20k"
CARRY_RUN="${RUN_ROOT}/di16_closed_sg_carry_20k"

RUN_DIRS=(
  "${RESET_RUN}" "${RESET_RUN}" "${RESET_RUN}" "${RESET_RUN}"
  "${RESET_RUN}" "${RESET_RUN}" "${RESET_RUN}"
  "${CARRY_RUN}" "${CARRY_RUN}" "${CARRY_RUN}" "${CARRY_RUN}"
  "${CARRY_RUN}" "${CARRY_RUN}"
)
STEPS=(2000 4000 6000 8000 10000 12000 20000 8000 10000 12000 14000 16000 18000)
POLICIES=(reset reset reset reset reset reset reset carry carry carry carry carry carry)

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#STEPS[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#STEPS[@]} - 1 ))" >&2
  exit 2
fi

RUN_DIR="${RUN_DIRS[$TASK_ID]}"
STEP="${STEPS[$TASK_ID]}"
POLICY="${POLICIES[$TASK_ID]}"
CKPT="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${STEP}").pkl"
if [[ "${POLICY}" == "reset" ]]; then
  OUT_JSON="${RUN_DIR}/eval/cold_full_reset_every_step_zero/step${STEP}.json"
  RESET_FLAG=(--reset-state-every-step)
else
  OUT_JSON="${RUN_DIR}/eval/cold_full_zero/step${STEP}.json"
  RESET_FLAG=()
fi

for required in "${CKPT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

echo "run=${RUN_DIR} step=${STEP} eval_state_policy=${POLICY} output=${OUT_JSON}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${CKPT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups 1 --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero "${RESET_FLAG[@]}" \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${OUT_JSON}"

echo "Completed v22_final matched state-policy evaluation: ${OUT_JSON}"
