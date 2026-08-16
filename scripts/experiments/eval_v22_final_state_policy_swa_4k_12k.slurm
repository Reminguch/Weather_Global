#!/usr/bin/env bash
# Matched reset-every-step evaluations needed for the v22_final res2 state-policy
# SWA comparison. Task 0 evaluates the reset-trained SWA(4k--12k) with its
# proper reset policy. Task 1 evaluates the best carry-trained SWA(4k--12k)
# with state forcibly reset; its normal carry evaluation already exists.
#
# Resource evidence: jobs 12356628, 12386408, and 12411995 used 5.6--5.9 GiB
# CPU RAM and 28--30 minutes for this exact 32-anchor, 40-step res2 shape.
# 12 GiB and 62 minutes preserve the verified margin.
#SBATCH --job-name=v22f-policy-swa-reset
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v22f_policy_swa_reset_%A_%a.out
#SBATCH --error=logs/v22f_policy_swa_reset_%A_%a.err

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

RESET_RUN="artifacts/checkpoints/v22_final/res2_gc500k_k12_full_mamba_state_ablation_20k/di16_closed_sg_reset_every_anchor_20k"
CARRY_RUN="artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k"
RUN_DIRS=("${RESET_RUN}" "${CARRY_RUN}")
LABELS=(reset_trained_swa carry_trained_swa_forced_reset)

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_DIRS[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#RUN_DIRS[@]} - 1 ))" >&2
  exit 2
fi

RUN_DIR="${RUN_DIRS[$TASK_ID]}"
LABEL="${LABELS[$TASK_ID]}"
CKPT="${RUN_DIR}/swa/swa_step04000-12000.pkl"
OUT_JSON="${RUN_DIR}/eval/cold_full_reset_every_step_zero/swa_step04000-12000.json"

for required in "${CKPT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

echo "variant=${LABEL} swa=4k-12k eval_state_policy=reset output=${OUT_JSON}"
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
  --eval-mode cold_full --residual-state-init zero --reset-state-every-step \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${OUT_JSON}"

echo "Completed v22_final state-policy SWA evaluation: ${OUT_JSON}"
