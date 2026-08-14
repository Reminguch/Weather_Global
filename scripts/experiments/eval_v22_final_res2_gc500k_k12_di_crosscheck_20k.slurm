#!/usr/bin/env bash
# Evaluate every available periodic checkpoint from the v22_final res2 GC500k
# AR-tail-K12 di16/di32 comparison.  The two variants use closed-SG feedback,
# so cold_full is the matched rollout protocol.
#
# Resource evidence: v22_final all-variable 40-step / 32-sample res2 tasks
# 12269478_0--8 completed in 26:32--28:04 using 5.7--6.0 GiB MaxRSS.
# 12 GiB and 62 minutes provide practical margin.  Limit concurrency to three
# A100s because all tasks read the same evaluation Zarr store.
#SBATCH --job-name=v22f-r2-di-k12-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-13%3
#SBATCH --output=logs/v22f_r2_di_k12_eval_%A_%a.out
#SBATCH --error=logs/v22f_r2_di_k12_eval_%A_%a.err

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

RUN_ROOT="artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck_20260812"
RUN_DIRS=(
  "${RUN_ROOT}/di16_closed_sg_20k"
  "${RUN_ROOT}/di32_closed_sg_20k"
)
D_INNERS=(16 32)
STEPS=(2000 4000 6000 8000 10000 12000 14000)

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASKS_PER_VARIANT="${#STEPS[@]}"
VARIANT_IDX=$(( TASK_ID / TASKS_PER_VARIANT ))
STEP_IDX=$(( TASK_ID % TASKS_PER_VARIANT ))
if (( VARIANT_IDX < 0 || VARIANT_IDX >= ${#RUN_DIRS[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#RUN_DIRS[@]} * TASKS_PER_VARIANT - 1 ))" >&2
  exit 2
fi

RUN_DIR="${RUN_DIRS[$VARIANT_IDX]}"
D_INNER="${D_INNERS[$VARIANT_IDX]}"
STEP="${STEPS[$STEP_IDX]}"
CKPT="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${STEP}").pkl"
OUT_JSON="${RUN_DIR}/eval/cold_full_zero/step${STEP}.json"

for required in "${CKPT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

echo "variant=di${D_INNER} checkpoint_step=${STEP} output=${OUT_JSON}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${CKPT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner "${D_INNER}" --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --no-temporal-stateful \
  --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"

echo "Completed v22_final evaluation: ${OUT_JSON}"
