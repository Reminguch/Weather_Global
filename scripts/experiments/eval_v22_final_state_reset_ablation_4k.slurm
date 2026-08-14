#!/usr/bin/env bash
# Reset-every-step recurrence ablation for the two focused stateful 4k models.
# Their normal-carry cold_full evaluations already exist; these tasks keep the
# checkpoint, inputs, feedback, RNG seed, and sampled anchors unchanged while
# restoring zero residual temporal state before every model application.
#
# Resource evidence: normal-carry v22_final stateful res2 evaluations used
# approximately 5.7--6.0 GiB CPU RAM.  di16 finished in about 29 minutes;
# di64-bcg4 finished within the established 62-minute evaluation limit.
#SBATCH --job-name=v22f-state-reset-4k
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v22f_state_reset_4k_%A_%a.out
#SBATCH --error=logs/v22f_state_reset_4k_%A_%a.err

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

RUN_ROOT="artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck"
RUN_DIRS=(
  "${RUN_ROOT}/di16_closed_sg_stateful_20k"
  "${RUN_ROOT}/di64_bcg4_closed_sg_stateful_20k"
)
D_INNERS=(16 64)
BC_GROUPS=(1 4)
STEP=4000

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_DIRS[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#RUN_DIRS[@]} - 1 ))" >&2
  exit 2
fi

RUN_DIR="${RUN_DIRS[$TASK_ID]}"
D_INNER="${D_INNERS[$TASK_ID]}"
BC_GROUP="${BC_GROUPS[$TASK_ID]}"
CKPT="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${STEP}").pkl"
NORMAL_JSON="${RUN_DIR}/eval/cold_full_zero/step${STEP}.json"
OUT_JSON="${RUN_DIR}/eval/cold_full_reset_every_step_zero/step${STEP}.json"

for required in "${CKPT}" "${NORMAL_JSON}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

echo "variant=di${D_INNER}_bcg${BC_GROUP}_stateful_reset_every_step checkpoint_step=${STEP} output=${OUT_JSON}"
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
  --temporal-bc-groups "${BC_GROUP}" --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero --reset-state-every-step \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${OUT_JSON}"

echo "Completed v22_final reset-every-step evaluation: ${OUT_JSON}"
