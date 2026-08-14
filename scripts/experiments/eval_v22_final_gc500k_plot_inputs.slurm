#!/usr/bin/env bash
# Re-evaluate every res2 model curve used by the gc500k/AR-k12 comparison plot
# through the organized v22_final evaluator. DeepMind res1 remains an external
# full-year CSV reference and is not part of this array.
#
# Budget evidence: jobs 11546572, 11841973, 11842403, and 11843696 completed
# in 27:44--29:19 with 6.2--6.5 GiB MaxRSS. Request 12G and 50 minutes.
#SBATCH --job-name=v22f-r2-plot-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/v22f_r2_plot_eval_%A_%a.out
#SBATCH --error=logs/v22f_r2_plot_eval_%A_%a.err

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

CURRENT_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k"
CURRENT_OPEN="${CURRENT_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_open_stateless_from_gc500k_50000steps"
CURRENT_CLOSED="${CURRENT_ROOT}/resmamba_res2_v20_k12_di16_l2_dc4_closed_sg_stateless_from_gc500k_50000steps"
EARLIER_ROOT="artifacts/checkpoints/7_years/resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k"
EARLIER_OPEN="${EARLIER_ROOT}/resmamba_res2_v20_k14_di16_l2_dc4_open_20000steps"
EARLIER_CLOSED="${EARLIER_ROOT}/resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_20000steps"
AR_DIR="artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k/vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps"
GC500K="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
GC50K="artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz"
AUX_RESIDUAL="${CURRENT_OPEN}/v13_residual_step50000.pkl"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
OUT_ROOT="plots/analyze_models/data/resolution_eval/resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k/v22_final_eval_json"

LABELS=(
  current_open_step50k
  current_closed_step50k
  current_open_swa42k_50k
  current_closed_swa42k_50k
  earlier_open_step16k
  earlier_closed_step16k
  vanilla_ar_k12_best_step24k
  vanilla_ar_k12_swa18k_26k
  source_vanilla_step500k
)
RESIDUAL_CKPTS=(
  "${CURRENT_OPEN}/v13_residual_step50000.pkl"
  "${CURRENT_CLOSED}/v13_residual_step50000.pkl"
  "${CURRENT_OPEN}/v13_residual_SWA_step42k-50k.pkl"
  "${CURRENT_CLOSED}/v13_residual_SWA_step42k-50k.pkl"
  "${EARLIER_OPEN}/v13_residual_step16000.pkl"
  "${EARLIER_CLOSED}/v13_residual_step16000.pkl"
  "${AUX_RESIDUAL}"
  "${AUX_RESIDUAL}"
  "${AUX_RESIDUAL}"
)
BASELINE_CKPTS=(
  "${GC500K}"
  "${GC500K}"
  "${GC500K}"
  "${GC500K}"
  "${GC50K}"
  "${GC50K}"
  "${AR_DIR}/ckpt_step24000.npz"
  "${AR_DIR}/ckpt_SWA_step18k-26k.npz"
  "${GC500K}"
)
EVAL_MODES=(cold_bp cold_full cold_bp cold_full cold_bp cold_full cold_bp cold_bp cold_bp)
STATEFUL=(false false false false true true false false false)
RESIDUAL_ALPHA=(1 1 1 1 1 1 0 0 0)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#LABELS[@]} )); then
  echo "Task ${TASK_ID} is outside 0-$(( ${#LABELS[@]} - 1 ))" >&2
  exit 2
fi

LABEL="${LABELS[$TASK_ID]}"
RESIDUAL_CKPT="${RESIDUAL_CKPTS[$TASK_ID]}"
BASELINE_CKPT="${BASELINE_CKPTS[$TASK_ID]}"
EVAL_MODE="${EVAL_MODES[$TASK_ID]}"
ALPHA="${RESIDUAL_ALPHA[$TASK_ID]}"

for required in "${RESIDUAL_CKPT}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  TARGET_STEPS=2
  WARMUP_STEPS=2
  N_SAMPLES=1
  OUT_JSON="${OUT_ROOT}/smoke/${LABEL}.json"
else
  TARGET_STEPS=40
  WARMUP_STEPS=24
  N_SAMPLES=32
  OUT_JSON="${OUT_ROOT}/${LABEL}.json"
fi
mkdir -p "$(dirname "${OUT_JSON}")"

STATEFUL_FLAG=(--temporal-stateful)
if [[ "${STATEFUL[$TASK_ID]}" != true ]]; then
  STATEFUL_FLAG=(--no-temporal-stateful)
fi

echo "label=${LABEL} mode=${EVAL_MODE} stateful=${STATEFUL[$TASK_ID]} alpha=${ALPHA} output=${OUT_JSON}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${RESIDUAL_CKPT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  "${STATEFUL_FLAG[@]}" \
  --eval-mode "${EVAL_MODE}" --residual-state-init zero --residual-alpha "${ALPHA}" \
  --target-steps "${TARGET_STEPS}" --warmup-steps "${WARMUP_STEPS}" \
  --n-samples "${N_SAMPLES}" --seed 0 --out-json "${OUT_JSON}"

echo "Completed v22_final evaluation: ${OUT_JSON}"
