#!/usr/bin/env bash
# Evaluate both periodic-SWA restart lineages at every retained stage. Each
# artifact is evaluated once with recurrent state carried and once with state
# forcibly reset, on the same res2/2022/32-anchor/40-lead protocol used by the
# existing carry-trained and reset-trained references.
#
# Mapping: 2 variants x 3 stages x 2 inference policies = 12 tasks.
# Exact precedent 12447447 used 5.6--5.9 GiB and 28:52--29:14 per task;
# 12 GiB and 62 minutes retain about 2x memory and time headroom.
#SBATCH --job-name=v22f-r2-swa-cycle-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/v22f_r2_swa_cycle_eval_%A_%a.out
#SBATCH --error=logs/v22f_r2_swa_cycle_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r2-swa-cycle-eval-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res2_gc500k_k12_swa_stabilized_20260814"
VARIANTS=(di16 di64_bcg4)
D_INNERS=(16 64)
BC_GROUPS=(1 4)
STAGES=(phase1_swa phase2_swa phase3_final)

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
policies_per_stage=2
stages_per_variant=3
stage_id=$((task_id / policies_per_stage))
policy_id=$((task_id % policies_per_stage))
variant_id=$((stage_id / stages_per_variant))
variant_stage_id=$((stage_id % stages_per_variant))

if ((variant_id < 0 || variant_id >= ${#VARIANTS[@]})); then
  echo "Task ${task_id} is outside 0-11" >&2
  exit 2
fi

variant="${VARIANTS[$variant_id]}"
d_inner="${D_INNERS[$variant_id]}"
bc_group="${BC_GROUPS[$variant_id]}"
stage="${STAGES[$variant_stage_id]}"

case "${stage}" in
  phase1_swa)
    run_dir="${RUN_ROOT}/${variant}_swa_restart_phase1_8k"
    artifact_tag="swa_step02000-08000"
    ckpt="${run_dir}/swa/${artifact_tag}.pkl"
    ;;
  phase2_swa)
    run_dir="${RUN_ROOT}/${variant}_swa_restart_phase2_8k"
    artifact_tag="swa_step02000-08000"
    ckpt="${run_dir}/swa/${artifact_tag}.pkl"
    ;;
  phase3_final)
    run_dir="${RUN_ROOT}/${variant}_swa_restart_phase3_4k"
    artifact_tag="step4000"
    ckpt="${run_dir}/checkpoints/checkpoint_step00004000.pkl"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac

reset_flag=()
if ((policy_id == 0)); then
  policy="carry"
  eval_dir="cold_full_zero"
else
  policy="reset_every_step"
  eval_dir="cold_full_reset_every_step_zero"
  reset_flag=(--reset-state-every-step)
fi
out_json="${run_dir}/eval/${eval_dir}/${artifact_tag}.json"

for required in "${ckpt}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "variant=${variant} stage=${stage} artifact=${artifact_tag} eval_state_policy=${policy} output=${out_json}"
python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner "${d_inner}" --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups "${bc_group}" --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero "${reset_flag[@]}" \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

echo "Completed v22_final periodic-SWA evaluation: ${out_json}"
