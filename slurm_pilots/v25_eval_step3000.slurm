#!/usr/bin/env bash
# Full eval of v25 step3000 (gated extreme head, phase 1 done) at res=2.
# Two tasks:
#   0: global RMSE (cold_bp) — for direct comparison vs v22 K=18 step20000 cold_bp
#   1: top-percentile residual RMSE + gate diagnostics
#SBATCH --job-name=v25-step3000-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_eval_step3000_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_eval_step3000_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
CKPT=${ROOT}/results/v25_gated_extreme/K18_gext_phase1_from_step20k/v13_residual_step3000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-27-v25-gext-eval
mkdir -p ${OUT_DIR}

[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

COMMON="
  --ckpt ${CKPT}
  --ckpt-in ${GCV1_K3}
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats
  --resolution 2.0 --mesh-size 4 --width 512
  --baseline-msg-steps 6 --residual-msg-steps 2
  --val-year 2022 --train-start-year 2015 --train-end-year 2021
  --input-duration 12h
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 128
  --temporal-d-state 16 --temporal-d-conv 4
  --temporal-dt-rank auto --temporal-layers 2
  --target-steps 40 --n-samples 32 --seed 0"

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== v25 step3000 — Global RMSE (cold_bp) ==="
    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py $COMMON \
      --eval-mode cold_bp --residual-state-init zero \
      --warmup-steps 24 \
      --out-json ${OUT_DIR}/v25_step3000_cold_bp_zero.json
    ;;
  1)
    echo "=== v25 step3000 — top-percentile residual + gate stats ==="
    $PYTHON -u scripts/training/full_mamba_v25/eval_top_percentile_residual.py $COMMON \
      --gate-bias-init -3.5 \
      --out-json ${OUT_DIR}/v25_step3000_top_percentile.json
    ;;
  2)
    # v22 K=18 step20000 (BASELINE — no extreme head) for WITH-vs-WITHOUT comparison
    BASE_CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
    BASE_COMMON=$(echo "${COMMON}" | sed "s#--ckpt ${CKPT}#--ckpt ${BASE_CKPT}#")
    echo "=== v22 K=18 step20000 (BASELINE no-extreme-head) — top-percentile residual ==="
    $PYTHON -u scripts/training/full_mamba_v25/eval_top_percentile_residual.py $BASE_COMMON \
      --baseline-mode \
      --out-json ${OUT_DIR}/v22_K18_step20000_top_percentile_baseline.json
    ;;
esac

echo "v25 step3000 eval task ${SLURM_ARRAY_TASK_ID} done"
