#!/usr/bin/env bash
# Early-signal eval on v25 gated extreme phase-1 step 500 ckpt:
#   - global RMSE (4-mode), to confirm we didn't break the base model
#   - top-5% / top-1% residual RMSE + gate diagnostics (extreme head's headline metric)
#SBATCH --job-name=v25-step500-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_eval_step500_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_eval_step500_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
CKPT=${ROOT}/results/v25_gated_extreme/K18_gext_phase1_from_step20k/v13_residual_step500.pkl
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
    # Global RMSE — must not regress vs K=18 baseline
    echo "=== v25 step500 — cold_bp global RMSE ==="
    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py $COMMON \
      --eval-mode cold_bp --residual-state-init zero \
      --warmup-steps 24 \
      --out-json ${OUT_DIR}/v25_step500_cold_bp_zero.json
    ;;
  1)
    # Top-percentile residual RMSE + gate diagnostics
    echo "=== v25 step500 — top-percentile residual + gate stats ==="
    $PYTHON -u scripts/training/full_mamba_v25/eval_top_percentile_residual.py $COMMON \
      --gate-bias-init -3.5 \
      --out-json ${OUT_DIR}/v25_step500_top_percentile.json
    ;;
esac

echo "v25 step500 eval task ${SLURM_ARRAY_TASK_ID} done"
