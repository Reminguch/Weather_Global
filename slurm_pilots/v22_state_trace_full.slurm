#!/usr/bin/env bash
# Trace Mamba SSM state under cold_FULL (closed-loop) rollout — compare to cold_bp
# K=18 step20000 (best ckpt) — see if state diverges under closed-loop
#SBATCH --job-name=v22-state-trace-full
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_state_trace_full_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_state_trace_full_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba/state_traces

# Use K=18 step 20000 — best ckpt
for MODE in cold_bp cold_full; do
  for K in 2 18; do
    STEP=20000
    CKPT=${ROOT}/results/v22_res2/K${K}_res2_fresh/v13_residual_step${STEP}.pkl
    OUT_NPZ=${OUT_DIR}/state_trace_K${K}_step${STEP}_${MODE}.npz
    echo "=== Tracing K=${K} step=${STEP} mode=${MODE} ==="
    $PYTHON -u scripts/training/full_mamba_v23/eval_state_trace.py \
      --ckpt ${CKPT} \
      --ckpt-in ${GCV1_K3} \
      --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --out-npz ${OUT_NPZ} \
      --resolution 2.0 --mesh-size 4 --width 512 \
      --baseline-msg-steps 6 --residual-msg-steps 2 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --target-steps 40 --seed 0 \
      --eval-mode ${MODE}
  done
done

echo "All traces done"
