#!/usr/bin/env bash
# Trace Mamba SSM state over cold_bp rollout for K=18 ckpt step 2000
#SBATCH --job-name=v22-state-trace
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_state_trace_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_state_trace_%j.err
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
mkdir -p ${OUT_DIR}

# Trace 3 K's (low/mid/high) to compare state dynamics
for K in 2 8 18; do
  STEP=2000
  [ $K -eq 2 ] && STEP=5000
  [ $K -eq 8 ] && STEP=4000
  CKPT=${ROOT}/results/v22_res2/K${K}_res2_fresh/v13_residual_step${STEP}.pkl
  OUT_NPZ=${OUT_DIR}/state_trace_K${K}_step${STEP}.npz
  [ -f "${CKPT}" ] || { echo "ERR: missing ckpt ${CKPT}"; exit 1; }

  echo "=== Tracing K=${K} step=${STEP} ==="
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
    --target-steps 40 --seed 0
  echo "K=${K} done"
done

echo "All traces done. See ${OUT_DIR}"
