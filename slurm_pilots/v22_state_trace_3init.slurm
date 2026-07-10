#!/usr/bin/env bash
# 3-init state trace comparison: K=18 step 20000, eval_mode=cold_bp
# init modes: ckpt / zero / warm24
#SBATCH --job-name=v22-trace-3init
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_trace_3init_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_trace_3init_%j.err
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

# K=18 step 20000 for all 3 inits
CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
COMMON="
  --ckpt ${CKPT}
  --ckpt-in ${GCV1_K3}
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats
  --resolution 2.0 --mesh-size 4 --width 512
  --baseline-msg-steps 6 --residual-msg-steps 2
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 128
  --temporal-d-state 16 --temporal-d-conv 4
  --temporal-dt-rank auto --temporal-layers 2
  --target-steps 40 --seed 0
  --eval-mode cold_bp"

for INIT in zero warm24; do  # ckpt already done
  echo "=== K=18 step20000 init=${INIT} ==="
  $PYTHON -u scripts/training/full_mamba_v23/eval_state_trace.py $COMMON \
    --residual-state-init ${INIT} \
    --out-npz ${OUT_DIR}/state_trace_K18_step20000_cold_bp_init_${INIT}.npz
done

echo "All 3-init traces done"
