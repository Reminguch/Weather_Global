#!/usr/bin/env bash
# Compute per-cell error map at lead 240h for v22closed K=10 in warm_full mode.
# Output NC has same schema as existing K22 NC, so the two can be diffed.
#SBATCH --job-name=percell-closed-K10
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/percell_closed_K10_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/percell_closed_K10_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-12-bias-maps
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/full_mamba_v23/compute_per_cell_map_closed.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K10_sg_fresh_20k/v13_residual_step20000.pkl \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --n-anchors 16 \
  --warmup-steps 24 \
  --target-steps 40 \
  --variable 2m_temperature \
  --lead-step-index 39 \
  --seed 0 \
  --out-nc ${OUT_DIR}/v22closed_K10_2mT_lead240h_warmfull_map.nc

echo "per-cell closed-loop map K10 done"
