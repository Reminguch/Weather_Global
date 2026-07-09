#!/usr/bin/env bash
# 10 cities × 2 vars (2m_temp + total_precip_6hr) trace for v22 K=22 open-loop ckpt.
# 4 seasonal anchors. cold-bp eval semantics (matches v22 training).
#SBATCH --job-name=city-trace-v22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/city_trace_v22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/city_trace_v22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces
mkdir -p ${OUT_DIR}

# 4 seasonal anchors
for ANCHOR in "2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00"; do
  TAG=$(echo $ANCHOR | sed 's/[-:T]//g' | cut -c1-8)
  echo ""
  echo "=================================================="
  echo "Running anchor $ANCHOR (tag $TAG)"
  echo "=================================================="
  $PYTHON -u scripts/training/full_mamba_v23/eval_city_trace.py \
    --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
    --temporal-layers 2 \
    --target-steps 40 \
    --anchor-date "${ANCHOR}" \
    --out-json ${OUT_DIR}/v22_K22_cities_${TAG}.json
done
echo "all 4 anchors done"
