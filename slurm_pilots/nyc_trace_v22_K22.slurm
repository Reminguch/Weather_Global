#!/usr/bin/env bash
# NYC 2m_temperature trace for v22 K=22 ckpt (open-loop semantics).
# Single anchor (Jul 1 2022), 40-step cold-bp rollout.
#SBATCH --job-name=nyc-trace-v22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/nyc_trace_v22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/nyc_trace_v22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-nyc-traces
mkdir -p ${OUT_DIR}

# Multiple anchors across seasons
for ANCHOR in "2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00"; do
  TAG=$(echo $ANCHOR | sed 's/[-:T]//g' | cut -c1-8)
  echo "=== Running anchor $ANCHOR ==="
  $PYTHON -u scripts/training/full_mamba_v23/eval_nyc_trace.py \
    --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
    --temporal-layers 2 \
    --target-steps 40 \
    --anchor-date "${ANCHOR}" \
    --out-json ${OUT_DIR}/v22_K22_nyc_${TAG}.json
done
echo "all 4 anchors done"
