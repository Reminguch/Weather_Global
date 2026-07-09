#!/usr/bin/env bash
# v22closed K=8 city-trace with SHIFTED anchors (= v22 anchor - 6 days)
# so its 24-step warmup ends exactly when v22's forecast starts.
# Metric phase: Jan 15-25, Apr 15-25, Jul 15-25, Oct 15-25  (matches v22)
#SBATCH --job-name=city-trace-v22closed-aligned
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/city_trace_v22closed_aligned_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/city_trace_v22closed_aligned_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces
mkdir -p ${OUT_DIR}

# Shifted by -6 days (= 24 warmup steps × 6h) so metric starts at Jan 15 etc.
for ANCHOR_TARGET in "2022-01-15:2022-01-09" "2022-04-15:2022-04-09" "2022-07-15:2022-07-09" "2022-10-15:2022-10-09"; do
  TARGET=${ANCHOR_TARGET%%:*}    # what v22 used (label for filename)
  ANCHOR=${ANCHOR_TARGET##*:}    # what v22closed uses
  TAG=$(echo $TARGET | sed 's/[-:T]//g' | cut -c1-8)
  echo ""
  echo "=== v22closed K=8 with anchor ${ANCHOR}T00 (metric @ ${TARGET}) ==="
  $PYTHON -u scripts/training/full_mamba_v23/eval_city_trace_closed.py \
    --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K8_sg_fresh_20k/v13_residual_step20000.pkl \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
    --temporal-layers 2 \
    --target-steps 40 --warmup-steps 24 \
    --anchor-date "${ANCHOR}T00:00:00" \
    --out-json ${OUT_DIR}/v22closed_K8_aligned_cities_${TAG}.json
done
echo "all 4 aligned anchors done"
