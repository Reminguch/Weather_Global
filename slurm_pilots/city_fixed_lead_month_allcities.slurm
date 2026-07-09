#!/usr/bin/env bash
# Fixed-lead skill at 10 cities × 2 vars (T + precip) for Jan 2022.
# One rollout per anchor, multi-city + multi-variable extraction.
# Saves one JSON per city to /home/lm8598/.../results/2026-06-12-fixed-lead-month
#SBATCH --job-name=cityfix-allcities
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=05:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/cityfix_allcities_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/cityfix_allcities_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-12-fixed-lead-month
mkdir -p ${OUT_DIR}

MONTH=${MONTH:-1}

$PYTHON -u scripts/training/full_mamba_v23/eval_city_fixed_lead_month.py \
  --ckpt-v22 /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
  --ckpt-v22closed /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K8_sg_fresh_20k/v13_residual_step20000.pkl \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --cities ALL \
  --variables 2m_temperature,total_precipitation_6hr \
  --month ${MONTH} --anchor-stride-hours 24 \
  --warmup-steps 24 --metric-steps 56 \
  --leads-days 1,2,4,6,8,10,14 \
  --seed 0 \
  --out-dir ${OUT_DIR}

# Plot all 10 cities
for CITY in NYC LA Chicago Tokyo Beijing Shanghai London Paris Mumbai Sydney; do
    $PYTHON -u scripts/training/full_mamba_v23/plot_city_fixed_lead_month.py \
      --in-json ${OUT_DIR}/${CITY}_month${MONTH}_fixed_lead.json \
      --out-png ${OUT_DIR}/${CITY}_month${MONTH}_fixed_lead.png
done

echo "all-cities fixed-lead done"
