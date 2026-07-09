#!/usr/bin/env bash
# City fixed-lead skill plot: for each anchor in Jan 2022, warm_bp rollout 8d,
# extract Chicago temperature at lead 2d/4d/6d/8d. 4 lines per panel:
# truth, baseline GC, v22 (open-loop train), v22closed K=8 (closed train).
#SBATCH --job-name=city-fixed-lead
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/city_fixed_lead_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/city_fixed_lead_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-12-fixed-lead-month
mkdir -p ${OUT_DIR}

CITY=${CITY:-Chicago}
MONTH=${MONTH:-1}

$PYTHON -u scripts/training/full_mamba_v23/eval_city_fixed_lead_month.py \
  --ckpt-v22 /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
  --ckpt-v22closed /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K8_sg_fresh_20k/v13_residual_step20000.pkl \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --city ${CITY} --month ${MONTH} --anchor-stride-hours 24 \
  --warmup-steps 24 --metric-steps 32 \
  --variable 2m_temperature --seed 0 \
  --out-json ${OUT_DIR}/${CITY}_month${MONTH}_fixed_lead.json
echo "city fixed-lead month eval done"
