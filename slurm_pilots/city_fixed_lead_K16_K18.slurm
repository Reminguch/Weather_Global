#!/usr/bin/env bash
# Continuation of K-sweep: K=16 and K=18 (mixed-mode eval).
# Output to same base dir as 9590977 (K-sweep) — K16/, K18/ subdirs.
#SBATCH --job-name=cityfix-K16K18
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/cityfix_K16K18_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/cityfix_K16K18_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

BASE_OUT=/home/lm8598/Weather_Global_experiments/results/2026-06-12-fixed-lead-month
MONTH=${MONTH:-1}

declare -A CKPTS
CKPTS[16]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K16_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[18]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K18_sg_fresh_20k/v13_residual_step20000.pkl

CITIES="NYC,LA,Chicago,Tokyo,Beijing,Shanghai,London,Paris,Mumbai,Sydney,TibetHotspot,AlaskaHotspot,SHoceanTemp,AntarcticCoast"

for K in 16 18; do
    OUT_DIR=${BASE_OUT}/K${K}
    mkdir -p ${OUT_DIR}
    echo "================================================"
    echo "=== K=$K  ckpt=${CKPTS[$K]}  out=${OUT_DIR}"
    echo "================================================"

    $PYTHON -u scripts/training/full_mamba_v23/eval_city_fixed_lead_month.py \
      --ckpt-v22 /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
      --ckpt-v22closed ${CKPTS[$K]} \
      --residual-msg-steps 2 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --cities ${CITIES} \
      --variables 2m_temperature,total_precipitation_6hr \
      --month ${MONTH} --anchor-stride-hours 24 \
      --warmup-steps 24 --metric-steps 56 \
      --leads-days 1,2,4,6,8,10,14 \
      --feedback-mode mixed \
      --seed 0 \
      --out-dir ${OUT_DIR}

    IFS=',' read -r -a CITY_LIST <<< "${CITIES}"
    for LOC in "${CITY_LIST[@]}"; do
        IN=${OUT_DIR}/${LOC}_month${MONTH}_fixed_lead.json
        [ -f "${IN}" ] || continue
        $PYTHON -u scripts/training/full_mamba_v23/plot_city_fixed_lead_month.py \
          --in-json ${IN} \
          --out-png ${OUT_DIR}/${LOC}_month${MONTH}_fixed_lead.png
    done
done

echo "K=16, K=18 done"
