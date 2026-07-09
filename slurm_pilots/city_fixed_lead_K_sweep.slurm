#!/usr/bin/env bash
# K-sweep: for each ar-tail-K value in {1, 2, 4, 6, 10, 14}, run fixed-lead
# eval at 10 cities + 4 hotspot regions. Each K gets its own subdir.
#
# - K=1: v22 K=1 spinup (open-loop, basically warm start before AR cascade)
# - K=2..14: v22closed series (closed-loop trained at that ar_tail_K)
#
# v22 (K=22) is kept as fixed reference line in all K-subdirs.
#SBATCH --job-name=cityfix-Ksweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/cityfix_Ksweep_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/cityfix_Ksweep_%j.err
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
CKPTS[1]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl
CKPTS[2]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K2_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[4]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K4_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[6]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K6_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[10]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K10_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[8]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K8_sg_fresh_20k/v13_residual_step20000.pkl
CKPTS[14]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K14_sg_fresh_20k/v13_residual_step20000.pkl

CITIES="NYC,LA,Chicago,Tokyo,Beijing,Shanghai,London,Paris,Mumbai,Sydney,TibetHotspot,AlaskaHotspot,SHoceanTemp,AntarcticCoast"

# Priority order: K=10, K=14 first (user's main interest), then K=8, 6, 4.
# K=1, K=2 dropped per user request. K=8 was previously bp-feedback (mismatched);
# new mixed-mode K=8 will replace the K8/ contents.
for K in 10 14 8 6 4; do
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

    # Plot all 14 locations for this K
    IFS=',' read -r -a CITY_LIST <<< "${CITIES}"
    for LOC in "${CITY_LIST[@]}"; do
        IN=${OUT_DIR}/${LOC}_month${MONTH}_fixed_lead.json
        [ -f "${IN}" ] || continue
        $PYTHON -u scripts/training/full_mamba_v23/plot_city_fixed_lead_month.py \
          --in-json ${IN} \
          --out-png ${OUT_DIR}/${LOC}_month${MONTH}_fixed_lead.png
    done
done

echo "K-sweep done"
