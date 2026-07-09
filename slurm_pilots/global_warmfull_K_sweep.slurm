#!/usr/bin/env bash
# Global lat-weighted RMSE K-sweep under warm_full (closed-loop) eval.
# For each K, runs eval_v22_clean.py --eval-mode warm_full → JSON with
# per_variable_per_step RMSE/MAE across 32 anchors × 40 lead steps (= 240h).
#
# Models evaluated:
# - v22 K=22 (open-loop trained) as reference
# - v22closed K=10, 14, 16, 18 (closed-loop trained at those ar_tail_K)
#
# Eval mode warm_full = 24-step truth warmup + closed-loop full-feedback rollout,
# with TWO independent branches (baseline self-rollout + model self-rollout).
# This is the "fair deployment scenario" comparison.
#SBATCH --job-name=global-warmfull-Ksweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/global_warmfull_Ksweep_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/global_warmfull_Ksweep_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-12-global-warmfull-Ksweep
mkdir -p ${OUT_DIR}

# K=22 = v22 open-loop ckpt;  K=10,14,16,18 = v22closed ckpts
declare -A CKPTS LABELS
CKPTS[22]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl
LABELS[22]=v22_openloop_K22
CKPTS[10]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K10_sg_fresh_20k/v13_residual_step20000.pkl
LABELS[10]=v22closed_K10
CKPTS[14]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K14_sg_fresh_20k/v13_residual_step20000.pkl
LABELS[14]=v22closed_K14
CKPTS[16]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K16_sg_fresh_20k/v13_residual_step20000.pkl
LABELS[16]=v22closed_K16
CKPTS[18]=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K18_sg_fresh_20k/v13_residual_step20000.pkl
LABELS[18]=v22closed_K18

# Priority order: 10, 14 first; then 16, 18; finally K=22 reference
for K in 10 14 16 18 22; do
    LABEL=${LABELS[$K]}
    OUT_JSON=${OUT_DIR}/${LABEL}_warm_full_K40.json
    echo "================================================"
    echo "=== K=$K  label=${LABEL}  ckpt=${CKPTS[$K]}"
    echo "================================================"

    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
      --ckpt ${CKPTS[$K]} \
      --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
      --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
      --residual-msg-steps 2 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --eval-mode warm_full \
      --target-steps 40 \
      --warmup-steps 24 \
      --n-samples 32 \
      --seed 0 \
      --out-json ${OUT_JSON}
done

echo "global warm_full K-sweep done"
