#!/usr/bin/env bash
# v5 = Ilya's gc_mamba architecture (Mamba interleaved INSIDE GraphCast
# processor) with FROZEN DeepMind GraphCast_small as backbone.
# Apples-to-apples vs v3 / v4 (same frozen backbone, same data, same lat-w
# loss). Difference: Mamba placement is interleaved (mesh_processor_interleaved)
# instead of outer residual.
#
# Config mirrors Ilya's defaults but at our 1° / m=5 resolution to keep
# baseline GraphCast comparable to v3/v4. Mamba hidden=128, layers=1.
# target_steps=1 (single-step training, deployment-friendly per our
# truth-anchored AR findings). Sequential 32-cycle segments with TBPTT
# via stateful SSM hidden state carry across BPTT positions inside a chunk.

#SBATCH --job-name=v5-gcm-real
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_real_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_real_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_graphcast.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT=/home/lm8598/Weather_Global_experiments/results/v5_gc_mamba

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --resolution 1.0 --mesh-size 5 --width 128 --processor-msg-steps 1 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v5_gcm_res1_m5_K1_stateful_seg32_to10k \
  --batch-size 4 \
  --max-steps 10000 --eval-every 500 --eval-batch-size 4 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --input-duration 12h \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-layers 1 \
  --temporal-stateful \
  --target-steps 1 \
  --sequential-segment-steps 32 \
  --trainable-part mamba
