#!/usr/bin/env bash
# v5 = Ilya residual_mamba 1:1 copy (src/models/mamba/residual_mamba/...).
#
# Architecture:
#   * Two GraphCast instances:
#     - baseline (frozen, --baseline-ckpt = DeepMind public GC_small)
#     - residual model (GC + Mamba interleaved inside processor, ALL trainable)
#   * residual_loss = ||residual_model(x) - (truth - baseline(x))||^2
#   * At deployment (truth-anchored AR): full_pred = baseline + residual,
#     fed back to rolling buffer for next step
#
# Smoke: 100 steps to verify trainer boots + double GC forward + loss.

#SBATCH --job-name=v5-rm-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v5_rm_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v5_rm_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT=/home/lm8598/Weather_Global_experiments/results/v5_residual_mamba

${PYTHON} -u -m src.models.mamba.residual_mamba.training.runner \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --baseline-ckpt "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v5_rm_res1_m5_K1_stateful_seg32_smoke \
  --batch-size 1 \
  --max-steps 100 --eval-every 50 --eval-batch-size 4 --checkpoint-every 50 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --input-duration 12h \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 \
  --temporal-d-conv 4 \
  --temporal-dt-rank auto \
  --temporal-layers 1 \
  --temporal-stateful \
  --target-steps 1 \
  --len-segment 32 \
  --bptt-steps 8 \
  --chunk-load-workers 6 \
  --data-cache-mode auto \
  --data-cache-max-gib 80 \
  --trainable-part mamba
