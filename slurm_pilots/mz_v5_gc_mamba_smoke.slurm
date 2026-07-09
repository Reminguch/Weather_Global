#!/usr/bin/env bash
# v5 smoke: gc_mamba (Mamba interleaved INSIDE GraphCast processor), Ilya-style.
# Architecture: NOT outer residual. Mamba sits between GraphCast processor
# message-passing layers (mesh latent space, ~10K nodes × hidden_size).
# GraphCast is TRAINABLE (warm-start from public ckpt overlay).
#
# Verify:
#   * Mamba module loads + temporal-stateful flag wires SSM hidden carry
#   * GraphCast + Mamba train together (gradient flows through both)
#   * Sequential segment with TBPTT works
#   * res=1 m=5 (matching v3/v4 baseline for apples-to-apples comparison)

#SBATCH --job-name=v5-gcm-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_smoke_%j.err

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
  --run-name v5_gcm_res1_m5_smoke \
  --batch-size 4 \
  --max-steps 100 --eval-every 50 --eval-batch-size 4 --checkpoint-every 50 \
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

echo "SMOKE DONE"
ls -la "${OUT}/"
