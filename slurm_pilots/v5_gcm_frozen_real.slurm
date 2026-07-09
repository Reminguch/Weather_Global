#!/usr/bin/env bash
# v5 = Ilya gc_mamba METHOD (Mamba interleaved INSIDE GC processor),
# but with our DeepMind public GraphCast_small as the FROZEN backbone.
#
# Architecture:
#   * GraphCast layers: loaded from DeepMind public GC_small via
#     overlay_matching_params (width=512, msg=16, edge_norm=0.618)
#     and FROZEN (--trainable-part mamba)
#   * Mamba interleaved at mesh-latent level inside GC processor;
#     fresh random init (no match in DeepMind ckpt); ONLY trainable params
#   * Stateful Mamba (SSM hidden carries across BPTT positions)
#
# Math note: with frozen GC, model output = GC(x) + Mamba_effect_through_decoder.
# Loss against truth pushes Mamba to learn the residual implicitly.
# Equivalent to Ilya residual_mamba (separate baseline + residual model with
# Mamba inside, both froze GC) in the limit of frozen-GC training, but with
# only ONE GC forward per step (= 2x faster than residual_mamba).
#
# Architecture matches DeepMind public ckpt exactly so overlay succeeds.

#SBATCH --job-name=v5-gcm-frozen
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_frozen_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v5_gcm_frozen_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_graphcast.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT=/home/lm8598/Weather_Global_experiments/results/v5_gcm_frozen

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v5_gcm_frozen_res1_m5_K1_stateful_seg32_to10k \
  --batch-size 1 \
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
