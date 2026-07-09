#!/usr/bin/env bash
# v4.1 real run: stateful TBPTT + K=4 closed-loop in chunk + lead-weighted loss.
#
# Architecture: same as v4 (hidden=128 layers=1, ~308K params Mamba head,
# frozen GraphCast small). Training change:
#   * tf_mask = [1,0,0,0,1,0,0,0]  (K=4 anchor every 4 positions)
#   * Anchor: residual head sees TRUE prev-cycle residual.
#   * Intra-anchor (lead 2..4): residual head sees its OWN previous predicted
#     residual (model self-feed) — exposes the head to deployment AR
#     distribution at training time.
#   * Loss lead-weight = (1 + lead_idx) ^ 1.0, normalised: deeper leads
#     get higher gradient weight ([0.4, 0.8, 1.2, 1.6] inside each anchor).
#
# Goal: push the truth-anchored AR cross-over from lead-3 (where v4 turned
# negative) out to lead-5 / lead-6, by training the head on its own
# self-fed residual input distribution.
#
# Warm-start: v3 K=1 step 4000 (architecture-identical, partial resume).

#SBATCH --job-name=mz-fm-v41-real
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_real_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_real_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v4_1/train_mz_fullmamba_v4_1.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
V3K1_CKPT=${RESULTS}/mz_fullmamba_v3_PARITY_K1_3500to6k/mz_residual_step4000.pkl

if [[ ! -f "${V3K1_CKPT}" ]]; then
  echo "ERROR: v3 K=1 step 4000 ckpt not found at ${V3K1_CKPT}"
  exit 1
fi

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v4_1_K1_K4cl_seg32_bptt8_lanes4_warmFromV3K1step4000_to10k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --len-segment 32 --bptt-steps 8 --num-lanes 4 \
  --target-steps-closed-loop 4 --loss-lead-weight-power 1.0 \
  --chunk-load-workers 6 --data-cache-mode auto --data-cache-max-gib 80 \
  --hidden-size 128 --layers 1 \
  --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --max-steps 10000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 --lr 1e-4 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${V3K1_CKPT}" \
  --resume-step 4000 \
  --allow-partial-resume
