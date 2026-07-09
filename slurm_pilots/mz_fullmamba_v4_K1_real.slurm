#!/usr/bin/env bash
# v4 K=1 stateful TBPTT real training run.
#
# Warm-start from v3 K=1 PARITY step 4000 (proven-good v1-equivalent K=1 ckpt).
# v4 just trains the additional stateful TBPTT cycling on top, isolating
# the "hidden depth exposure" effect.
#
# Config:
#   len_segment = 32
#   bptt_steps  = 8        TBPTT depth
#   num_lanes   = 4        independent segments in parallel batch
#   target_steps = 1       K=1 only
#   max_steps   = 20000    (resume_step=4000 -> trains 16000 more)
#
# Speed: with batched baseline forward (single GraphCast call per chunk
# across T·B=32 positions instead of T sequential calls), expect ~10s/step.
# 20000 steps * 10s = ~55 hours = needs 3 SLURM chunks at 24h cap.

#SBATCH --job-name=mz-fm-v4-K1-real
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v4_K1_real_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v4_K1_real_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v4/train_mz_fullmamba_v4.py
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
  --run-name mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --len-segment 32 --bptt-steps 8 --num-lanes 4 \
  --chunk-load-workers 6 --data-cache-mode auto --data-cache-max-gib 80 \
  --hidden-size 128 --layers 1 \
  --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --max-steps 20000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 --lr 1e-4 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${V3K1_CKPT}" \
  --resume-step 4000 \
  --allow-partial-resume
