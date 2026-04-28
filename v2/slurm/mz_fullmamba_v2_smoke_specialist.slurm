#!/usr/bin/env bash
# Stage 1 smoke test: K=1, specialist-heads ON, no anchor-as-batch, p=0.
#
# Purpose: verify --specialist-heads code path doesn't crash, parameter init
# works (upper/surface heads zero-init), forward/backward shapes are correct,
# loss is finite and in roughly the same ballpark as Stage 0 (Run 0).
#
# This is a 10-step bug-check, NOT a full comparison run. If this passes, we
# decide whether to launch a longer Run 1 (e.g. 1000+ steps) for the real
# Mod G evaluation against K=1 v1 best.
#
# Run AFTER Run 0 (job 7416328) confirms v2/v1 parity — otherwise two unknowns
# at once.

#SBATCH --job-name=mz-fm-v2-smk-spec
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_specialist_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_specialist_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Fermihubbardnumeric/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba_v2.py
cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_fullmamba_v2_SMOKE_specialist \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --loss-lead-weight-power 0.0 \
  --max-steps 10 --eval-every 5 --eval-max-segments 2 --checkpoint-every 5 \
  --grad-clip 1.0 --warmup-steps 5 \
  --baseline-precision fp32 --precision bf16
