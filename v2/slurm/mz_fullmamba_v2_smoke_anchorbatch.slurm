#!/usr/bin/env bash
# Stage 2 smoke: K=1, --anchor-as-batch ON, no specialist heads, p=0.
#
# Purpose: verify --anchor-as-batch code path doesn't crash. The segment
# tensor shape changes from [T=S*K=16, B=1, lat, lon, F] to [T=K=1, B=S=16,
# lat, lon, F]. At K=1 the SSM time axis is degenerate (1 step, no
# recurrence), so this is a pure shape-correctness test, not a training
# behaviour test. Numerical loss may differ from Run 0 because the SSM
# hidden state propagation across anchors changes (legacy: anchors flatten
# into one length-16 sequence; anchor-as-batch: 16 independent batch
# entries with T=1 each).
#
# Bug check criteria: (1) finite loss, (2) no NaN, (3) loss in reasonable
# ballpark (~0.1 ish, similar order to Run 0). Real anchor-as-batch
# benefit only manifests at K>1.

#SBATCH --job-name=mz-fm-v2-smk-anch
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_anchorbatch_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_anchorbatch_%j.err

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
  --run-name mz_fullmamba_v2_SMOKE_anchorbatch \
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
  --anchor-as-batch \
  --loss-lead-weight-power 0.0 \
  --max-steps 10 --eval-every 5 --eval-max-segments 2 --checkpoint-every 5 \
  --grad-clip 1.0 --warmup-steps 5 \
  --baseline-precision fp32 --precision bf16
