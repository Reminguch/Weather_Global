#!/usr/bin/env bash
# v4.1 K-step closed-loop residual Mamba + lead-weighted loss — smoke test.
# Verify:
#   * tf_mask = [1,0,0,0,1,0,0,0] applied (K=4 closed-loop in T=8 chunk)
#   * lead-weighted loss prints expected weight pattern
#   * 200 steps without NaN, lane reset / hidden carry still works
# Same architecture as v4 (hidden=128, layers=1) for apples-to-apples ablation.
# Warm-start from v3 K=1 step 4000 (the v4 starting point), partial-resume to
# pick up matching params (architecture identical to v4).

#SBATCH --job-name=mz-fm-v41-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v4_1/train_mz_fullmamba_v4_1.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
V3K1_CKPT=${RESULTS}/mz_fullmamba_v3_PARITY_K1_3500to6k/mz_residual_step4000.pkl

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v4_1_K4cl_smoke \
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
  --max-steps 4200 --eval-every 100 --eval-max-segments 4 --checkpoint-every 100 \
  --grad-clip 1.0 --warmup-steps 50 --lr 1e-4 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${V3K1_CKPT}" \
  --resume-step 4000 \
  --allow-partial-resume
