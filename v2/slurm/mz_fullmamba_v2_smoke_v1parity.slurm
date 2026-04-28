#!/usr/bin/env bash
# v2 smoke / v1-parity test.
#
# Run train_mz_fullmamba_v2.py with all v2-only flags DISABLED:
#   - no --specialist-heads
#   - no --anchor-as-batch
#   - --loss-lead-weight-power 0.0   (uniform; same as v1)
#
# Everything else mirrors mz_fullmamba_fullvars_smoke.slurm exactly. Loss
# trajectory and eval numbers should match v1 SMOKE step-for-step (modulo
# any dataset-shuffle nondeterminism — same --seed should make this exact).
#
# Compare:
#   diff <(jq -c '.[] | {step,loss}' results/mz_residual_memory/mz_fullmamba_fullvars_SMOKE/train_log.json) \
#        <(jq -c '.[] | {step,loss}' results/mz_residual_memory/mz_fullmamba_v2_SMOKE_v1parity/train_log.json)

#SBATCH --job-name=mz-fm-v2-smk-parity
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_v1parity_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fullmamba_v2_smoke_v1parity_%j.err

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
  --run-name mz_fullmamba_v2_SMOKE_v1parity \
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
  --loss-lead-weight-power 0.0 \
  --max-steps 10 --eval-every 5 --eval-max-segments 2 --checkpoint-every 5 \
  --grad-clip 1.0 --warmup-steps 5 \
  --baseline-precision fp32 --precision bf16
