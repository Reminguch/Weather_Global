#!/usr/bin/env bash
# v2-GA K=2 smoke (10 steps, fresh init, no resume). Verifies:
#   1. Patch 1 -> [layout] effective_anchor_as_batch=True at K>=2
#   2. Patch 2 -> [layout] allow_tf_at_t0=True at K>=2
#   3. anchor-as-batch shape change ([T=2, B=S=16]) doesn't crash JIT compile
#   4. tf_mask=[1, 0] at t=0 actually injects truth_residual via tpr[0]
#      (Patch 2 r_prev branch)
#   5. residual_objective_ar_eval (closed-loop AR) still works with new code
#
# Fresh init -> no v1 ckpt resume issues. 10 steps.

#SBATCH --job-name=mz-fm-v2GA-K2-smk
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K2_smk_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K2_smk_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Fermihubbardnumeric/logs
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
  --run-name mz_fullmamba_v2_GA_SMOKE_K2 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --loss-lead-weight-power 0.0 \
  --max-steps 10 --eval-every 5 --eval-max-segments 2 --checkpoint-every 5 \
  --grad-clip 1.0 --warmup-steps 5 \
  --baseline-precision fp32 --precision bf16
