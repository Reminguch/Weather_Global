#!/usr/bin/env bash
# v2-GA K=1 smoke (10 steps). Verifies:
#   1. Guard 1 doesn't reject the valid combo
#   2. Guard 2 channel index permutation check passes
#   3. Patch 1 -> [layout] effective_anchor_as_batch=False at K=1 (fallback)
#   4. Patch 2 -> allow_tf_at_t0=False at K=1 (fallback consistent with Patch 1)
#   5. specialist heads (Mod G) zero-init -> early loss matches v1 ish
#   6. concat+take in _decode_from_mesh produces finite outputs

#SBATCH --job-name=mz-fm-v2GA-K1-smk
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K1_smk_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K1_smk_%j.err

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
  --run-name mz_fullmamba_v2_GA_SMOKE_K1 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
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
