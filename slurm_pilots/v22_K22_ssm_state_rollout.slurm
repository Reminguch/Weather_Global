#!/usr/bin/env bash
# Same SSM h_t diag but on the v22 PAPER K=22 ckpt (open-loop trained).
# 2 tasks: cold_bp and cold_full — same anchor + rollout config as v22cl diag,
# so we can overlay v22cl vs v22paper h_t curves.
#SBATCH --job-name=v22paper-ssm-diag
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22paper_ssm_diag_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22paper_ssm_diag_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT=${ROOT}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-02-ssm-state-rollout-v22paper
mkdir -p ${OUT_DIR}

MODES=(cold_bp cold_full)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== SSM state rollout  v22 PAPER K=22 step 26000  mode=${MODE} ==="
$PYTHON -u scripts/training/full_mamba_v23/diag_ssm_state_rollout.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --target-steps 40 --warmup-steps 24 \
  --anchor-date 2022-07-15T00:00:00 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --mode ${MODE} \
  --seed 0 \
  --out-dir ${OUT_DIR}
echo "Done v22 paper mode=${MODE}"
