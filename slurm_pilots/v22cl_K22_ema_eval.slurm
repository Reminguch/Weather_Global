#!/usr/bin/env bash
# Eval K=22 EMA ckpt (mean of step 5000..10000).
# 4 tasks: cold_bp + cold_full × forecast_eval / h_t_rollout_diag
#SBATCH --job-name=v22cl-K22-ema
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_ema_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_ema_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
CKPT=${ROOT}/results/v22cl_r1_K22_ema_5k_10k.pkl

# Task 0/1: forecast eval cold_bp / cold_full
# Task 2/3: h_t rollout diag cold_bp / cold_full
OUT_EVAL=/home/lm8598/Weather_Global_experiments/results/2026-07-02-v22cl-r1-K22-EMA-eval
OUT_DIAG=/home/lm8598/Weather_Global_experiments/results/2026-07-02-v22cl-r1-K22-EMA-h_t
mkdir -p ${OUT_EVAL} ${OUT_DIAG}

COMMON="--ckpt ${CKPT} --ckpt-in \"${DM_CKPT}\"
  --data-path ${DATA}
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats
  --resolution 1.0 --mesh-size 5 --width 512
  --baseline-msg-steps 16 --residual-msg-steps 2
  --input-duration 12h
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 128
  --temporal-d-state 16 --temporal-d-conv 4
  --temporal-dt-rank auto --temporal-layers 2"

case $SLURM_ARRAY_TASK_ID in
  0)  # eval cold_bp
    OUT_JSON=${OUT_EVAL}/v22cl_r1_K22_EMA_cold_bp_zero.json
    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
      --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
      --data-path ${DATA} \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --resolution 1.0 --mesh-size 5 --width 512 \
      --baseline-msg-steps 16 --residual-msg-steps 2 \
      --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
      --input-duration 12h \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --eval-mode cold_bp --residual-state-init zero \
      --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
      --out-json "${OUT_JSON}"
    ;;
  1)  # eval cold_full
    OUT_JSON=${OUT_EVAL}/v22cl_r1_K22_EMA_cold_full_zero.json
    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
      --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
      --data-path ${DATA} \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --resolution 1.0 --mesh-size 5 --width 512 \
      --baseline-msg-steps 16 --residual-msg-steps 2 \
      --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
      --input-duration 12h \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --eval-mode cold_full --residual-state-init zero \
      --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
      --out-json "${OUT_JSON}"
    ;;
  2)  # h_t diag cold_bp
    $PYTHON -u scripts/training/full_mamba_v23/diag_ssm_state_rollout.py \
      --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
      --data-path ${DATA} \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --resolution 1.0 --mesh-size 5 --width 512 \
      --baseline-msg-steps 16 --residual-msg-steps 2 \
      --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
      --input-duration 12h --target-steps 40 --warmup-steps 24 \
      --anchor-date 2022-07-15T00:00:00 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --mode cold_bp --seed 0 \
      --out-dir ${OUT_DIAG}
    ;;
  3)  # h_t diag cold_full
    $PYTHON -u scripts/training/full_mamba_v23/diag_ssm_state_rollout.py \
      --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
      --data-path ${DATA} \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --resolution 1.0 --mesh-size 5 --width 512 \
      --baseline-msg-steps 16 --residual-msg-steps 2 \
      --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
      --input-duration 12h --target-steps 40 --warmup-steps 24 \
      --anchor-date 2022-07-15T00:00:00 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --mode cold_full --seed 0 \
      --out-dir ${OUT_DIAG}
    ;;
esac
echo "Done task $SLURM_ARRAY_TASK_ID"
