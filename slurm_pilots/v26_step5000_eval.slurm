#!/usr/bin/env bash
# v26 tail calib phase 1 step 5000 — full eval suite.
# Task 0: global cold_bp (must not regress vs v22 K=18)
# Task 1: extreme records nc save (for 3-way plot vs GC base + v22 K=18)
# Task 2: v22 K=18 tail-OFF ablation (same model but Delta_tail forced to 0)
#         — Actually skip, we already have v22 K=18 nc as "tail-off baseline"
#SBATCH --job-name=v26-step5k-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v26_step5k_eval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v26_step5k_eval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
CKPT=${ROOT}/results/v26_tail_calib/K18_tail_calib_phase1/v13_residual_step5000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-29-v26-eval
mkdir -p ${OUT_DIR}

[ -f "${CKPT}" ] || { echo "missing"; exit 1; }

COMMON="
  --ckpt ${CKPT} --ckpt-in ${GCV1_K3}
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats
  --resolution 2.0 --mesh-size 4 --width 512
  --baseline-msg-steps 6 --residual-msg-steps 2
  --val-year 2022 --train-start-year 2015 --train-end-year 2021
  --input-duration 12h
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 128
  --temporal-d-state 16 --temporal-d-conv 4
  --temporal-dt-rank auto --temporal-layers 2"

case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== Task 0: cold_bp global eval (v26 with tail-calib-head) ==="
    $PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py $COMMON \
      --tail-calib-head \
      --eval-mode cold_bp --residual-state-init zero --warmup-steps 24 \
      --target-steps 40 --n-samples 32 --seed 0 \
      --out-json ${OUT_DIR}/v26_step5000_cold_bp_zero.json
    ;;
  1)
    echo "=== Task 1: save extreme records nc (v26 with tail-calib-head) ==="
    OUT_NC=${ROOT}/results/2026-6-27-v25-extreme/v26_step5000_res2_extreme_records_K40_2022.nc
    $PYTHON -u scripts/training/full_mamba_v20/save_extreme_records_data_v20.py \
      --ckpt ${CKPT} --ckpt-in "${GCV1_K3}" \
      --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
      --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
      --resolution 2.0 --mesh-size 4 \
      --baseline-msg-steps 6 --residual-msg-steps 2 \
      --temporal-location mesh_processor_interleaved \
      --temporal-hidden-size 128 --temporal-d-inner 128 \
      --temporal-d-state 16 --temporal-d-conv 4 \
      --temporal-dt-rank auto --temporal-layers 2 \
      --target-steps 40 --anchor-stride-days 6 \
      --tail-calib-head \
      --out-nc "${OUT_NC}"
    ;;
esac
echo "Done"
