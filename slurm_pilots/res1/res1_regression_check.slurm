#!/usr/bin/env bash
# REGRESSION: resume the ORIGINAL K20 run from its step-14000 ckpt and run 20 more
# steps in the frozen-clean tree. If steps 14001.. reproduce the original train_log
# losses (2.9269, 2.7418, 2.9054, ...), the data pipeline + trainer are byte-matched.
#SBATCH --job-name=res1-regcheck
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res1_regcheck_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res1_regcheck_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
RESUME=${ROOT}/results/v22_res1_closedloop_fresh/K20_closedloop_sg_fresh_dmckpt/v13_residual_step14000.pkl
OUT=${ROOT}/results/res1_sync/reg_${RUN_TAG:-x}
mkdir -p ${OUT}

echo "=== REGRESSION: resume K20 from step14000, run to 14020 (seed 22, warmup 0) ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --resume-from "${RESUME}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K20_reg_${RUN_TAG:-x} \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 --batch-size 1 \
  --start-step 14001 --max-steps 14010 --checkpoint-every 10 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 20 --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2
echo "=== regcheck done ==="
