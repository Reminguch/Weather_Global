#!/usr/bin/env bash
# ============================================================================
# res=1 CLOSED-LOOP training (K=20). One of a matched trio (open / closed / SWA)
# for cross-code synchronization. IDENTICAL to res1_open_loop.slurm EXCEPT
# --feedback-mode closed_loop_sg (closed loop: AR tail feeds back
# baseline + stop_gradient(residual); residual target = truth - sg(baseline)
# formed LIVE each step). This is the production residual-Mamba training mode.
# For a quick agreement check vs another codebase, lower --max-steps (e.g. 1000)
# and compare the training-loss trajectory.
# ============================================================================
#SBATCH --job-name=res1-closed-K20
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res1_closed_K20_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res1_closed_K20_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen        # frozen-clean tree; relative script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=${ROOT}/results/res1_sync/closed_loop_K${K:-20}
mkdir -p ${OUT}

echo "=== res=1 CLOSED-LOOP K=20 (feedback-mode closed_loop_sg) ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K${K:-20}_res1_closed_loop \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${K:-20} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
echo "=== res=1 closed-loop done ==="
