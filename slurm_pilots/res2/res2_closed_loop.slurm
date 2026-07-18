#!/usr/bin/env bash
# ============================================================================
# res=2 CLOSED-LOOP training. One of a matched trio (open / closed / SWA).
# IDENTICAL to res2_open_loop.slurm EXCEPT --feedback-mode closed_loop_sg
# (AR tail feeds back baseline + stop_gradient(residual); residual target =
# truth - sg(baseline) live). res=2 specifics: resolution 2.0 / mesh 4 /
# baseline-msg-steps 6 / GCv1-K3 baseline / warmup 200 / max-steps 2000.
# K is a parameter (--ar-tail-K ${K:-18}); same script for every K.
# ============================================================================
#SBATCH --job-name=res2-closed
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res2_closed_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res2_closed_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen        # frozen-clean tree; relative script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz   # res=2 frozen baseline
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_res2/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
[ -f "${GCV1_K3}" ] || { echo "ERR: missing GCv1 K3 baseline"; exit 1; }
OUT=${ROOT}/results/res2_sync/closed_loop_K${K:-18}
mkdir -p ${OUT}
if ls ${OUT}/K${K:-18}_res2_closed_loop/v13_residual_step*.pkl >/dev/null 2>&1; then
  echo "ERROR: ${OUT}/K${K:-18}_res2_closed_loop already has checkpoints. Move/delete before a fresh run (avoids mixing ckpts in SWA)."; exit 1
fi

echo "=== res=2 CLOSED-LOOP K=${K:-18} (feedback-mode closed_loop_sg) ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K${K:-18}_res2_closed_loop \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 --batch-size 1 \
  --max-steps 2000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${K:-18} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
echo "=== res=2 closed-loop done ==="
