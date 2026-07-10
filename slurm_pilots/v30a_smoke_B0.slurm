#!/usr/bin/env bash
# v30a SMOKE B0 — real climate-matched fresh sampler + 500 step warmup.
#
# Goal: prove per-step fresh ERA5 positives + climate-matched negatives can
# drive stable InfoNCE training WITHOUT collapsing to the "3-step memorize
# dummy" regime we saw in C2.
#
# Config:
#   adapter_scale = 0.0   (auxiliary-only phase — forecast path unchanged)
#   lambda_con    = 0.03  (mid-strength InfoNCE weight)
#   base params   FROZEN (Phase 1 mask)
#   event_head_lr = event_target_encoder_lr = 1e-4
#
# Sampler config:
#   positive:  current BPTT chunk's contrastive anchor (last), random patch center
#   negative:  same month, ≥7 days apart in time, same lat_idx as pos,
#              lon ≥ patch_size away (mod circular)
#   T_win = 1 (single target frame per tube — B0 simplification)
#   Nneg  = 4
#   patch = 16
#
# Success criteria:
#   aux_L_con starts ~log(5)=1.61 (elevated by tau=0.1 → seen ~2.3 at init)
#   500 steps: aux_L_con should decrease slowly but NOT collapse to 0
#   grad_norm stays > 0 throughout (event params still updating)
#   base loss ~ 6 (matches A0/A1)
#   no NaN
#
#SBATCH --job-name=v30a-B0
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_B0_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_B0_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
INIT_CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
OUT=${ROOT}/results/v30a_B0

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

echo "=== runtime guard: v30 + C0 + C2 + B0 patches ==="
${PYTHON} - <<'PY'
import inspect, sys
sys.path.insert(0, ".")
import scripts.training.full_mamba_v30.event_adapter_head as m
from scripts.training.full_mamba_v30 import (
    GCResidualWithEventAdapter, PerStepEventSampler)
print(f"[v30 file] {m.__file__}")
src_cls = inspect.getsource(GCResidualWithEventAdapter)
expected = ["temporal_event_damp", "effective_adapter", "has_negs",
            "self._event_patch_size", "_aux_scalar"]
missing = [pat for pat in expected if pat not in src_cls]
if missing:
    print(f"FATAL: head missing {missing}"); sys.exit(2)
# B0 sampler check
sampler_src = inspect.getsource(PerStepEventSampler)
for pat in ["_sample_neg_indices", "min_time_gap_ns", "pool_store_idx",
            "pool_month"]:
    if pat not in sampler_src:
        print(f"FATAL: sampler missing {pat}"); sys.exit(2)
# Train script check
train_src = open("scripts/training/full_mamba_v20/train_mz_v20.py").read()
for pat in ["event_sampler", "event_batch_step", "--event-t-win",
            "PerStepEventSampler"]:
    if pat not in train_src:
        print(f"FATAL: train_mz_v20 missing {pat}"); sys.exit(2)
print("[v30 guard] all patches present.")
PY

echo "=== v30a B0: 500 step contrastive-only warmup with real sampler ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name B0_climate_matched_500steps \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 500 --checkpoint-every 100 \
  --lr 3e-6 --weight-decay 0.0 --seed 30 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --event-adapter-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --adapter-scale 0.0 \
  --lambda-con 0.03 \
  --lambda-delta-reg 0.0 \
  --tau-contrastive 0.1 \
  --event-patch-size 16 \
  --event-t-win 1 \
  --event-c-vars 4 \
  --event-trunk-dim 64 --event-proj-dim 64 \
  --event-gate-bias-init -2.0 --damp-bias-init -4.0 --event-delta-cap 0.5 \
  --event-head-lr 1e-4 \
  --event-target-encoder-lr 1e-4 \
  --contrastive-anchor-idx 23 \
  --n-neg 4 \
  --event-min-time-gap-days 7

echo "=== v30a B0 done ==="
