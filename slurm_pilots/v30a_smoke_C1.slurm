#!/usr/bin/env bash
# v30a SMOKE C1 — predictor-level JIT smoke with REAL GraphCast forward +
# explicit-neg InfoNCE.  Single forward + grad, no BPTT, no training loop.
#
# Goals:
#   1. JIT compile passes (full v30 predictor + bf16 + DirectResidualNormalizer
#      wrappers + event_batch kwarg)
#   2. aux_L_con finite and ~log(1+Nneg)=log(5)≈1.6 at random init
#   3. encoder + event_trunk + event_proj grads NON-ZERO
#   4. event_gate / event_delta / event_damp grads ≈ 0 (because adapter_scale=0
#      means their outputs don't reach r_final, so no grad from L_base; and they
#      don't connect to L_con either since L_con only depends on event_trunk/proj
#      output via z_pred)
#   5. effective_adapter_rms = 0 exactly
#
#SBATCH --job-name=v30a-C1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_C1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_C1_%j.err
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

[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

echo "=== runtime guard: verify v30 file path + Step-1 + C0 patches ==="
${PYTHON} - <<'PY'
import inspect, sys
sys.path.insert(0, ".")
import scripts.training.full_mamba_v30.event_adapter_head as m
from scripts.training.full_mamba_v30.event_adapter_head import (
    GCResidualWithEventAdapter, info_nce_loss)
print(f"[v30 file] {m.__file__}")
expected_class = ["temporal_event_damp", "effective_adapter",
                  "force_init_target_encoder", "self._adapter_scale == 0.0",
                  "self._event_patch_size", "has_negs", "z_neg"]
src_cls = inspect.getsource(GCResidualWithEventAdapter)
missing = [pat for pat in expected_class if pat not in src_cls]
if missing:
    print(f"FATAL: missing in head: {missing}"); sys.exit(2)
# C0 InfoNCE signature
sig = inspect.signature(info_nce_loss)
if "z_neg" not in sig.parameters:
    print(f"FATAL: info_nce_loss missing z_neg param"); sys.exit(2)
print("[v30 guard] all Step-1 + C0 patches present.")
PY

echo "=== v30a C1: real GC forward + dummy event_batch + grad ==="

${PYTHON} -u scripts/training/full_mamba_v30/smoke_c1.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --adapter-scale 0.0 \
  --lambda-con 0.01 \
  --lambda-delta-reg 0.0 \
  --tau-contrastive 0.1 \
  --event-patch-size 16 \
  --event-trunk-dim 64 --event-proj-dim 64 \
  --gate-bias-init -2.0 --damp-bias-init -4.0 --delta-cap 0.5 \
  --n-neg 4 \
  --seed 30 --precision bf16

echo "=== v30a C1 done ==="
