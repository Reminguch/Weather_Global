#!/usr/bin/env bash
# v3 K=1 step 4000 ckpt evaluated at horizon ∈ {1, 4, 6, 8}, TF mode.
#
# Hypothesis test: if the K=1-trained Mamba model can reproduce K=4/K=6
# performance just by rolling out longer at eval time, then the K-curriculum
# fine-tuning was unnecessary — the SSM hidden state would build up memory
# during the eval rollout itself.
#
# Reading the result:
#   * If Δ% Z @ h=4 with K=1 ckpt ≈ K=4 step 6000's +1.88% Z @ h=4
#     → memory emerges from long eval rollout, K=4 fine-tune adds nothing
#   * If Δ% Z @ h=4 with K=1 ckpt << +1.88% (e.g. ~+0.5%)
#     → K=4 fine-tune actually taught the model something
#
# Same logic for h=6 vs K=6 step 8500's +2.42% Z @ h=6.
#
# Layout note: K=1 PARITY chain trained with cfg.anchor_as_batch=True but
# min_k=2, so K=1 training ran in LEGACY layout (T=S, B=1). At eval h>=2
# this auto-flips to anchor-as-batch ([T=K, B=S]). This is a real layout
# mismatch (params optimized under legacy, eval uses anchor-as-batch)
# but it's how eval_per_level.py is wired today; both layouts compute the
# same MSE forward, just with different SSM hidden-state evolution semantics.
#
# Memory budget (gpu80, T·B kept ~32 via --segment-steps overrides):
#   h=1: seg=32 -> T·B = 32
#   h=4: seg=8  -> T·B = 32
#   h=6: seg=5  -> T·B = 30
#   h=8: seg=4  -> T·B = 32
# Each eval ~25-30 min on gpu80.

#SBATCH --job-name=eval-K1-multiH
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_multiH_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_multiH_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_v3_K1_step4000_multi_horizon
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local warm=$3
    local out_name=$4
    echo "=========================================="
    echo "K=1 step 4000 ckpt -> eval horizon=${horizon}, seg=${seg}, warm=${warm}, TF"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3K1_DIR}" \
      --step 4000 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --warm-start-cycles ${warm} \
      --mode tf \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# K=1 ckpt was trained with segment_steps=16 (legacy K=1 layout: SSM hidden
# propagates across 16 consecutive 6h cycles). Warm-start with 16 past cycles
# matches the training-time hidden-state distribution.
WARM=16

# Apples-to-apples: fixed seg=2 across ALL horizons so segment partitioning
# (and therefore anchor pool / time points) is identical across horizons.
# Memory budget T*B is bounded by 40*2=80 even at h=40, well within gpu80.
run_eval 1 2 ${WARM} "v3K1_step4000_h1_tf_warm${WARM}_seg2"
run_eval 4 2 ${WARM} "v3K1_step4000_h4_tf_warm${WARM}_seg2"
run_eval 6 2 ${WARM} "v3K1_step4000_h6_tf_warm${WARM}_seg2"
run_eval 8 2 ${WARM} "v3K1_step4000_h8_tf_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
