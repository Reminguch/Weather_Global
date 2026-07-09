#!/usr/bin/env bash
# v3 K=6 step 8500 ckpt evaluated at horizon ∈ {6, 8, 12, 24, 40}, TF mode.
#
# Pairs with the K=1 multi-horizon eval (slurm 7564720). Together the two
# slurms answer:
#   * K=1 ckpt @ {1,4,6,8}  vs  K=6 ckpt @ {6,8,12,24,40}
#   * Does the K=6-trained Mamba memory transfer to LONGER eval rollouts
#     (h=12, 24, 40) that go far beyond the K=6 training horizon?
#   * At what eval horizon does K=6's correction collapse?
#
# Layout: K=6 ckpt has cfg.anchor_as_batch=True and min_k=2, so all eval
# horizons here use anchor-as-batch ([T=K, B=S]). seg overrides keep T·B
# bounded so gpu80 (80GB) doesn't OOM:
#
#   h=6  seg=5  T·B = 30   matches K=6 training memory budget
#   h=8  seg=4  T·B = 32
#   h=12 seg=3  T·B = 36
#   h=24 seg=2  T·B = 48
#   h=40 seg=2  T·B = 80   (largest; may be tight on gpu80, watch for OOM)
#
# Wall-clock estimate: rollout_ar is a sequential T-step Python loop, so
# h=40 takes ~7× longer per segment than h=6. With 64 segments per eval:
#   h=6:  ~25 min
#   h=8:  ~30 min
#   h=12: ~40 min
#   h=24: ~80 min
#   h=40: ~140 min
# Plus XLA recompile per shape change. Total ~5-6h. Set time=8h to be safe.

#SBATCH --job-name=eval-K6-multiH
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K6_multiH_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K6_multiH_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_v3_K6_step8500_multi_horizon
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local out_name=$3
    echo "=========================================="
    echo "K=6 step 8500 ckpt -> eval horizon=${horizon}, seg=${seg}, TF"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3K6_DIR}" \
      --step 8500 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --mode tf \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# Apples-to-apples redo: fixed seg=2 across ALL horizons so K=6 row
# in the K=1-vs-K=6 comparison shares segment partitioning (and therefore
# anchor pool / time points) with the K=1 evals at the same horizons.
run_eval 6  2 "v3K6_step8500_h6_tf_seg2"
run_eval 8  2 "v3K6_step8500_h8_tf_seg2"
run_eval 12 2 "v3K6_step8500_h12_tf_seg2"
run_eval 24 2 "v3K6_step8500_h24_tf_seg2"
run_eval 40 2 "v3K6_step8500_h40_tf_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
