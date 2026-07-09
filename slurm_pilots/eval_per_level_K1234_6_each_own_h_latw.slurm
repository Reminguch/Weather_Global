#!/usr/bin/env bash
# Each K's best ckpt evaluated on its OWN training horizon, with lat-weighted
# (cos(lat) area-proportional, GraphCast/WB2 standard) metrics. This produces
# the apples-to-apples "best result per K" comparison the paper would show:
#
#   K=1 step 4000  -> horizon=1 (6h)         FullMamba 4k_cont
#   K=2 step 5000  -> horizon=2 (12h avg)    FullMamba K2_5k
#   K=4 step 6000  -> horizon=4 (24h avg)    FullMamba K4_7k
#   K=6 step 7500  -> horizon=6 (36h avg)    FullMamba K6_8k (latest available)
#
# Both TF and AR modes per ckpt -> 8 JSONs total. Each segment is forced to
# segment_steps=8 to keep T = 8*K bounded (K=2 ckpt was trained with seg=16
# but eval uses seg=8 to match other ckpts and avoid the 26 GB OOM we hit
# before).

#SBATCH --job-name=mz-perlvl-each
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_each_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_each_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-26_per_level_eval_latw
mkdir -p "${OUT_DIR}"

K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont
K2_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k
K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k
K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k

# K=1 best on h=1 (single step; AR == TF, but we record both for consistency)
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 1 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K1_step4000_h1_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 1 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K1_step4000_h1_ar.json"

# K=2 best on h=2
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 2 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K2_step5000_h2_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 2 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K2_step5000_h2_ar.json"

# K=4 best on h=4
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K4_step6000_h4_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K4_step6000_h4_ar.json"

# K=6 latest available on h=6 (step 7500 — closest available ckpt to the
# training peak at step 7600; 8000 not yet saved at submission time)
${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 7500 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K6_step7500_h6_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 7500 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K6_step7500_h6_ar.json"
