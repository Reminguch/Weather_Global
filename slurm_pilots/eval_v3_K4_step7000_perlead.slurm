#!/usr/bin/env bash
# D1 — per-lead breakdown for v3 K=4 step 7000 ckpt at horizon=4 TF.
# eval_per_level.py emits per_channel_per_leadtime in the JSON, so we get
# per-(variable, lead_time_h) latw Δ% — answers "is the Z +1.7% concentrated
# at +6h or does it persist to +24h, i.e. is there real long-horizon memory?"

#SBATCH --job-name=eval-perlead-K4
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_perlead_K4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_perlead_K4_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-30_K4_perlead_diagnostic
mkdir -p "${OUT_DIR}"

${PYTHON} ${SCRIPT} \
  --run-dir "${V3K4_DIR}" \
  --step 7000 \
  --num-segments 64 \
  --horizon 4 \
  --mode tf \
  --out "${OUT_DIR}/perlead_K4_step7000_h4_tf.json"

echo
echo "DONE."
ls -la "${OUT_DIR}/"
