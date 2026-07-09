#!/usr/bin/env bash
# Sweep a_log_init at paper baseline + meshed m=5 h=128 seg=16.
# Submits 3 slurm jobs, one per a_log_init value, all 400 steps for quick A/B
# against the already-running 7286412 (same config, a_log_init=-0.1 default).
#
# Usage: bash mz_paperckpt_meshed_m5_h128_alog_sweep.sh

set -euo pipefail

SLURM_DIR=/home/lm8598/Fermihubbardnumeric/slurm_pilots
mkdir -p "${SLURM_DIR}"

for ALOG in -2.0 -3.0 -4.0; do
    TAG=$(echo "${ALOG}" | sed 's/\./p/;s/-/m/')   # -2.0 -> m2p0
    SLURM_FILE="${SLURM_DIR}/mz_paperckpt_m5h128_alog${TAG}.slurm"
    cat > "${SLURM_FILE}" <<EOF
#!/usr/bin/env bash
# Paper-ckpt + meshed m=5 h=128 seg=16 400 steps, a_log_init=${ALOG}.
# A/B against 7286412 (same config, a_log_init=-0.1).

#SBATCH --job-name=mz-p-alog${TAG}
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_paperckpt_m5h128_alog${TAG}_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_paperckpt_m5h128_alog${TAG}_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Fermihubbardnumeric/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "\${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_mz_residual_memory.py
cd /home/lm8598/Weather_Global_experiments

\${PYTHON} \${SCRIPT} \\
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \\
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \\
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \\
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \\
  --run-name mz_paperckpt_r1_in2_seg16_meshed_m5_h128_alog${TAG} \\
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \\
  --resolution 1.0 --mesh-size 5 \\
  --input-duration 12h \\
  --segment-steps 16 \\
  --target-steps 1 --train-mode teacher \\
  --hidden-size 128 --layers 1 \\
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \\
  --a-log-init ${ALOG} \\
  --max-steps 400 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \\
  --grad-clip 1.0 --warmup-steps 200 \\
  --baseline-precision fp32 --precision bf16
EOF
    echo "built ${SLURM_FILE}"
    sbatch "${SLURM_FILE}"
done
