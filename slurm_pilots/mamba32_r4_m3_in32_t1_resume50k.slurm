#!/bin/bash
# 24h auto-resume for stateless Mamba(h=32) IID long-window run.

#SBATCH --job-name=mb32-r4m3-i32-r50k
#SBATCH --account=dabanin
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/logs/mamba32_r4_m3_in32_t1_resume50k_%j.out
#SBATCH --error=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/logs/mamba32_r4_m3_in32_t1_resume50k_%j.err

PROJECT_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab
DATA_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
OUT_DIR=artifacts/checkpoints/long_iid_mamba
RUN_NAME=mamba32_r4_m3_in32_t1
RUN_DIR="${PROJECT_ROOT}/${OUT_DIR}/${RUN_NAME}"

cd "${PROJECT_ROOT}"

module purge 2>/dev/null || true
module load anaconda3/2025.6 2>/dev/null || true
if [ -f "/usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh" ]; then
  source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)" || true
fi
conda activate /home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311

set -euo pipefail
mkdir -p logs "${OUT_DIR}"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTHONPATH="${PROJECT_ROOT}/third_party/graphcast:${PYTHONPATH:-}"

LATEST_CKPT=$(ls -v "${RUN_DIR}"/ckpt_step*.npz 2>/dev/null | tail -n 1 || true)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found in ${RUN_DIR}; refusing to start a fresh duplicate run." >&2
  exit 1
fi
RESUME_STEP=$(basename "${LATEST_CKPT}" .npz | sed 's/ckpt_step//')
echo "Resuming ${RUN_NAME} from ${LATEST_CKPT} at step ${RESUME_STEP}"

python -u scripts/training/train_graphcast.py \
  --data-path "${DATA_ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr" \
  --stats-dir "${DATA_ROOT}/data/graphcast/graphcast/stats" \
  --ckpt-in "${LATEST_CKPT}" --resume-step "${RESUME_STEP}" \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 4.0 --mesh-size 3 --width 128 --processor-msg-steps 1 \
  --batch-size 1 --max-steps 50000 --eval-every 1000 --eval-batch-size 8 \
  --checkpoint-every 1000 --lr 1e-4 --weight-decay 1e-4 --seed 0 \
  --precision bf16 --input-duration 192h --target-steps 1 \
  --temporal-backbone mamba \
  --temporal-location mesh_post_processor_sequence_residual \
  --temporal-hidden-size 32 --temporal-layers 1 --temporal-dropout 0.0 \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}"
