#!/bin/bash
#SBATCH --job-name=gc025-score
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=gc025_score_%j.out
#SBATCH --error=gc025_score_%j.err

module purge 2>/dev/null || true
module load anaconda3/2025.6 2>/dev/null || true
if [ -f "/usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh" ]; then
  source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)" || true
fi
conda activate /scratch/gpfs/DABANIN/lm8598/home_offload/Weather_Global_experiments_conda/envs/graphcast311

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
echo "host: $(hostname)  args: $*"
python -u "${SLURM_SUBMIT_DIR}/scripts/baselines/gc025_wb2grid/gc025_rollout_score.py" "$@"
