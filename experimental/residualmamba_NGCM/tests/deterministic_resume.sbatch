#!/usr/bin/env bash
#SBATCH --job-name=ngcm-resume-fixed
#SBATCH --account=mengdiw
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/deterministic-resume-%j.log
set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild
export TMPDIR=$PWD/.tmp XDG_CACHE_HOME=$PWD/.cache MPLCONFIGDIR=$PWD/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export NGCM_GPU_EXECUTION_POLICY=deterministic_gpu_v1
export XLA_FLAGS='--xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
unset LD_LIBRARY_PATH
NGCM_PYTHON=../Weather_Global/.venv_ngcm_data/bin/python
NGCM_QA_ROOT=$PWD/logs/deterministic-resume-${SLURM_JOB_ID}
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/gpu_smoke.py --gc-checkpoint data/prerequisites/graphcast_small.npz --output "$NGCM_QA_ROOT/gpu-smoke.json"
for mode in full partial resume check; do
  "$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/training_restart_smoke.py "$mode" --root "$NGCM_QA_ROOT/trainer" --gc-checkpoint data/prerequisites/graphcast_small.npz
done
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/native_smoke.py --download-root ../Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104 --checkpoint ../Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl --output "$NGCM_QA_ROOT/native"
