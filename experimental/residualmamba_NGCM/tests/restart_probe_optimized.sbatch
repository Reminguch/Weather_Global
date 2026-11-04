#!/usr/bin/env bash
#SBATCH --job-name=ngcm-restart-opt
#SBATCH --account=mengdiw
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/restart-opt-%j.log
set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild
export TMPDIR=$PWD/.tmp XDG_CACHE_HOME=$PWD/.cache MPLCONFIGDIR=$PWD/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export NGCM_GPU_EXECUTION_POLICY=diagnostic_external
unset LD_LIBRARY_PATH
NGCM_PYTHON=../Weather_Global/.venv_ngcm_data/bin/python
NGCM_PROBE_ROOT=$PWD/logs/restart-opt-${SLURM_JOB_ID}
NGCM_PROBE_ARGS=(--checkpoint logs/gpu-smoke-16860678.checkpoint.pkl --gc-checkpoint data/prerequisites/graphcast_small.npz)
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/restart_probe.py "${NGCM_PROBE_ARGS[@]}" --output-root "$NGCM_PROBE_ROOT/optimized"
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/restart_probe.py "${NGCM_PROBE_ARGS[@]}" --output-root "$NGCM_PROBE_ROOT/optimized_fresh" --compare-process "$NGCM_PROBE_ROOT/optimized/reference.pkl"
