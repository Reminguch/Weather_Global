#!/usr/bin/env bash
#SBATCH --job-name=ngcm-v2-smoke
#SBATCH --account=mengdiw
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/v2-smoke-%j.log
set -euo pipefail
# Run a committed source archive, so later worktree edits cannot affect this job.
cd "${1:?Pass the absolute path to the committed source archive}"
NGCM_QA_ROOT=${2:?Pass a new absolute scratch output directory}
NGCM_WORKSPACE=/scratch/gpfs/MENGDIW/sh4809/weatherforecast
NGCM_REPO=$NGCM_WORKSPACE/Weather_Global_rebuild
NGCM_PYTHON=$NGCM_WORKSPACE/Weather_Global/.venv_ngcm_data/bin/python
mkdir -p "$NGCM_QA_ROOT/tmp" "$NGCM_QA_ROOT/cache"
export TMPDIR=$NGCM_QA_ROOT/tmp XDG_CACHE_HOME=$NGCM_QA_ROOT/cache MPLCONFIGDIR=$NGCM_QA_ROOT/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export NGCM_GPU_EXECUTION_POLICY=deterministic_gpu_v1
export XLA_FLAGS='--xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
unset LD_LIBRARY_PATH
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/gpu_smoke.py \
  --gc-checkpoint "$NGCM_REPO/data/prerequisites/graphcast_small.npz" \
  --donor-root "$NGCM_REPO/artifacts/diagnostic_2k_20261105" \
  --output "$NGCM_QA_ROOT/gpu-smoke.json"
for mode in full partial resume check; do
  "$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/training_restart_smoke.py "$mode" \
    --root "$NGCM_QA_ROOT/trainer" --gc-checkpoint "$NGCM_REPO/data/prerequisites/graphcast_small.npz"
done
"$NGCM_PYTHON" -u experimental/residualmamba_NGCM/tests/native_smoke.py \
  --download-root "$NGCM_WORKSPACE/Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104" \
  --checkpoint "$NGCM_WORKSPACE/Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl" \
  --output "$NGCM_QA_ROOT/native"
PYTHONPATH=experimental/residualmamba_NGCM "$NGCM_PYTHON" -m residualmamba_ngcm qualify \
  --config experimental/residualmamba_NGCM/configs/default.json \
  --run-root "$NGCM_QA_ROOT/qualification" \
  --gpu-smoke "$NGCM_QA_ROOT/gpu-smoke.json" \
  --native-smoke "$NGCM_QA_ROOT/native/result.json" \
  --resume-smoke "$NGCM_QA_ROOT/trainer/comparison.json"
