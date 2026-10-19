#!/usr/bin/env bash
#SBATCH --job-name=ngcm-local-smoke
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Original K20 smoke jobs 14937571/14937572: 7m19s/18m40s,
# peak CPU RSS 12.30/9.33 GiB. Budget applies to one smoke phase only.
set -euo pipefail
ulimit -c 0
ngcm_root="${NGCM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$ngcm_root"
phase="${1:?Expected k2-preflight, k20-preflight, k2-resume, or k20-resume}"
shift
case "$phase" in
  k2-preflight) entry=experimental/weather_only/preflight.py ;;
  k20-preflight) entry=experimental/weather_k20/preflight.py ;;
  k2-resume) entry=experimental/weather_only/lifecycle_smoke.py ;;
  k20-resume) entry=experimental/weather_k20/lifecycle_smoke.py ;;
  *) echo "Unknown smoke phase: $phase" >&2; exit 2 ;;
esac
unset PYTHONPATH PYTHONHOME
source scripts/graphcast_env.sh
source scripts/neuralgcm_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$ngcm_root/.tmp/ngcm-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
exec python -u "$entry" "$@"
