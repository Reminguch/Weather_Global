#!/usr/bin/env bash
#SBATCH --job-name=ngcm-download-net
#SBATCH --account=mengdiw
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/restore_20261104/network-h200-%j.log
set -uo pipefail
hostname
date -u
getent ahosts storage.googleapis.com
curl --connect-timeout 10 --max-time 25 --head --fail --silent --show-error \
  https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zmetadata
