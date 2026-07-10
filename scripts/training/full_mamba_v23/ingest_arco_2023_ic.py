"""Build a mini-zarr of GraphCast input states for a list of 2023 init times.

Source: gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
Target grid: matches existing training zarr (1°, 13 levels, 6-hourly).

For each init time t_0, ingests t_0-6h and t_0 (input_steps=2) so that
build_single_sample() can construct the GraphCast batch downstream.

Handling:
- Spatial: subset to 1° grid via sel (exact — our lat/lon are subsets of ARCO)
- Levels: 13 [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
- Total precipitation 6hr: sum ARCO hourly `total_precipitation` over the last 6h
- Toa incident solar radiation: sum ARCO hourly over the last 6h
- Static: geopotential_at_surface / land_sea_mask pulled from existing zarr (constants)

Usage:
  python ingest_arco_2023_ic.py --init-times 2023-07-24T12,2023-08-30T00,2023-01-09T12 \\
                                --out-zarr /scratch/.../era5_2023_tcbench_ic_1deg.zarr
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
REF_ZARR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
PLEVEL_VARS = [
    "temperature", "geopotential", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "specific_humidity",
]
STATIC_VARS = ["geopotential_at_surface", "land_sea_mask"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-times", required=True,
                   help="Comma-separated list of ISO init times, e.g. 2023-07-24T12,2023-08-30T00")
    p.add_argument("--out-zarr", required=True)
    p.add_argument("--input-steps", type=int, default=2,
                   help="How many prior states to fetch per init (default 2 = t0-6h + t0)")
    p.add_argument("--target-steps", type=int, default=40,
                   help="How many future 6h states to fetch (default 40 = 240h forecast horizon)")
    p.add_argument("--dt-hours", type=int, default=6)
    return p.parse_args()


def build_time_set(init_times: list[pd.Timestamp], input_steps: int,
                   target_steps: int, dt_h: int):
    """All 6-hourly stamps we need in the mini-zarr.
    Per init: prior input_steps-1 offsets, t0 itself, and target_steps future offsets.
    """
    stamps = set()
    for t0 in init_times:
        for k in range(-(input_steps - 1), target_steps + 1):
            stamps.add(t0 + pd.Timedelta(hours=k * dt_h))
    return sorted(stamps)


def main():
    cfg = parse_args()
    init_times = [pd.Timestamp(s.strip()) for s in cfg.init_times.split(",")]
    stamps = build_time_set(init_times, cfg.input_steps, cfg.target_steps, cfg.dt_hours)
    print(f"Init times: {len(init_times)}. Total unique 6h stamps to ingest: {len(stamps)}")
    print(f"  first: {stamps[0]}, last: {stamps[-1]}")

    ref = xr.open_zarr(REF_ZARR, consolidated=True)
    target_lat = ref.lat.values
    target_lon = ref.lon.values

    t0 = time.time()
    print(f"\n[1/4] Opening ARCO zarr...")
    arco = xr.open_zarr(ARCO_URL, consolidated=True, storage_options={"token": "anon"})
    print(f"  ARCO opened in {time.time()-t0:.1f}s")

    # Time subset first for all stamps at once
    stamps_np = np.array([np.datetime64(s, "ns") for s in stamps])
    # Also need hourly stamps for total_precip accumulation (last 6h before each stamp)
    hourly_stamps = set()
    for s in stamps:
        for h in range(6):
            hourly_stamps.add(pd.Timestamp(s) - pd.Timedelta(hours=h))
    hourly_np = np.array(sorted([np.datetime64(t, "ns") for t in hourly_stamps]))

    print(f"\n[2/4] Selecting 6h stamps for surface + pressure vars ({len(stamps)} times, "
          f"{len(target_lat)}x{len(target_lon)} grid, {len(TARGET_LEVELS)} levels)...")
    subset_vars = SURFACE_VARS + PLEVEL_VARS
    arco_slab = (
        arco[subset_vars]
        .sel(time=stamps_np, method=None)
        .sel(latitude=target_lat, longitude=target_lon, method="nearest")
        .sel(level=TARGET_LEVELS)
    )
    print(f"  slab dims: {dict(arco_slab.sizes)}")

    print(f"\n[3/4] Loading slab into memory (this triggers the actual download)...")
    arco_slab = arco_slab.load()
    print(f"  loaded in {time.time()-t0:.1f}s cumulative")

    # Precip: sum 6 preceding hourly values ending at each stamp (inclusive of stamp hour)
    # ERA5 total_precipitation is per-hour accumulation (m).
    print(f"\n[4/4] Computing total_precipitation_6hr accumulation from hourly ARCO data...")
    tp_hourly = (
        arco["total_precipitation"]
        .sel(time=hourly_np, method=None)
        .sel(latitude=target_lat, longitude=target_lon, method="nearest")
        .load()
    )
    tp_6h_data = np.zeros((len(stamps), len(target_lat), len(target_lon)), dtype=np.float32)
    for i, s in enumerate(stamps):
        hrs = [pd.Timestamp(s) - pd.Timedelta(hours=h) for h in range(6)]
        hrs_np = np.array([np.datetime64(t, "ns") for t in hrs])
        tp_6h_data[i] = tp_hourly.sel(time=hrs_np).sum(dim="time").values
    tp_6h = xr.DataArray(
        tp_6h_data,
        dims=("time", "latitude", "longitude"),
        coords={"time": stamps_np, "latitude": target_lat, "longitude": target_lon},
        name="total_precipitation_6hr",
    )

    # Also 6h accumulated toa incident solar radiation
    print(f"  computing toa_incident_solar_radiation 6h accumulation...")
    toa_hourly = (
        arco["toa_incident_solar_radiation"]
        .sel(time=hourly_np, method=None)
        .sel(latitude=target_lat, longitude=target_lon, method="nearest")
        .load()
    )
    toa_6h_data = np.zeros((len(stamps), len(target_lat), len(target_lon)), dtype=np.float32)
    for i, s in enumerate(stamps):
        hrs = [pd.Timestamp(s) - pd.Timedelta(hours=h) for h in range(6)]
        hrs_np = np.array([np.datetime64(t, "ns") for t in hrs])
        toa_6h_data[i] = toa_hourly.sel(time=hrs_np).sum(dim="time").values
    toa_6h = xr.DataArray(
        toa_6h_data,
        dims=("time", "latitude", "longitude"),
        coords={"time": stamps_np, "latitude": target_lat, "longitude": target_lon},
        name="toa_incident_solar_radiation",
    )

    # Assemble
    print(f"\n[assemble] Combining into mini-zarr...")
    out = arco_slab.rename({"latitude": "lat", "longitude": "lon"})
    out["total_precipitation_6hr"] = tp_6h.rename({"latitude": "lat", "longitude": "lon"})
    out["toa_incident_solar_radiation"] = toa_6h.rename({"latitude": "lat", "longitude": "lon"})
    # Static fields — pull from ref zarr (they're constants over time)
    for sv in STATIC_VARS:
        if sv in ref.data_vars:
            static = ref[sv]
            if "time" in static.dims:
                static = static.isel(time=0)
            out[sv] = static

    # Ensure dtypes match ref
    for v in out.data_vars:
        out[v] = out[v].astype(np.float32)
        # Strip source zarr v2 codec metadata; let target choose default
        out[v].encoding = {}
    for c in out.coords:
        out[c].encoding = {}

    # Save to zarr
    out_path = Path(cfg.out_zarr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  writing zarr → {out_path}")
    out.to_zarr(out_path, mode="w", consolidated=True)
    print(f"\n✓ Done. Total time {time.time()-t0:.0f}s")
    print(f"  Vars: {sorted(out.data_vars)}")
    print(f"  Dims: {dict(out.sizes)}")


if __name__ == "__main__":
    main()
