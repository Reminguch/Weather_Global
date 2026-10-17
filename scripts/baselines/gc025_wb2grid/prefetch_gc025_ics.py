#!/usr/bin/env python3
"""Prefetch GraphCast-operational 0.25deg ICs for the SAME 32 anchors used by
the FengWu / our-model WB2-grid eval (fengwu/manifest.json). Runs on della-vis1
(internet). Per anchor: pull the 2 input frames (t0-6h, t0) of all 13 GraphCast
prognostic+static variables at native 0.25deg from the WB2 ERA5 zarr, save to
gc025_ics/{tag}.nc. Statics (geopotential_at_surface, land_sea_mask) come along.
Targets are NOT stored (AR rollout + we score vs the existing truth15 npy).
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

FW = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/fengwu")
SRC = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
OUT = FW / "gc025_ics"; OUT.mkdir(exist_ok=True)
# GraphCast operational input variables that live in the zarr (statics + prognostic).
# forcings (TISR, year/day progress) are computed at inference, not stored.
PROG = ["2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
        "10m_u_component_of_wind", "temperature", "geopotential",
        "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
        "specific_humidity"]
STATIC = ["geopotential_at_surface", "land_sea_mask"]

ds = xr.open_zarr(SRC, storage_options={"token": "anon"})
manifest = json.load(open(FW / "manifest.json"))
print(f"opened zarr; {len(manifest)} anchors")
for n, m in enumerate(manifest):
    tag = m["tag"]; t0 = pd.Timestamp(m["t0"])
    out = OUT / f"{tag}.nc"
    if out.exists():
        print(f"[{n+1}/{len(manifest)}] {tag} exists, skip"); continue
    tt = [t0 - pd.Timedelta(hours=6), t0]
    sub = ds[PROG + STATIC].sel(time=tt)
    # statics may lack a time dim already; prognostic have (time, level?, lat, lon)
    for v in sub.variables:
        sub[v].encoding = {}
    sub = sub.load()
    sub.to_netcdf(out)
    print(f"[{n+1}/{len(manifest)}] {tag} -> {out.name} ({out.stat().st_size/2**20:.0f}MB)", flush=True)
print("PREFETCH GC025 ICS: DONE")
