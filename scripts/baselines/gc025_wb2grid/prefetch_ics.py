#!/usr/bin/env python3
"""Prefetch FengWu initial conditions + regridded truth for our 32 anchors.

For each anchor (from the K22 eval JSON): two input frames (t0-6h, t0) as
(69, 721, 1440) float32 in FengWu channel order, saved at native 0.25 deg,
latitude NORTH->SOUTH (ERA5 native, FengWu convention; WB2 zarr is S->N so
we flip). Truth for the 40 valid times is saved for z500/t850/q700 only,
conservatively regridded to the WB2 1.5-deg grid (ascending lat).
"""
import json
import numpy as np
import pandas as pd
import xarray as xr

FW = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/fengwu"
SRC = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
JSON = ("/home/lm8598/Weather_Global_experiments/results/"
        "2026-07-02-v22cl-r1-allK-EMA-eval/v22cl_r1_K22_EMA_cold_full_zero.json")
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURF = ["10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "mean_sea_level_pressure"]
UPPER = ["geopotential", "specific_humidity", "u_component_of_wind",
         "v_component_of_wind", "temperature"]   # z, q, u, v, t
TRUTH_CH = [("geopotential", 500), ("temperature", 850), ("specific_humidity", 700)]

# --- conservative regrid 0.25 -> 1.5 (ascending lat in, ascending out) ---
def lat_edges(c):
    m = 0.5 * (c[1:] + c[:-1]); return np.concatenate([[-90.0], m, [90.0]])
def lon_edges(c):
    s = c[1] - c[0]; return np.concatenate([c - s/2, [c[-1] + s/2]])
def wlat(src, tgt):
    se, te = lat_edges(src), lat_edges(tgt)
    slo, sup = np.sin(np.deg2rad(se[:-1])), np.sin(np.deg2rad(se[1:]))
    W = np.zeros((len(tgt), len(src)))
    for j in range(len(tgt)):
        tlo, tup = np.sin(np.deg2rad(te[j])), np.sin(np.deg2rad(te[j+1]))
        W[j] = np.maximum(np.minimum(sup, tup) - np.maximum(slo, tlo), 0)
    return W / W.sum(1, keepdims=True)
def wlon(src, tgt):
    se, te = lon_edges(src), lon_edges(tgt)
    W = np.zeros((len(tgt), len(src)))
    for j in range(len(tgt)):
        tlo, tup = te[j], te[j+1]
        for sh in (-360.0, 0.0, 360.0):
            W[j] += np.maximum(np.minimum(se[1:]+sh, tup) - np.maximum(se[:-1]+sh, tlo), 0)
    return W / W.sum(1, keepdims=True)

SRC_LAT = np.arange(-90, 90.001, 0.25); SRC_LON = np.arange(0, 360, 0.25)
TGT_LAT = np.arange(-90, 90.001, 1.5);  TGT_LON = np.arange(0, 360, 1.5)
WLA, WLO = wlat(SRC_LAT, TGT_LAT), wlon(SRC_LON, TGT_LON)
def regrid(a):   # (..., 721, 1440) ascending-lat -> (..., 121, 240)
    t = np.tensordot(a, WLO.T, axes=([-1], [0]))
    t = np.tensordot(t, WLA.T, axes=([-2], [0]))
    return np.moveaxis(t, -1, -2)

def frame_to_fengwu(ds_t):
    """ds_t: single-time Dataset (ascending lat). Returns (69,721,1440) N->S."""
    chans = [np.asarray(ds_t[v].values, np.float32) for v in SURF]
    for v in UPPER:
        da = ds_t[v].sel(level=LEVELS)
        chans.extend(np.asarray(da.values, np.float32))  # (13,721,1440)
    arr = np.stack([c.squeeze() for c in chans])         # (69,721,1440)
    return arr[:, ::-1, :]                                # flip to N->S

def main():
    d = json.load(open(JSON))
    idxs = d["chosen_idx"]
    t0s = [pd.Timestamp("2022-01-01") + pd.Timedelta(hours=6*int(i)) for i in idxs]
    ds = xr.open_zarr(SRC, storage_options={"token": "anon"})
    mean = np.load(f"{FW}/repo/data_mean.npy")
    import os
    os.makedirs(f"{FW}/ics", exist_ok=True); os.makedirs(f"{FW}/truth", exist_ok=True)
    manifest = []
    for n, (i, t0) in enumerate(zip(idxs, t0s)):
        tag = t0.strftime("%Y%m%d%H")
        f1, f2 = f"{FW}/ics/{tag}_in1.npy", f"{FW}/ics/{tag}_in2.npy"
        ft = f"{FW}/truth/{tag}_truth15.npy"
        if not os.path.exists(f1):
            a1 = frame_to_fengwu(ds.sel(time=t0 - pd.Timedelta(hours=6)))
            a2 = frame_to_fengwu(ds.sel(time=t0))
            if n == 0:   # channel-order sanity vs FengWu's own data_mean
                m_ours = a2.mean((1, 2))
                print("[sanity] our ch-means vs data_mean (first 8):")
                for c in range(8):
                    print(f"   ch{c}: ours {m_ours[c]:12.2f}  fengwu {mean[c]:12.2f}")
                print(f"   ch11 (z500): ours {m_ours[11]:12.1f}  fengwu {mean[11]:12.1f}")
            np.save(f1, a1); np.save(f2, a2)
        if not os.path.exists(ft):
            frames = []
            for k in range(1, 41):
                tk = t0 + pd.Timedelta(hours=6*k)
                fr = [np.asarray(ds[v].sel(time=tk, level=lev).values, np.float64).squeeze()
                      for v, lev in TRUTH_CH]
                frames.append(regrid(np.stack(fr)))       # ascending lat in!
            np.save(ft, np.stack(frames).astype(np.float32))  # (40,3,121,240)
        manifest.append({"idx": int(i), "t0": str(t0), "tag": tag})
        print(f"[{n+1}/32] {tag} done", flush=True)
    json.dump(manifest, open(f"{FW}/manifest.json", "w"), indent=1)
    print("ALL ICS PREFETCHED")

if __name__ == "__main__":
    main()
