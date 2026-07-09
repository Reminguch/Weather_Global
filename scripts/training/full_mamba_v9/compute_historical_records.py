"""Compute per-pixel historical extremes (max / min) over train years
2015-2021 for 2m_temperature and 10m wind speed."""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("--data-path", required=True)
p.add_argument("--start-year", type=int, default=2015)
p.add_argument("--end-year", type=int, default=2021)
p.add_argument("--out-npz", required=True)
cfg = p.parse_args()

print(f"[hist-records] loading {cfg.data_path}")
ds = xr.open_zarr(cfg.data_path, consolidated=False)
# select train years
start = np.datetime64(f"{cfg.start_year}-01-01T00:00:00")
end = np.datetime64(f"{cfg.end_year}-12-31T23:59:59")
ds = ds.sel(time=slice(str(start), str(end)))
print(f"  time: {ds.time.values[0]} -> {ds.time.values[-1]}  ({ds.time.size} steps)")

# 2m_t records — per pixel max and min, monthly to reduce mem
print("[hist-records] computing 2m_t max and min by streaming month batches...")
import pandas as pd
months = pd.unique(pd.DatetimeIndex(pd.to_datetime(ds.time.values)).to_period("M"))
t2_max = None; t2_min = None
ws_max = None
for i, m in enumerate(months):
    m_start = m.to_timestamp(); m_end = (m + 1).to_timestamp() - pd.Timedelta(seconds=1)
    chunk = ds.sel(time=slice(str(m_start), str(m_end)))
    if chunk.time.size == 0: continue
    t2 = chunk['2m_temperature'].values  # (T, lon, lat)
    u10 = chunk['10m_u_component_of_wind'].values
    v10 = chunk['10m_v_component_of_wind'].values
    ws = np.sqrt(u10**2 + v10**2)
    if t2_max is None:
        t2_max = t2.max(axis=0); t2_min = t2.min(axis=0); ws_max = ws.max(axis=0)
    else:
        t2_max = np.maximum(t2_max, t2.max(axis=0))
        t2_min = np.minimum(t2_min, t2.min(axis=0))
        ws_max = np.maximum(ws_max, ws.max(axis=0))
    if (i + 1) % 12 == 0:
        print(f"  {i+1}/{len(months)} months", flush=True)

print(f"[hist-records] 2m_t  max range: [{t2_max.min():.1f}, {t2_max.max():.1f}] K")
print(f"[hist-records] 2m_t  min range: [{t2_min.min():.1f}, {t2_min.max():.1f}] K")
print(f"[hist-records] ws10  max range: [{ws_max.min():.1f}, {ws_max.max():.1f}] m/s")

out = Path(cfg.out_npz)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out,
                    t2_max=t2_max, t2_min=t2_min, ws_max=ws_max,
                    lat=ds.lat.values, lon=ds.lon.values,
                    start_year=cfg.start_year, end_year=cfg.end_year)
print(f"[hist-records] wrote {out}")
