"""Local diagnostic around IBTrACS centers.

Answers: "Did SWA actually smooth away the storm, or does tracker miss it?"

For each smoke init × storm (via IBTrACS best-track), at each forecast lead,
compute:
  - min MSLP within 5° radius of IBTrACS center at that valid_time
  - max 10m wind (VECMAG) within 5° radius
  - offset (km) from IBTrACS center to the local MSLP-min location

If SWA has "shallow" low compared to GC baseline (or truth) → over-smoothing.
If SWA still has deep low but tracker threshold too strict → tracker issue.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

R_EARTH_KM = 6371.0088


def gcd_km(lat1, lon1, lat2, lon2):
    """Haversine great-circle distance in km."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2*R_EARTH_KM*np.arcsin(np.sqrt(a))


def local_min_max(field2d: xr.DataArray, center_lat: float, center_lon: float,
                  radius_deg: float = 5.0):
    """Return (min_val, min_lat, min_lon, max_val, max_lat, max_lon) within
    great-circle radius (approximate via lat/lon window then GCD filter)."""
    lat = field2d.lat.values
    lon = field2d.lon.values
    LON, LAT = np.meshgrid(lon, lat)
    # Wrap-around for longitude near 0/360 boundary
    center_lon_360 = center_lon % 360
    dlon = np.minimum(np.abs(LON - center_lon_360),
                      360 - np.abs(LON - center_lon_360))
    dist = gcd_km(LAT, LON, center_lat, center_lon_360)
    mask = dist <= radius_deg * 111.0   # 1° ≈ 111 km
    if not mask.any():
        return None
    vals = field2d.values
    ii, jj = np.where(mask)
    idx_min = np.argmin(vals[ii, jj])
    idx_max = np.argmax(vals[ii, jj])
    return dict(
        min_val=float(vals[ii[idx_min], jj[idx_min]]),
        min_lat=float(LAT[ii[idx_min], jj[idx_min]]),
        min_lon=float(LON[ii[idx_min], jj[idx_min]]),
        max_val=float(vals[ii[idx_max], jj[idx_max]]),
        max_lat=float(LAT[ii[idx_max], jj[idx_max]]),
        max_lon=float(LON[ii[idx_max], jj[idx_max]]),
    )


def diagnose_storm(nc_path: Path, ibtracs_csv: Path, sid: str, init_time: pd.Timestamp,
                   radius_deg: float = 5.0):
    """Print min-MSLP / max-wind table for {truth (from IBTrACS position, forecast
    values), baseline, v18} at leads 6, 12, ..., 72h and 96, 120h."""
    ds = xr.open_dataset(nc_path)

    # Find anchor
    ai = int(np.where(ds.anchor_time.values == np.datetime64(init_time, "ns"))[0][0])
    print(f"\n{'='*100}")
    print(f"Storm SID={sid}  init={init_time}  (anchor_time idx {ai})")
    print(f"{'='*100}")

    # Load IBTrACS for this SID
    ib = pd.read_csv(ibtracs_csv, skiprows=[1], low_memory=False)
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib = ib[ib["SID"] == sid][["ISO_TIME", "LAT", "LON", "USA_WIND", "USA_PRES"]].copy()
    ib["LAT"] = pd.to_numeric(ib["LAT"], errors="coerce")
    ib["LON"] = pd.to_numeric(ib["LON"], errors="coerce")
    ib["USA_WIND"] = pd.to_numeric(ib["USA_WIND"], errors="coerce")
    ib["USA_PRES"] = pd.to_numeric(ib["USA_PRES"], errors="coerce")

    LEADS_H = [6, 12, 18, 24, 48, 72, 96, 120]
    lead_h_all = ds.lead_h.values  # array of hours

    header = (f"{'lead':>5s} | {'IBTrACS':^28s} | "
              f"{'GC baseline (min MSLP hPa / max wind m/s / offset km)':^55s} | "
              f"{'v22cl SWA cold_full (same)':^55s}")
    print(header)
    print("-" * len(header))
    for L in LEADS_H:
        # Find lead index
        if L not in lead_h_all: continue
        li = int(np.where(lead_h_all == L)[0][0])
        vt = pd.Timestamp(init_time) + pd.Timedelta(hours=L)

        # IBTrACS position at valid time
        obs = ib[ib["ISO_TIME"] == vt]
        if len(obs) == 0:
            # Try nearest within 3h
            candidates = ib[(ib["ISO_TIME"] >= vt - pd.Timedelta(hours=3)) &
                             (ib["ISO_TIME"] <= vt + pd.Timedelta(hours=3))]
            if len(candidates) == 0:
                print(f"{L:>3d}h  | {'no IBTrACS point':^28s} | {'—':^55s} | {'—':^55s}")
                continue
            obs = candidates.iloc[[0]]
        obs_lat, obs_lon = float(obs["LAT"].iloc[0]), float(obs["LON"].iloc[0])
        obs_lon_360 = obs_lon % 360
        obs_wind = obs["USA_WIND"].iloc[0]
        obs_pres = obs["USA_PRES"].iloc[0]

        def _fmt_local(tag):
            msl = ds[f"mean_sea_level_pressure_{tag}"].isel(anchor_time=ai, lead_h=li)
            u = ds[f"10m_u_component_of_wind_{tag}"].isel(anchor_time=ai, lead_h=li)
            v = ds[f"10m_v_component_of_wind_{tag}"].isel(anchor_time=ai, lead_h=li)
            wmag = np.sqrt(u.values**2 + v.values**2)
            wmag_da = xr.DataArray(wmag, coords={"lat": u.lat, "lon": u.lon}, dims=("lat","lon"))
            m = local_min_max(msl, obs_lat, obs_lon_360, radius_deg)
            w = local_min_max(wmag_da, obs_lat, obs_lon_360, radius_deg)
            if m is None or w is None:
                return f"{'—':^55s}"
            off = gcd_km(m['min_lat'], m['min_lon'], obs_lat, obs_lon_360)
            return f"{m['min_val']/100:6.1f} / {w['max_val']:5.1f} / {off:5.0f}"

        obs_str = f"{obs_lat:5.2f},{obs_lon:6.2f} W={obs_wind:>4} P={obs_pres:>4}"
        print(f"{L:>3d}h  | {obs_str:^28s} | {_fmt_local('baseline'):^55s} | {_fmt_local('v18'):^55s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True)
    p.add_argument("--ibtracs", default="/home/lm8598/Weather_Global_experiments/external/TCBench/data/ibtracs/ibtracs.ALL.list.v04r01.csv")
    p.add_argument("--sid", required=True)
    p.add_argument("--init-time", required=True, help="ISO like 2023-07-24T12:00")
    p.add_argument("--radius-deg", type=float, default=5.0)
    args = p.parse_args()
    diagnose_storm(Path(args.nc), Path(args.ibtracs), args.sid,
                   pd.Timestamp(args.init_time), args.radius_deg)


if __name__ == "__main__":
    main()
