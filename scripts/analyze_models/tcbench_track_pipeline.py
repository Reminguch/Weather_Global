"""Turn v22cl SWA extreme-records NC into TC tracks via TempestExtremes.

Steps for each (init_time, model) pair:
  1. Extract the (anchor_time=init, K=1..40) slab from the saver's NC.
  2. Rename vars to TempestExtremes conventions and convert units:
       mean_sea_level_pressure_{tag} → msl        (Pa; TempestExtremes handles Pa fine)
       10m_u_component_of_wind_{tag} → u10        (m/s)
       10m_v_component_of_wind_{tag} → v10        (m/s)
       geopotential_300_{tag}        → z300       (m; divide m²/s² by g=9.80665)
       geopotential_500_{tag}        → z500       (m; same)
  3. Replace (anchor_time, lead_h) with a single `time` axis carrying valid_time =
     anchor_time + lead_h*6h.
  4. Write single-init NetCDF that TempestExtremes DetectNodes can consume.
  5. Run DetectNodes → nodes.dat and StitchNodes → tracks CSV.
  6. Convert TempestExtremes tracks CSV → TCBench matched_tracks format
     (columns: SID placeholder, Initial Time, Valid Time, lat, lon,
     wind max, pressure min).

Usage: python tcbench_track_pipeline.py --nc INPUT.nc --tag {baseline,v18} \\
                                        --out-dir DIR --tempest-bin TE_BIN
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

G = 9.80665  # standard gravity for geopotential → height


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True,
                   help="Input NC produced by save_extreme_records_data_v22cl.py")
    p.add_argument("--tag", required=True, choices=["baseline", "v18", "truth"],
                   help="Which branch to track ('baseline' = pure GC self-rollout, "
                        "'v18' = residual/SWA full trajectory, 'truth' = ERA5 fields "
                        "for tracker calibration only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tempest-bin",
                   default="/home/lm8598/miniconda3/envs/tcbench/bin",
                   help="Directory containing DetectNodes and StitchNodes")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip TempestExtremes exec; only produce preprocessed NCs.")
    return p.parse_args()


def preprocess_init(ds: xr.Dataset, anc_idx: int, tag: str) -> tuple[xr.Dataset, pd.Timestamp]:
    """Produce a TempestExtremes-ready xr.Dataset for one init."""
    d = ds.isel(anchor_time=anc_idx)
    anchor = pd.Timestamp(d.anchor_time.values)
    lead_h = d.lead_h.values  # 1..40 in units of 6h steps? Let me check
    # In the saver, leads_h = np.arange(1, K+1) * 6, so 6..240 (hours)
    valid_times = np.array([np.datetime64(anchor + pd.Timedelta(hours=int(h)), "ns")
                            for h in lead_h])

    def pick(var, unit_scale=1.0):
        key = f"{var}_{tag}"
        return d[key] * unit_scale

    ds_out = xr.Dataset(
        data_vars={
            "msl": (("time", "lat", "lon"), pick("mean_sea_level_pressure").values),
            "u10": (("time", "lat", "lon"), pick("10m_u_component_of_wind").values),
            "v10": (("time", "lat", "lon"), pick("10m_v_component_of_wind").values),
            "z300": (("time", "lat", "lon"),
                     (pick("geopotential_300") / G).values),  # m²/s² → m
            "z500": (("time", "lat", "lon"),
                     (pick("geopotential_500") / G).values),
        },
        coords={
            "time": valid_times,
            "lat": d.lat.values,
            "lon": d.lon.values,
        },
    )
    return ds_out, anchor


def run_tempest(nc_path: Path, out_prefix: Path, tempest_bin: Path) -> Path | None:
    """DetectNodes + StitchNodes → CSV of tracks."""
    node_file = out_prefix.with_suffix(".nodes.dat")
    tracks_csv = out_prefix.with_suffix(".tracks.csv")

    detect_cmd = [
        str(tempest_bin / "DetectNodes"),
        "--in_data", str(nc_path),
        "--out", str(node_file),
        "--searchbymin", "msl",
        # msl closed-contour: min < 200Pa above center over 5.5° radius (Pa units)
        # warm-core: _DIFF(z300,z500) higher at center than -58.8 m over 6.5° radius
        "--closedcontourcmd", "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0",
        "--mergedist", "6.0",
        "--outputcmd", "msl,min,0;_VECMAG(u10,v10),max,2",
        "--verbosity", "0",
    ]
    print(f"  DetectNodes → {node_file.name}")
    r = subprocess.run(detect_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    DetectNodes STDERR: {r.stderr[-600:]}")
        return None

    stitch_cmd = [
        str(tempest_bin / "StitchNodes"),
        "--in", str(node_file),
        "--out", str(tracks_csv),
        "--in_fmt", "lon,lat,slp,wind10",
        "--range", "8.0",
        "--mintime", "12h",
        "--threshold", "wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
        "--out_file_format", "csv",
    ]
    print(f"  StitchNodes → {tracks_csv.name}")
    r = subprocess.run(stitch_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    StitchNodes STDERR: {r.stderr[-600:]}")
        return None
    return tracks_csv


def stitch_csv_to_tcbench(tracks_csv: Path, init_time: pd.Timestamp, out_csv: Path):
    """Convert StitchNodes CSV to TCBench matched-tracks style.

    StitchNodes CSV format:
        track_id, year, month, day, hour, i, j, lon, lat, slp, wind10
    (header on first line, one row per storm timestep)

    Output columns follow TCBench matched_tracks convention:
        SID (placeholder), Initial Time, Valid Time, lat, lon,
        wind max (kt), pressure min (hPa).
    """
    if not tracks_csv.exists() or tracks_csv.stat().st_size == 0:
        print(f"  no tracks emitted (empty CSV) → skipping conversion")
        pd.DataFrame(columns=["SID", "Initial Time", "Valid Time",
                              "lat", "lon", "wind max", "pressure min"]).to_csv(
            out_csv, index=False)
        return

    df = pd.read_csv(tracks_csv, skipinitialspace=True)
    if df.empty:
        pd.DataFrame(columns=["SID", "Initial Time", "Valid Time",
                              "lat", "lon", "wind max", "pressure min"]).to_csv(
            out_csv, index=False)
        print(f"  wrote {out_csv} (0 rows — no storms tracked)")
        return

    df["Valid Time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
    out = pd.DataFrame({
        "SID": df["track_id"].map(lambda i: f"track_{int(i):04d}"),
        "Initial Time": init_time,
        "Valid Time": df["Valid Time"],
        "lat": df["lat"],
        "lon": df["lon"],
        "wind max": df["wind10"] * 1.9438,     # m/s → knots
        "pressure min": df["slp"] / 100.0,     # Pa → hPa
    })
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({len(out)} track-points across "
          f"{df['track_id'].nunique()} tracks)")


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tempest_bin = Path(cfg.tempest_bin)

    ds = xr.open_dataset(cfg.nc)
    n_anc = ds.sizes["anchor_time"]
    print(f"Loaded {cfg.nc}: n_anchors={n_anc}, tag={cfg.tag}")

    all_tracks = []
    for ai in range(n_anc):
        ds_te, anchor = preprocess_init(ds, ai, cfg.tag)
        prefix = out_dir / f"init_{anchor.strftime('%Y%m%dT%H')}__{cfg.tag}"
        nc_path = prefix.with_suffix(".preproc.nc")
        ds_te.to_netcdf(nc_path)
        print(f"[{ai+1}/{n_anc}] init={anchor}  → {nc_path.name}")

        if cfg.dry_run:
            continue

        tracks_csv = run_tempest(nc_path, prefix, tempest_bin)
        if tracks_csv is None:
            continue
        tcbench_csv = prefix.with_name(prefix.name + ".tcbench.csv")
        stitch_csv_to_tcbench(tracks_csv, anchor, tcbench_csv)
        if tcbench_csv.exists():
            all_tracks.append(pd.read_csv(tcbench_csv))

    # Combine
    if all_tracks:
        combined = pd.concat(all_tracks, ignore_index=True)
        combined_out = out_dir / f"{cfg.tag}_all_inits.tcbench.csv"
        combined.to_csv(combined_out, index=False)
        print(f"\nCombined all inits → {combined_out} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
