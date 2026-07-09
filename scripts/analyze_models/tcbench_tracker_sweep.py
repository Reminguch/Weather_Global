"""Tracker parameter sweep on the smoke NC.

Reuses the preprocessed per-init/per-tag NCs already produced by
tcbench_track_pipeline.py in tracks_{tag}/init_YYYYMMDDT12__{tag}.preproc.nc.

Runs 5 tracker configs per (init, tag), reports track length, detection rate,
and DPE against IBTrACS best-track at leads {24, 48, 72, 96, 120}h.

Choose locked config on `truth` + `baseline` sanity only (evaluation leakage
avoidance). Then re-run for v18 (SWA) with the same config for the actual
comparison.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import xarray as xr

R_EARTH_KM = 6371.0088


def gcd_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


@dataclass
class Config:
    name: str
    closedcontour: str      # DetectNodes --closedcontourcmd
    mergedist: float
    mintime: str            # StitchNodes --mintime
    stitch_range: float     # StitchNodes --range (max deg between consecutive nodes)
    threshold: str          # StitchNodes --threshold
    maxgap: str | None      # StitchNodes --maxgap (None = don't pass)


CONFIGS: list[Config] = [
    # 0) Default (TCBench's TempestExtremes_example.sh)
    Config(
        name="default",
        closedcontour="msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0",
        mergedist=6.0,
        mintime="12h",
        stitch_range=8.0,
        threshold="wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
        maxgap=None,
    ),
    # 1) Add maxgap=1: tolerate 1 missing timestep in linking
    Config(
        name="maxgap1",
        closedcontour="msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0",
        mergedist=6.0,
        mintime="12h",
        stitch_range=8.0,
        threshold="wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
        maxgap="1",
    ),
    # 2) Relax warm-core: -58.8m → -30m (accounts for 1° resolution dilution)
    Config(
        name="warmcore_relaxed",
        closedcontour="msl,200.0,5.5,0;_DIFF(z300,z500),-30.0,6.5,1.0",
        mergedist=6.0,
        mintime="12h",
        stitch_range=8.0,
        threshold="wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
        maxgap="1",
    ),
    # 3) Relax MSLP closed-contour: 200 Pa → 100 Pa
    Config(
        name="mslp_relaxed",
        closedcontour="msl,100.0,5.5,0;_DIFF(z300,z500),-30.0,6.5,1.0",
        mergedist=6.0,
        mintime="12h",
        stitch_range=8.0,
        threshold="wind10,>=,5.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
        maxgap="1",
    ),
    # 4) Very permissive: warm-core off, wind ≥ 5 m/s, larger range
    Config(
        name="permissive",
        closedcontour="msl,100.0,5.5,0",
        mergedist=6.0,
        mintime="12h",
        stitch_range=10.0,
        threshold="wind10,>=,5.0,1;lat,<=,50.0,1;lat,>=,-50.0,1",
        maxgap="2",
    ),
]


def run_tempest(nc_path: Path, out_dir: Path, cfg: Config, tempest_bin: Path) -> Path | None:
    node_file = out_dir / f"{nc_path.stem}__{cfg.name}.nodes.dat"
    tracks_csv = out_dir / f"{nc_path.stem}__{cfg.name}.tracks.csv"
    det_cmd = [
        str(tempest_bin / "DetectNodes"),
        "--in_data", str(nc_path),
        "--out", str(node_file),
        "--searchbymin", "msl",
        "--closedcontourcmd", cfg.closedcontour,
        "--mergedist", str(cfg.mergedist),
        "--outputcmd", "msl,min,0;_VECMAG(u10,v10),max,2",
        "--verbosity", "0",
    ]
    r = subprocess.run(det_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    stitch_cmd = [
        str(tempest_bin / "StitchNodes"),
        "--in", str(node_file),
        "--out", str(tracks_csv),
        "--in_fmt", "lon,lat,slp,wind10",
        "--range", str(cfg.stitch_range),
        "--mintime", cfg.mintime,
        "--threshold", cfg.threshold,
        "--out_file_format", "csv",
    ]
    if cfg.maxgap is not None:
        stitch_cmd += ["--maxgap", cfg.maxgap]
    r = subprocess.run(stitch_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return tracks_csv


def dpe_curve(tracks_csv: Path, ibtracs_df: pd.DataFrame, sid: str,
              init_time: pd.Timestamp, leads_h=(24, 48, 72, 96, 120)):
    """Compute DPE per lead by picking the track whose init-time node is closest
    to IBTrACS init position (i.e., the track that corresponds to this storm)."""
    if not tracks_csv.exists() or tracks_csv.stat().st_size == 0:
        return {L: None for L in leads_h}, 0, 0
    try:
        df = pd.read_csv(tracks_csv, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return {L: None for L in leads_h}, 0, 0
    if df.empty:
        return {L: None for L in leads_h}, 0, 0
    df["Valid Time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
    df["lon_norm"] = np.where(df["lon"] > 180, df["lon"] - 360, df["lon"])
    # Get IBTrACS position at init time
    ib_init = ibtracs_df[(ibtracs_df["SID"] == sid) &
                         (ibtracs_df["ISO_TIME"] == init_time)]
    if len(ib_init) == 0:
        # Use nearest
        cand = ibtracs_df[(ibtracs_df["SID"] == sid) &
                          (np.abs((ibtracs_df["ISO_TIME"] - init_time).dt.total_seconds()) <= 3*3600)]
        if len(cand) == 0:
            return {L: None for L in leads_h}, 0, len(df)
        ib_init = cand.iloc[[0]]
    ib_lat = float(ib_init["LAT"].iloc[0])
    ib_lon = float(ib_init["LON"].iloc[0])
    # For each track_id, find its earliest node's distance to IBTrACS init
    best_track_id = None
    best_dist = float("inf")
    for tid in df["track_id"].unique():
        sub = df[df["track_id"] == tid]
        first = sub.iloc[0]
        d = gcd_km(first["lat"], first["lon_norm"], ib_lat, ib_lon)
        if d < best_dist:
            best_dist = d; best_track_id = tid
    # If the best match is > 500 km away, storm probably not tracked
    if best_dist > 500:
        return {L: None for L in leads_h}, 0, len(df["track_id"].unique())
    matched = df[df["track_id"] == best_track_id]
    # Compute DPE at each lead
    result = {}
    for L in leads_h:
        vt = init_time + pd.Timedelta(hours=int(L))
        # Predicted point at that valid_time
        p = matched[matched["Valid Time"] == vt]
        if len(p) == 0:
            result[L] = None; continue
        obs = ibtracs_df[(ibtracs_df["SID"] == sid) &
                         (ibtracs_df["ISO_TIME"] == vt)]
        if len(obs) == 0:
            cand = ibtracs_df[(ibtracs_df["SID"] == sid) &
                              (np.abs((ibtracs_df["ISO_TIME"] - vt).dt.total_seconds()) <= 3*3600)]
            if len(cand) == 0:
                result[L] = None; continue
            obs = cand.iloc[[0]]
        dpe = gcd_km(p["lat"].iloc[0], p["lon_norm"].iloc[0],
                     obs["LAT"].iloc[0], obs["LON"].iloc[0])
        result[L] = float(dpe)
    n_pts = len(matched)
    return result, n_pts, len(df["track_id"].unique())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc-dir", required=True,
                   help="Dir with preproc NCs, e.g. /home/lm8598/.../tracks_baseline")
    p.add_argument("--tag", required=True,
                   help="baseline / v18 / truth")
    p.add_argument("--ibtracs", default="/home/lm8598/Weather_Global_experiments/external/TCBench/data/ibtracs/ibtracs.ALL.list.v04r01.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tempest-bin", default="/home/lm8598/miniconda3/envs/tcbench/bin")
    args = p.parse_args()

    tempest_bin = Path(args.tempest_bin)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load IBTrACS
    ib = pd.read_csv(args.ibtracs, skiprows=[1], low_memory=False)
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib["LAT"] = pd.to_numeric(ib["LAT"], errors="coerce")
    ib["LON"] = pd.to_numeric(ib["LON"], errors="coerce")
    ib = ib.dropna(subset=["ISO_TIME", "LAT", "LON"])

    # Fixed 3 smoke storms (SID, init_time)
    smoke_storms = [
        ("2023201N13134", "2023-07-24 12:00", "Doksuri (WP major)"),
        ("2023234N18128", "2023-08-30 00:00", "Saola (WP major)"),
        ("2023005S18142", "2023-01-09 12:00", "Hale (SH dateline)"),
    ]

    preproc_dir = Path(args.preproc_dir)
    rows = []
    for cfg in CONFIGS:
        for sid, init_str, name in smoke_storms:
            init_time = pd.Timestamp(init_str)
            nc = preproc_dir / f"init_{init_time.strftime('%Y%m%dT%H')}__{args.tag}.preproc.nc"
            if not nc.exists():
                print(f"SKIP missing: {nc}")
                continue
            tracks_csv = run_tempest(nc, out_dir, cfg, tempest_bin)
            if tracks_csv is None:
                rows.append(dict(config=cfg.name, sid=sid, storm=name, n_pts=0, n_tracks=0,
                                 dpe_24=None, dpe_48=None, dpe_72=None, dpe_96=None, dpe_120=None))
                continue
            dpes, n_pts, n_tracks = dpe_curve(tracks_csv, ib, sid, init_time)
            rows.append(dict(config=cfg.name, sid=sid, storm=name, n_pts=n_pts, n_tracks=n_tracks,
                             **{f"dpe_{k}": v for k, v in dpes.items()}))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"sweep_{args.tag}.csv", index=False)
    print(f"\n=== Tracker sweep for tag='{args.tag}' ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.0f}" if isinstance(x, float) and not pd.isna(x) else str(x)))
    print(f"\nSaved → {out_dir / f'sweep_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
