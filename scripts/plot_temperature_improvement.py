#!/usr/bin/env python3
"""Plot baseline vs MZ-corrected vs truth for temperature at a specific grid
point, with shading over time windows where MZ beats baseline.

Input: ``.npz`` produced by ``scripts/infer_mz_meshed_save_tensors.py``
Output: a PNG per selected (variable, location).

Defaults plot 2m_temperature at a small set of representative locations
(Beijing, NYC, Sydney) across all available segments, concatenated into one
timeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LOCATIONS = [
    ("Beijing", 39.9, 116.4),
    ("NYC", 40.7, 360 - 74.0),   # dataset longitude is 0..360
    ("Sydney", -33.9, 151.2),
    ("Nairobi", -1.3, 36.8),
]


def find_var_slice(feature_order, feature_slices, name):
    idx = list(feature_order).index(name)
    start, stop = int(feature_slices[idx][0]), int(feature_slices[idx][1])
    return start, stop


def nearest_idx(arr, target):
    return int(np.argmin(np.abs(arr - target)))


def plot_timeseries(
    times, base, corr, truth, city, var_label,
    level_idx: int | None, out_path: Path, pressure_level: float | None = None,
):
    """times [T], base/corr/truth [T] (1D scalar per time step)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Convert K to Celsius for 2m_T, upper-air T, and surface fields; keep units otherwise
    is_K = var_label in {"2m_temperature", "temperature"}
    if is_K:
        offset = -273.15
        base_p, corr_p, truth_p = base + offset, corr + offset, truth + offset
        ylab = "Temperature (°C)"
    else:
        base_p, corr_p, truth_p = base, corr, truth
        ylab = var_label

    # Shading where |corrected - truth| < |baseline - truth| (MZ improves)
    abs_base = np.abs(base_p - truth_p)
    abs_corr = np.abs(corr_p - truth_p)
    improve = abs_corr < abs_base

    # Collapse consecutive improve points into shaded regions
    ax1.plot(times, truth_p, color="k", lw=2.2, label="Truth (ERA5)", zorder=5)
    ax1.plot(times, base_p, color="#888", lw=1.5, ls="--", label="GraphCast baseline", alpha=0.9)
    ax1.plot(times, corr_p, color="#1f77b4", lw=1.5, label="GraphCast + MZ (FullMamba K=2)")

    # Shade improvement windows
    changes = np.diff(improve.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if s < len(times) and (e - 1) < len(times):
            ax1.axvspan(times[s], times[min(e, len(times)-1)], color="#7fbf7f", alpha=0.25, zorder=1)

    title_suffix = ""
    if pressure_level is not None:
        title_suffix = f" @ {int(pressure_level)} hPa"
    ax1.set_ylabel(ylab)
    ax1.set_title(f"{city}  —  {var_label}{title_suffix}    (green shading: MZ closer to truth than baseline)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Error magnitude subplot
    ax2.plot(times, abs_base, color="#888", lw=1.2, ls="--", label="|baseline − truth|")
    ax2.plot(times, abs_corr, color="#1f77b4", lw=1.2, label="|MZ − truth|")
    ax2.fill_between(times, abs_corr, abs_base, where=improve,
                     color="#7fbf7f", alpha=0.25, step="pre")
    ax2.set_xlabel("Date (UTC)")
    ax2.set_ylabel("|error|" + (" (°C)" if is_K else ""))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.autofmt_xdate()

    # Summary metrics in title
    base_mae = float(np.mean(abs_base))
    corr_mae = float(np.mean(abs_corr))
    improve_frac = float(np.mean(improve))
    delta = 100.0 * (base_mae - corr_mae) / base_mae if base_mae else 0.0
    fig.suptitle(
        f"{city}  —  {var_label}{title_suffix}   "
        f"|  baseline MAE = {base_mae:.3f}   MZ MAE = {corr_mae:.3f}   "
        f"Δ = {delta:+.2f}%   win-rate = {improve_frac*100:.1f}%",
        fontsize=10, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", required=True, help="Path to npz from infer_mz_meshed_save_tensors.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variable", default="2m_temperature")
    p.add_argument("--pressure-level", type=float, default=None,
                   help="For upper-air variables, pressure level (hPa). "
                        "Defaults to 850 for temperature, 500 for geopotential.")
    p.add_argument("--title-suffix", default="")
    args = p.parse_args()

    z = np.load(args.tensors, allow_pickle=True)
    feature_order = [s.decode() if isinstance(s, bytes) else str(s) for s in z["feature_order"]]
    feature_slices = z["feature_slices"]
    lat = z["lat"]
    lon = z["lon"]
    pressure_levels = list(z["pressure_levels"])
    # If the npz was produced with --locations, it stores [seg, T, 1, n_loc, F]
    # rather than the full grid. Detect and switch the iteration accordingly.
    location_names = list(z["location_names"]) if "location_names" in z.files else []
    location_mode = len(location_names) > 0

    var = args.variable
    if var not in feature_order:
        raise ValueError(
            f"Variable {var!r} not in run's feature_order {feature_order}. "
            "Use --variable with a name from the list."
        )
    start, stop = find_var_slice(feature_order, feature_slices, var)
    width = stop - start
    is_3d = width > 1

    # Default pressure level
    level_idx = None
    pressure_level = None
    if is_3d:
        if args.pressure_level is not None:
            pressure_level = args.pressure_level
        else:
            pressure_level = {"temperature": 850.0, "geopotential": 500.0,
                              "specific_humidity": 700.0}.get(var, 500.0)
        level_idx = int(np.argmin([abs(float(pl) - pressure_level) for pl in pressure_levels]))
        pressure_level = float(pressure_levels[level_idx])

    # Arrays: [seg, T, 1, lat, lon, F]
    base_arr = z["baseline"][..., start:stop]
    corr_arr = z["corrected"][..., start:stop]
    truth_arr = z["truth"][..., start:stop]
    times_arr = pd.to_datetime(z["times"].reshape(-1))

    # Slice level if 3D
    if is_3d:
        base_arr = base_arr[..., level_idx]
        corr_arr = corr_arr[..., level_idx]
        truth_arr = truth_arr[..., level_idx]
    else:
        base_arr = base_arr.squeeze(-1)
        corr_arr = corr_arr.squeeze(-1)
        truth_arr = truth_arr.squeeze(-1)

    # Shape now:
    #   full-grid mode:    [seg, T, 1, lat, lon]
    #   location mode:     [seg, T, 1, n_loc]
    base_arr = base_arr.squeeze(2)
    corr_arr = corr_arr.squeeze(2)
    truth_arr = truth_arr.squeeze(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if location_mode:
        # Iterate over the locations stored in the npz.
        for i, city in enumerate(location_names):
            base_ts = base_arr[:, :, i].reshape(-1)
            corr_ts = corr_arr[:, :, i].reshape(-1)
            truth_ts = truth_arr[:, :, i].reshape(-1)
            out_path = out_dir / f"{var}_{'p' + str(int(pressure_level)) if pressure_level else 'sfc'}_{city.replace(' ', '_')}.png"
            plot_timeseries(
                times=times_arr.to_numpy(),
                base=base_ts, corr=corr_ts, truth=truth_ts,
                city=city, var_label=var,
                level_idx=level_idx, out_path=out_path,
                pressure_level=pressure_level,
            )
    else:
        for city, city_lat, city_lon in DEFAULT_LOCATIONS:
            ilat = nearest_idx(lat, city_lat)
            ilon = nearest_idx(lon, city_lon)
            base_ts = base_arr[:, :, ilat, ilon].reshape(-1)
            corr_ts = corr_arr[:, :, ilat, ilon].reshape(-1)
            truth_ts = truth_arr[:, :, ilat, ilon].reshape(-1)
            out_path = out_dir / f"{var}_{'p' + str(int(pressure_level)) if pressure_level else 'sfc'}_{city.replace(' ', '_')}.png"
            plot_timeseries(
                times=times_arr.to_numpy(),
                base=base_ts, corr=corr_ts, truth=truth_ts,
                city=city, var_label=var,
                level_idx=level_idx, out_path=out_path,
                pressure_level=pressure_level,
            )


if __name__ == "__main__":
    main()
