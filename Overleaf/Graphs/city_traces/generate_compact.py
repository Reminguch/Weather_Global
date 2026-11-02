"""Condense the existing January city example without changing its trajectories.

Adapted from plot_cities.py in the 2026-09-25-headline-figures export.
Run with --npz to override the original export location.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_style import apply_style

SOURCE = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-09-25-headline-figures/cities_allvars_S128_K20_15yr_SWA6k-11k_15th.npz")
CITIES = ("Paris", "NYC")
ROWS = (
    ("2m_temperature", None, "2 m temperature (°C)", 1., -273.15),
    ("u_component_of_wind", 850, "U850 (m s$^{-1}$)", 1., 0.),
    ("total_precipitation_6hr", None, "6 h accumulated\nprecipitation (mm)", 1e3, 0.),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=SOURCE)
    args = parser.parse_args()
    apply_style()
    summary = []
    with np.load(args.npz, allow_pickle=False) as z:
        a = int(np.flatnonzero(z["anchor_times"].astype("datetime64[D]") == np.datetime64("2022-01-15"))[0])
        t0 = z["anchor_times"][a].astype("datetime64[s]").astype(object)
        times = list(z["valid_times"][a].astype("datetime64[s]").astype(object))
        history_times = [t0 - (times[1] - times[0]), t0]
        late = (z["valid_times"][a] - z["anchor_times"][a]) / np.timedelta64(1, "h") >= 120
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 5.4), sharex="col")
        fig.subplots_adjust(left=.125, right=.98, bottom=.085, top=.855, hspace=.29, wspace=.29)
        for c, city in enumerate(CITIES):
            ci = list(z["cities"]).index(city)
            lat, lon = float(z["city_lat"][ci]), float(z["city_lon"][ci])
            lon = lon - 360 if lon > 180 else lon
            title = "New York City" if city == "NYC" else city
            axes[0, c].set_title(f"{title} ({lat:.0f}°N, {abs(lon):.0f}°{'W' if lon < 0 else 'E'})", pad=7)
            for r, (var, level, label, scale, offset) in enumerate(ROWS):
                values = []
                for kind in ("truth", "baseline", "full", "hist"):
                    value = z[f"{kind}__{city}__{var}"][a]
                    if level is not None:
                        value = value[:, int(np.flatnonzero(z["levels"] == level)[0])]
                    values.append(value * scale + offset)
                truth, baseline, full, hist = values
                ax = axes[r, c]
                ax.grid(color="#e6e6e6", lw=.5)
                ax.spines[["top", "right"]].set_visible(False)
                for spine in ("bottom", "left"):
                    ax.spines[spine].set_color("#b0b0b0")
                    ax.spines[spine].set_linewidth(.6)
                ax.tick_params(colors="#52514e", width=.6, length=2.5)
                ax.plot(history_times + times, np.r_[hist, truth], color="#0b0b0b", lw=1.1, label="ERA5", zorder=5)
                ax.plot([t0] + times, np.r_[hist[-1], baseline], color="#D55E00", lw=1., ls=(0, (4, 1.5)), label="GraphCast", zorder=4)
                ax.plot([t0] + times, np.r_[hist[-1], full], color="#0072B2", lw=1.1, label="GraphCast–Mamba", zorder=6)
                ax.axvline(t0, color="#898781", lw=.7, ls=(0, (2, 2)))
                if c == 0:
                    ax.set_ylabel(label)
                summary.append({"city": city, "variable": var, "level_hpa": level,
                                "graphcast_mae": float(np.abs(baseline[late] - truth[late]).mean()),
                                "corrected_mae": float(np.abs(full[late] - truth[late]).mean())})
            # Label the initialization and every three days without crowded dates.
            axes[-1, c].set_xticks([t0] + times[11::12])
            axes[-1, c].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.55, .985))
        output = HERE / "cities_Paris_NYC_20220115_compact.pdf"
        fig.savefig(output)
        plt.close(fig)
    provenance = {"source": str(args.npz), "source_sha256": hashlib.sha256(args.npz.read_bytes()).hexdigest(),
                  "initialization_utc": str(t0), "mae_window_hours": [120, 240], "panels": summary}
    output.with_suffix(".sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {output}")
    print(f"Lower MAE in {sum(x['corrected_mae'] < x['graphcast_mae'] for x in summary)} of {len(summary)} panels")


if __name__ == "__main__":
    main()
