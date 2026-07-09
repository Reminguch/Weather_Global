#!/usr/bin/env python
"""Plot lat-weighted global mean traces (3 lines: ERA5 / baseline / residual+mamba)
per channel per lead step, from JSON produced by eval_save_means.py.

Outputs one PNG per channel (5 surface + 6 atmospheric x 13 levels = 83).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_JSON = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-06-01-mean-traces/v22_K22_means.json")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-06-01-mean-traces/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNITS = {
    "2m_temperature": "K",
    "temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
    "vertical_velocity": "Pa/s",
}


def unit_for(channel: str) -> str:
    base = channel.split("_level")[0]
    return UNITS.get(base, "")


def main() -> None:
    data = json.load(open(IN_JSON))
    K = data["target_steps"]
    leads_h = np.arange(1, K + 1) * 6
    chans = sorted(data["per_channel"])
    print(f"plotting {len(chans)} channels")

    for chan in chans:
        d = data["per_channel"][chan]
        truth = np.asarray(d["truth"])
        baseline = np.asarray(d["baseline"])
        full = np.asarray(d["full"])
        u = unit_for(chan)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(leads_h, truth, "k-",  lw=2.0, label="ERA5 truth")
        ax.plot(leads_h, baseline, "C0--", lw=1.6, label="baseline (GraphCast)")
        ax.plot(leads_h, full, "C3-", lw=1.6, label="residual+Mamba (v22 K=22)")
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(f"lat-weighted global mean ({u})" if u else "lat-weighted global mean")
        ax.set_title(f"{chan} — global mean over 32 anchors (2022)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        out = OUT_DIR / f"mean_{chan}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"saved {len(chans)} plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
