#!/usr/bin/env python
"""Plot mean drift comparison: v22 open-loop ckpt vs v22closed ckpts.
All evaluated in closed-loop mode (full-pred feedback) with W=24 warmup.

For each key channel, 4 lines:
  - ERA5 truth (black solid)
  - baseline GraphCast self-rollout (blue dashed) — same in all 3 evals
  - v22 K=22 OPEN-LOOP ckpt under closed-loop eval (red) — expect drift
  - v22closed K=8 _sg under closed-loop eval (green) — expect controlled
  - v22closed K=10 _sg under closed-loop eval (purple)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEAN_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
                "results/2026-06-05-v22closed/mean_traces")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-06-05-v22closed/mean_drift_plots_closed_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONS = {
    "v22closed K=8 (closed train)":  "v22closed_K8_W24_closed_means.json",
    "v22closed K=10 (closed train)": "v22closed_K10_W24_closed_means.json",
}

UNITS = {
    "2m_temperature": "K", "temperature": "K",
    "10m_u_component_of_wind": "m/s", "10m_v_component_of_wind": "m/s",
    "u_component_of_wind": "m/s", "v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
    "vertical_velocity": "Pa/s",
}

KEY_CHANNELS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "vertical_velocity_level850",
    "vertical_velocity_level500",
    "geopotential_level500",
    "temperature_level850",
    "specific_humidity_level850",
    "u_component_of_wind_level850",
    "v_component_of_wind_level850",
]


def unit_for(chan: str) -> str:
    base = chan.split("_level")[0]
    return UNITS.get(base, "")


def main():
    # Load all 3 JSONs
    data = {label: json.load(open(MEAN_DIR / fn)) for label, fn in JSONS.items()}
    print(f"loaded {len(data)} eval JSONs")

    # Get all channels available
    label0 = list(data.keys())[0]
    chans_all = set(data[label0]["per_channel_per_step"].keys())
    for lbl in data:
        chans_all &= set(data[lbl]["per_channel_per_step"].keys())
    chans_all = sorted(chans_all)
    print(f"common channels: {len(chans_all)}")

    K = data[label0]["target_steps"]
    leads_h = np.arange(1, K + 1) * 6

    # === Generate plots for KEY channels only first ===
    for chan in KEY_CHANNELS:
        if chan not in chans_all:
            print(f"skip {chan} — not in eval data")
            continue
        u = unit_for(chan)

        fig, ax = plt.subplots(figsize=(10, 5.5))

        # All evals have the same truth (same 32 anchors) and same baseline (= GraphCast self-rollout)
        # So we can take truth/baseline from ANY of the 3, they should be identical.
        d0 = data[label0]["per_channel_per_step"][chan]
        ax.plot(leads_h, d0["mean_truth"], "k-", lw=2.2, label="ERA5 truth")
        ax.plot(leads_h, d0["mean_baseline"], "C0--", lw=1.6, label="baseline (GraphCast self-rollout)")

        # 2 closed-loop trained full predictions
        colors = {"v22closed K=8 (closed train)": "C2",
                  "v22closed K=10 (closed train)": "C4"}
        for label, ev in data.items():
            d = ev["per_channel_per_step"][chan]
            ax.plot(leads_h, d["mean_full"], "-", color=colors[label], lw=1.6, label=label)

        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(f"lat-weighted global mean ({u})" if u else "lat-weighted global mean")
        ax.set_title(f"{chan} — closed-loop eval, W=24 truth warmup. 32 anchors (2022).")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        out = OUT_DIR / f"meandrift_{chan}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    print(f"\nDone. {len(KEY_CHANNELS)} key-channel plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
