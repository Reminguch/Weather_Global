"""Plot MSLP error trajectories (pred - truth) to amplify residual visibility.
4 anchors × 10 cities × MSLP. y-axis = MSLP error in hPa.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-20-v22-city-mslp")
OUT_DIR = IN_DIR / "plots"
ANCHORS = ["20220115","20220415","20220715","20221015"]
SEASONS = ["Jan 15", "Apr 15", "Jul 15", "Oct 15"]
CITIES = ["NYC","LA","Chicago","Tokyo","Beijing","Shanghai","London","Paris","Mumbai","Sydney"]

data = {a: json.load(open(IN_DIR / f"v22_K22_cities_mslp_{a}.json")) for a in ANCHORS}
K_t = data[ANCHORS[0]]["target_steps"]
lead_days = (np.arange(1, K_t + 1) * 6) / 24.0

n_c = len(CITIES)
fig, axes = plt.subplots(n_c, 4, figsize=(20, 2.6 * n_c), squeeze=False)
for ci, city in enumerate(CITIES):
    for ai, anchor in enumerate(ANCHORS):
        ax = axes[ci, ai]
        d = data[anchor]
        v = d['cities'][city]['vars']['mean_sea_level_pressure']
        truth = np.asarray(v['truth']) / 100.0
        base = np.asarray(v['baseline']) / 100.0
        full = np.asarray(v['full']) / 100.0
        err_b = base - truth
        err_f = full - truth
        ax.axhline(0, color="k", lw=1.0, label="zero (ERA5)")
        ax.plot(lead_days, err_b, "--", color="C0", lw=1.5, label="baseline err")
        ax.plot(lead_days, err_f, "-", color="C2", lw=1.7, label="v22 K=22 err")
        ax.fill_between(lead_days, 0, err_b, color="C0", alpha=0.08)
        ax.fill_between(lead_days, 0, err_f, color="C2", alpha=0.10)
        mae_b = np.abs(err_b).mean(); mae_f = np.abs(err_f).mean()
        ax.set_title(f"{city} — {SEASONS[ai]}\n"
                     f"MAE base={mae_b:.2f}, v22={mae_f:.2f} hPa", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("lead (days)", fontsize=8)
        if ai == 0: ax.set_ylabel("MSLP error (hPa)", fontsize=9)
        if ci == 0 and ai == 0: ax.legend(fontsize=7, loc="best")

fig.suptitle("MSLP forecast error (pred − truth) — baseline vs v22 K=22\n"
             "Near-zero = perfect. v22 (green) closer to 0 = better.",
             fontsize=12, y=1.005)
plt.tight_layout()
p = OUT_DIR / "v22_K22_city_mslp_error.png"
plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"saved {p}")
