"""Plot raw MSLP trajectories: ERA5 truth vs baseline vs v22 K=22.
4 seasonal anchors × 10 cities, MSLP values on y-axis directly (in hPa).
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        # convert Pa to hPa
        truth = np.asarray(v['truth']) / 100.0
        base = np.asarray(v['baseline']) / 100.0
        full = np.asarray(v['full']) / 100.0
        ax.plot(lead_days, truth, "k-", lw=2.2, label="ERA5 truth")
        ax.plot(lead_days, base, "--", color="C0", lw=1.4, label="baseline")
        ax.plot(lead_days, full, "-", color="C2", lw=1.6, label="v22 K=22")
        ax.set_title(f"{city} — {SEASONS[ai]}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("lead (days)", fontsize=8)
        if ai == 0: ax.set_ylabel("MSLP (hPa)", fontsize=9)
        if ci == 0 and ai == 0: ax.legend(fontsize=8, loc="best")

fig.suptitle("MSLP — ERA5 truth vs baseline vs v22 K=22 (cold_bp open-loop, 40-step rollout)",
             fontsize=12, y=1.005)
plt.tight_layout()
p = OUT_DIR / "v22_K22_city_mslp_3way.png"
plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"saved {p}")
