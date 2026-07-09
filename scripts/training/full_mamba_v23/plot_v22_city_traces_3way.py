"""Plot ERA5 truth vs baseline vs v22 K=22 (residual_mamba) for big cities × 4 seasons.

Two figures per city/var combo:
  - 2m_temperature
  - total_precipitation_6hr
Each panel: 4 anchor dates (2022-01-15, 04-15, 07-15, 10-15), 10-day rollout.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-05-city-traces")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-05-city-traces/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHORS = ["20220115","20220415","20220715","20221015"]
SEASON_LABELS = ["Jan 15 (winter)", "Apr 15 (spring)", "Jul 15 (summer)", "Oct 15 (fall)"]
CITIES = ["NYC","LA","Chicago","Tokyo","Beijing","Shanghai","London","Paris","Mumbai","Sydney"]

# Load all 4 anchors
data = {a: json.load(open(IN_DIR / f"v22_K22_cities_{a}.json")) for a in ANCHORS}
K_t = data[ANCHORS[0]]["target_steps"]
lead_days = (np.arange(1, K_t + 1) * 6) / 24.0


def plot_city_var(var, var_label, scale=1.0, ylim=None):
    n_c = len(CITIES)
    fig, axes = plt.subplots(n_c, 4, figsize=(20, 2.2*n_c), squeeze=False)
    for ci, city in enumerate(CITIES):
        for ai, anchor in enumerate(ANCHORS):
            ax = axes[ci, ai]
            d = data[anchor]
            v = d['cities'][city]['vars'][var]
            truth = np.asarray(v['truth']) * scale
            base = np.asarray(v['baseline']) * scale
            full = np.asarray(v['full']) * scale
            ax.plot(lead_days, truth, "k-", lw=2.2, label="ERA5 truth")
            ax.plot(lead_days, base, "--", color="C0", lw=1.4, label="baseline (DeepMind small)")
            ax.plot(lead_days, full, "-", color="C2", lw=1.6, label="v22 K=22 (residual Mamba)")
            err_b = np.sqrt(((base - truth)**2).mean())
            err_f = np.sqrt(((full - truth)**2).mean())
            imp = 100 * (1 - err_f / max(err_b, 1e-12))
            ax.set_title(f"{city} — {SEASON_LABELS[ai]}\n"
                         f"RMSE base={err_b:.2f}, v22={err_f:.2f} ({imp:+.1f}%)",
                         fontsize=8)
            ax.grid(alpha=0.3)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel("lead (days)", fontsize=8)
            if ai == 0: ax.set_ylabel(var_label, fontsize=9)
            if ci == 0 and ai == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{var} — ERA5 truth vs baseline vs v22 K=22 (cold_bp open-loop, 40-step rollout)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    p = OUT_DIR / f"v22_K22_city_traces_3way_{var}.png"
    plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


plot_city_var("2m_temperature", "2m T (K)")
plot_city_var("total_precipitation_6hr", "precip 6h (mm)", scale=1000.0)  # convert m to mm
