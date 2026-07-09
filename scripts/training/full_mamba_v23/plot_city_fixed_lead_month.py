"""Plot fixed-lead city trajectories over a month.

Reads JSON from eval_city_fixed_lead_month.py (new multi-variable format).
Renders 4 rows (leads 2d/4d/6d/8d) x 2 cols (temp + precip) per city.
4 lines per panel: truth (black), baseline (blue dashed), v22 (red), v22closed (green).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


VAR_DISPLAY = {
    "2m_temperature":         ("2m T", "K", 1.0),
    "total_precipitation_6hr": ("Precip 6h", "mm", 1000.0),  # m → mm
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True)
    p.add_argument("--out-png", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    d = json.load(open(cfg.in_json))
    leads = d["leads_days"]
    variables = d["variables"]
    has_v22cl = d.get("v22closed_at_lead") is not None or any(
        d["vars"][v].get("v22closed_at_lead") is not None for v in variables)
    valid_t = d["valid_times_per_lead"]

    fig, axes = plt.subplots(len(leads), len(variables),
                              figsize=(8 * len(variables), 3.0 * len(leads)),
                              squeeze=False)

    for row_i, k_d in enumerate(leads):
        k = str(k_d)
        vt = pd.to_datetime(valid_t[k])
        for col_i, var in enumerate(variables):
            ax = axes[row_i, col_i]
            label, unit, scale = VAR_DISPLAY.get(var, (var, "", 1.0))
            v = d["vars"][var]
            t = np.asarray(v["truth_at_lead"][k]) * scale
            b = np.asarray(v["baseline_at_lead"][k]) * scale
            f = np.asarray(v["v22_at_lead"][k]) * scale

            rmse_b = float(np.sqrt(np.mean((b - t)**2)))
            rmse_f = float(np.sqrt(np.mean((f - t)**2)))
            bias_b = float(np.mean(b - t))
            bias_f = float(np.mean(f - t))

            ax.plot(vt, t, "k-",   lw=2.2, label="ERA5 truth")
            ax.plot(vt, b, "C0--", lw=1.6, alpha=0.85,
                    label=f"baseline (RMSE={rmse_b:.2f}, bias={bias_b:+.2f})")
            ax.plot(vt, f, "C3-",  lw=1.6, alpha=0.9,
                    label=f"v22 open-loop (RMSE={rmse_f:.2f}, bias={bias_f:+.2f})")
            v22cl_data = v.get("v22closed_at_lead")
            if v22cl_data is not None:
                fc = np.asarray(v22cl_data[k]) * scale
                rmse_c = float(np.sqrt(np.mean((fc - t)**2)))
                bias_c = float(np.mean(fc - t))
                ax.plot(vt, fc, "C2-", lw=1.6, alpha=0.85,
                        label=f"v22closed K=8 (RMSE={rmse_c:.2f}, bias={bias_c:+.2f})")

            ax.set_title(f"{d['city']} — {label} at fixed lead = {k_d} days")
            ax.set_ylabel(f"{label} ({unit})")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            # Show per-panel RMSE/bias legend in EVERY panel (numbers vary by lead)
            ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Fixed-lead skill at {d['city']} — {d['variables']} over month {d['month']} {d['year']}\n"
        f"For each anchor (every {d['anchor_stride_hours']}h), warm_bp rollout, "
        f"extract value at lead k. Lines = actual vs forecasts targeting same valid_time.",
        fontsize=11)
    plt.tight_layout()
    Path(cfg.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {cfg.out_png}")


if __name__ == "__main__":
    main()
