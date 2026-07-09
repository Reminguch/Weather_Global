#!/usr/bin/env python3
"""Plot v3 K=2 phase eval Δ% trajectory + linear fit + permutation noise band.
Goal: visually distinguish "real but weak signal" from "pure noise"."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

K2 = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
         "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k")
K4 = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
         "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def latw_dpct(e, var):
    if var == "overall":
        b = e.get("baseline_overall_MAE_latw")
        c = e.get("corrected_overall_MAE_latw")
    else:
        b = e.get(f"baseline_{var}_RMSE_latw")
        c = e.get(f"corrected_{var}_RMSE_latw")
    if b is None or c is None or b == 0:
        return None
    return 100 * (b - c) / b


# Pull K=2 phase
ev_k2 = json.load(open(K2 / "eval_log.json"))
ev_k2 = sorted([e for e in ev_k2 if e["step"] > 4000], key=lambda e: e["step"])
steps_k2 = np.array([e["step"] for e in ev_k2])
overall_k2 = np.array([latw_dpct(e, "overall") for e in ev_k2])
z_k2 = np.array([latw_dpct(e, "geopotential") for e in ev_k2])
mslp_k2 = np.array([latw_dpct(e, "mean_sea_level_pressure") for e in ev_k2])

# Pull K=4 phase (so far)
ev_k4 = json.load(open(K4 / "eval_log.json"))
ev_k4 = sorted([e for e in ev_k4 if e["step"] > 5500], key=lambda e: e["step"])
steps_k4 = np.array([e["step"] for e in ev_k4])
overall_k4 = np.array([latw_dpct(e, "overall") for e in ev_k4])
z_k4 = np.array([latw_dpct(e, "geopotential") for e in ev_k4])
mslp_k4 = np.array([latw_dpct(e, "mean_sea_level_pressure") for e in ev_k4])


def plot_with_perm_band(ax, steps, vals, color, label, n_perm=2000, seed=42):
    """Plot trajectory + linear fit + permutation noise band (95% CI)."""
    if len(vals) < 4:
        ax.plot(steps, vals, color=color, marker="o", lw=2, markersize=7, label=label)
        return None, None
    # Linear fit
    slope, intercept = np.polyfit(steps, vals, 1)
    fit_vals = slope * steps + intercept
    # Permutation: shuffle vals, refit, collect slopes for null distribution
    rng = np.random.default_rng(seed)
    null_slopes = np.array([np.polyfit(steps, rng.permutation(vals), 1)[0] for _ in range(n_perm)])
    p_perm = (np.abs(null_slopes) >= np.abs(slope)).mean()
    # 95% noise band: shuffled trajectory's value at each step
    null_curves = np.zeros((n_perm, len(steps)))
    for i in range(n_perm):
        sh = rng.permutation(vals)
        s_, b_ = np.polyfit(steps, sh, 1)
        null_curves[i] = s_ * steps + b_
    p2_5 = np.percentile(null_curves, 2.5, axis=0)
    p97_5 = np.percentile(null_curves, 97.5, axis=0)
    # Plot trajectory
    ax.plot(steps, vals, color=color, marker="o", lw=2, markersize=7, label=label)
    # Plot linear fit
    ax.plot(steps, fit_vals, color=color, lw=1.5, ls="--", alpha=0.7,
            label=f"  linear fit (slope={slope*1000:+.3f}pp/1000step, p={p_perm:.3f})")
    # Plot 95% noise band from permutation
    ax.fill_between(steps, p2_5, p97_5, color=color, alpha=0.12,
                    label=f"  95% noise band (permutation, n={n_perm})")
    return slope, p_perm


fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Panel 1 — overall Δ%
ax = axes[0]
plot_with_perm_band(ax, steps_k2, overall_k2, "C0", "v3 K=2 phase (4200-5800)")
if len(steps_k4) >= 2:
    ax.plot(steps_k4, overall_k4, color="C3", marker="s", lw=2, markersize=7,
            label=f"v3 K=4 phase ({steps_k4[0]}-{steps_k4[-1]}, only {len(steps_k4)} evals)")
ax.axvline(5500, color="black", ls=":", lw=1, alpha=0.5, label="K=2 → K=4 boundary")
ax.set_ylabel("Eval Δ% overall MAE (lat-weighted)")
ax.set_title(
    "Is the eval Δ% trajectory real signal or noise?\n"
    "Solid line = data, dashed = linear fit, shaded = 95% noise band from permutation test\n"
    "If solid line stays mostly INSIDE shaded band, it's noise. If it CLEARLY rises above band, it's signal."
)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

# Panel 2 — Z
ax = axes[1]
plot_with_perm_band(ax, steps_k2, z_k2, "C0", "v3 K=2 phase Z")
if len(steps_k4) >= 2:
    ax.plot(steps_k4, z_k4, color="C3", marker="s", lw=2, markersize=7,
            label=f"v3 K=4 phase Z ({len(steps_k4)} evals)")
ax.axvline(5500, color="black", ls=":", lw=1, alpha=0.5)
ax.set_ylabel("Z (geopotential 13L) RMSE Δ% (latw)")
ax.set_title("Z RMSE Δ% — strongest signal across variables")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

# Panel 3 — MSLP
ax = axes[2]
plot_with_perm_band(ax, steps_k2, mslp_k2, "C0", "v3 K=2 phase MSLP")
if len(steps_k4) >= 2:
    ax.plot(steps_k4, mslp_k4, color="C3", marker="s", lw=2, markersize=7,
            label=f"v3 K=4 phase MSLP ({len(steps_k4)} evals)")
ax.axvline(5500, color="black", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Training step")
ax.set_ylabel("MSLP RMSE Δ% (latw)")
ax.set_title("MSLP RMSE Δ% — degrading (real but weak signal)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
out = OUT / "v3_K2_signal_vs_noise.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.close(fig)
