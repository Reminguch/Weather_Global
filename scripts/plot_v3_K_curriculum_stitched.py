#!/usr/bin/env python3
"""Stitch together the v3 K=1, K=2, K=4, K=6 phases as ONE connected curve so
the user can SEE whether each Δ% jump happens *exactly* at the K-switch step,
or gradually within each phase.

CRITICAL CAVEAT: each phase's eval runs at its own horizon=K, so:
  K=1 phase eval Δ% averages over lead +6h only
  K=2 phase eval Δ% averages over leads +6h, +12h
  K=4 phase eval Δ% averages over leads +6h..+24h
  K=6 phase eval Δ% averages over leads +6h..+36h
Because baseline RMSE grows with lead and (baseline-corrected)/baseline tends
to grow with lead too, **part of the apparent jump at each boundary is a
measurement-horizon artifact, not pure learning.** That artifact is
quantified below by also plotting the baseline-side RMSE (which jumps at
boundaries purely because horizon changes).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Each (label, dir, step_lo, step_hi) tuple describes one K phase.
# step_lo/hi defines the inclusive range of training steps that BELONG to
# this phase (so we don't double-count the seeded prefix from earlier phases).
RESULTS = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory")

PHASES = [
    ("K=1 phase",   "mz_fullmamba_v3_PARITY_K1_3500to6k",
                    0,    3500),
    ("K=2 phase",   "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k",
                    3501, 5500),
    ("K=4 phase",   "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500",
                    5501, 7200),
    ("K=6 phase",   "mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500",
                    7001, 99999),
]
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-01_K6_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def load_eval(dir_name):
    p = RESULTS / dir_name / "eval_log.json"
    if not p.exists():
        print(f"WARN missing eval_log: {p}")
        return []
    return sorted(json.load(open(p)), key=lambda e: e["step"])


def filter_phase(ev, lo, hi):
    return [e for e in ev if lo <= e["step"] <= hi]


def pull(evs, var):
    s = np.array([e["step"] for e in evs])
    if var == "overall":
        b = np.array([e.get("baseline_overall_MAE_latw", np.nan) for e in evs])
        c = np.array([e.get("corrected_overall_MAE_latw", np.nan) for e in evs])
    else:
        b = np.array([e.get(f"baseline_{var}_RMSE_latw", np.nan) for e in evs])
        c = np.array([e.get(f"corrected_{var}_RMSE_latw", np.nan) for e in evs])
    d = np.where(b > 0, 100 * (b - c) / b, np.nan)
    return s, b, c, d


# Build connected sequences across phases.
# Use COLOURS to mark phase identity, but plot with a single connecting line
# so the eye sees jumps clearly.

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
phase_colors = {"K=1 phase": "C2", "K=2 phase": "C0", "K=4 phase": "C1",
                "K=6 phase": "C3"}

for var, ax, name in [("geopotential", axes[0], "Z (geopotential) RMSE Δ% latw"),
                       ("mean_sea_level_pressure", axes[1], "MSLP RMSE Δ% latw"),
                       ("overall", axes[2], "overall MAE Δ% latw")]:

    all_steps = []
    all_d = []
    all_b = []
    phase_boundaries = []   # for vertical lines

    for label, dirn, lo, hi in PHASES:
        evs = filter_phase(load_eval(dirn), lo, hi)
        if not evs:
            print(f"  {label}: no eval points in [{lo}, {hi}]")
            continue
        s, b, c, d = pull(evs, var)
        if np.all(np.isnan(d)):
            continue
        ax.plot(s, d, color=phase_colors[label], marker="o", lw=1.8, ms=6,
                label=f"{label} (horizon=K, n={len(s)}, mean Δ%={np.nanmean(d):+.2f})")
        all_steps.extend(s.tolist())
        all_d.extend(d.tolist())
        all_b.extend(b.tolist())
        phase_boundaries.append((s[0], s[-1], label))

    # Connecting line across ALL phases (regardless of phase colour) — this
    # is what reveals jump-vs-ramp. Use a thin grey line behind the markers.
    order = np.argsort(all_steps)
    s_all = np.array(all_steps)[order]
    d_all = np.array(all_d)[order]
    ax.plot(s_all, d_all, color="black", lw=0.8, alpha=0.5, zorder=0,
            label="connected across all phases")

    # Phase-boundary verticals (use last K=N step as the boundary)
    for label, dirn, lo, hi in PHASES[1:]:
        ax.axvline(lo, color="grey", ls=":", alpha=0.5)

    ax.set_xlabel("training step")
    ax.set_ylabel(name)
    ax.set_title(f"{name} stitched across K=1 -> K=2 -> K=4 -> K=6 phases")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(
    "Per-variable eval Δ% latw, K=1 → K=2 → K=4 → K=6 (connected).\n"
    "Each K-phase eval's horizon = K, so the averaged-over-lead range *changes* at each\n"
    "phase boundary. Apparent jumps at boundaries combine (a) genuine learning at the new K\n"
    "and (b) horizon-coverage artifact (Δ% naturally grows when long leads are added).",
    y=1.00, fontsize=11
)
plt.tight_layout()
out1 = OUT / "v3_K_curriculum_stitched_dpct.png"
plt.savefig(out1, dpi=120, bbox_inches="tight")
print(f"saved: {out1}")
plt.close(fig)

# A second plot: baseline RMSE (no model involved) across phases. If baseline
# RMSE jumps at K-switch boundaries, that's a pure-artifact signal: nothing
# the MZ model is doing changed, but the eval averaging window did.
fig, ax = plt.subplots(1, 1, figsize=(13, 6))
for var, color, lbl in [("geopotential", "C0", "Z baseline RMSE latw"),
                        ("mean_sea_level_pressure", "C2", "MSLP baseline RMSE latw")]:
    all_steps = []; all_b = []
    for label, dirn, lo, hi in PHASES:
        evs = filter_phase(load_eval(dirn), lo, hi)
        if not evs: continue
        s, b, c, d = pull(evs, var)
        all_steps.extend(s.tolist()); all_b.extend(b.tolist())
    order = np.argsort(all_steps)
    s_all = np.array(all_steps)[order]; b_all = np.array(all_b)[order]
    # Normalise to first phase's mean for visibility on a 0-100% scale
    if len(b_all) and not np.isnan(b_all[0]):
        ratio = 100 * b_all / np.nanmean(b_all[:5])
        ax.plot(s_all, ratio, color=color, marker="o", lw=1.8, ms=5, label=lbl + " (% of K=1 mean)")
for label, dirn, lo, hi in PHASES[1:]:
    ax.axvline(lo, color="grey", ls=":", alpha=0.5)

ax.set_xlabel("training step")
ax.set_ylabel("baseline latw RMSE, normalised to first K=1 phase mean (=100%)")
ax.set_title(
    "Baseline (frozen GraphCast) RMSE across K phases. Baseline weights NEVER change,\n"
    "but RMSE jumps at boundaries because eval horizon = K so lead range grows. Any 'jump'\n"
    "in the baseline curve is a pure measurement artifact and an upper bound on how much of\n"
    "the corresponding Δ% jump in the previous figure is artifact rather than learning."
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
out2 = OUT / "v3_K_curriculum_baseline_horizon_artifact.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
print(f"saved: {out2}")
plt.close(fig)
