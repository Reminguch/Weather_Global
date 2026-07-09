"""Per-config eval improvement% vs training step, for res=2 K=18 arch-screen.

Reads eval JSONs from:
  arch_eval_step2000/     (initial eval @ step 2000)
  arch_eval_step10000/    (final eval @ step 10000)
  arch_eval_by_step/      (dense: step 1000,3000-9000)

Plots per-config improvement% curves at lead 108h (K=18) and 240h (extrapol).
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-v22cl-res2-arch-screen/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["inner128", "inner256", "inner512", "inner1024", "state32", "depth4", "conv8"]
LEADS = [(17, "108h (K=18)"), (39, "240h (extrap)")]
STEPS_PER_EPOCH = 424

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
SIG_DS = xr.open_dataset("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(SIG_DS[v].values if SIG_DS[v].values.ndim == 0
                  else SIG_DS[v].values.mean())
         for v in W_VAR if v in SIG_DS}


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else np.nan


def find_json(config: str, step: int) -> Path | None:
    """Look in all 3 candidate dirs."""
    for subdir in ["arch_eval_by_step",
                   f"arch_eval_step{step}",
                   "arch_eval_step2000" if step == 2000 else None,
                   "arch_eval_step10000" if step == 10000 else None]:
        if subdir is None: continue
        p = ROOT / subdir / f"{config}_step{step}_cold_full_zero.json"
        if p.exists(): return p
    return None


def load_curve(config: str, leads=LEADS):
    """Return dict {step → {lead_lbl → imp%}} for this config, all available steps."""
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    out = {}
    for step in steps:
        p = find_json(config, step)
        if p is None:
            continue
        ev = json.loads(p.read_text())
        out[step] = {lbl: mse_imp(ev, k_idx) for k_idx, lbl in leads}
    return out


fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
cmap = plt.get_cmap("tab10")

for i, config in enumerate(CONFIGS):
    curve = load_curve(config)
    if not curve:
        print(f"{config}: NO DATA")
        continue
    steps = sorted(curve.keys())
    epochs = [s / STEPS_PER_EPOCH for s in steps]
    for ax, (k_idx, lbl) in zip(axes, LEADS):
        ys = [curve[s][lbl] for s in steps]
        ax.plot(epochs, ys, "-o", ms=5, lw=1.7, color=cmap(i),
                label=f"{config}  ({len(steps)} pts, last={ys[-1]:+.2f}%)")
    print(f"{config}: {len(steps)} steps  "
          f"108h_last={curve[max(steps)][LEADS[0][1]]:+.2f}%  "
          f"240h_last={curve[max(steps)][LEADS[1][1]]:+.2f}%")

for ax, (_, lbl) in zip(axes, LEADS):
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(f"epoch (= step / {STEPS_PER_EPOCH})")
    ax.set_ylabel("paper-weighted MSE improvement% vs GCv1 K=3 baseline")
    ax.set_title(f"Eval lead = {lbl}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

fig.suptitle("v22cl res=2 K=18 arch-screen: eval improvement% vs training epoch (per config)",
             fontsize=13)
plt.tight_layout()
out = OUT_DIR / "eval_improvement_vs_epoch.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {out}")
