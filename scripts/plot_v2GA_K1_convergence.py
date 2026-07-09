#!/usr/bin/env python3
"""Diagnose v2-GA K=1 (cancelled at step 3800) convergence.

Run dir: results/mz_residual_memory/mz_fullmamba_v2_GA_paperckpt_..._K1_4k/
Compare against v1 K=1 best (...fullvars_4k_cont) at matched steps.

Three panels:
  1. Train loss MA-200 (v2-GA K=1 vs v1 K=1)
  2. Eval overall MAE Δ% trajectory
  3. Eval per-variable RMSE Δ% (Z, MSLP) at matched steps
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

R = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory")
V1 = R / "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont"
V2GA = R / "mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k"
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def smooth(x, w=200):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load(d):
    tr = json.load(open(d / "train_log.json"))
    tr.sort(key=lambda e: e["step"])
    ev = json.load(open(d / "eval_log.json"))
    ev.sort(key=lambda e: e["step"])
    return tr, ev


tr1, ev1 = load(V1)
tr2, ev2 = load(V2GA)

s1 = np.array([e["step"] for e in tr1])
l1 = np.array([e.get("loss", e.get("total_loss", 0)) for e in tr1])
s2 = np.array([e["step"] for e in tr2])
l2 = np.array([e.get("loss", e.get("total_loss", 0)) for e in tr2])

es1 = np.array([e["step"] for e in ev1])
ed1 = np.array([100 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"]) / e["baseline_overall_MAE"] for e in ev1])
ez1 = np.array([100 * (e["baseline_geopotential_RMSE"] - e["corrected_geopotential_RMSE"]) / e["baseline_geopotential_RMSE"] for e in ev1])
em1 = np.array([100 * (e["baseline_mean_sea_level_pressure_RMSE"] - e["corrected_mean_sea_level_pressure_RMSE"]) / e["baseline_mean_sea_level_pressure_RMSE"] for e in ev1])

es2 = np.array([e["step"] for e in ev2])
ed2 = np.array([100 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"]) / e["baseline_overall_MAE"] for e in ev2])
ez2 = np.array([100 * (e["baseline_geopotential_RMSE"] - e["corrected_geopotential_RMSE"]) / e["baseline_geopotential_RMSE"] for e in ev2])
em2 = np.array([100 * (e["baseline_mean_sea_level_pressure_RMSE"] - e["corrected_mean_sea_level_pressure_RMSE"]) / e["baseline_mean_sea_level_pressure_RMSE"] for e in ev2])


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel 1: train loss MA-200
ax = axes[0, 0]
sm1 = smooth(l1, 200)
sm2 = smooth(l2, 200)
ax.plot(s1[199:], sm1, color="C0", lw=2, label=f"v1 K=1 best (4000 steps)")
ax.plot(s2[199:], sm2, color="C3", lw=2, label=f"v2-GA K=1 (cancelled @ step 3800)")
ax.axvline(3500, color="black", ls=":", lw=1, alpha=0.5, label="last v2 ckpt (step 3500)")
ax.set_xlabel("Training step")
ax.set_ylabel("Train loss (MA-200)")
ax.set_title("Train loss")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

# Panel 2: eval overall Δ%
ax = axes[0, 1]
ax.plot(es1, ed1, color="C0", lw=2, marker="o", ms=4, label="v1 K=1")
ax.plot(es2, ed2, color="C3", lw=2, marker="s", ms=4, label="v2-GA K=1")
ax.axvline(3500, color="black", ls=":", lw=1, alpha=0.5, label="step 3500 ckpt")
ax.set_xlabel("Training step")
ax.set_ylabel("Eval Δ% overall MAE (TF, val 2022)")
ax.set_title("Eval improvement: overall MAE")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# Panel 3: Z RMSE Δ%
ax = axes[1, 0]
ax.plot(es1, ez1, color="C0", lw=2, marker="o", ms=4, label="v1 K=1")
ax.plot(es2, ez2, color="C3", lw=2, marker="s", ms=4, label="v2-GA K=1")
ax.axvline(3500, color="black", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Z (geopotential 13L) RMSE Δ%")
ax.set_title("Z RMSE improvement")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# Panel 4: MSLP RMSE Δ%
ax = axes[1, 1]
ax.plot(es1, em1, color="C0", lw=2, marker="o", ms=4, label="v1 K=1")
ax.plot(es2, em2, color="C3", lw=2, marker="s", ms=4, label="v2-GA K=1")
ax.axvline(3500, color="black", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Training step")
ax.set_ylabel("MSLP RMSE Δ%")
ax.set_title("MSLP improvement (Mod G effect)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

plt.suptitle(
    "v2-GA K=1 (specialist heads + anchor-as-batch min_k=2 fallback) "
    f"vs v1 K=1 best — convergence diagnosis\n"
    f"v2-GA last ckpt = step 3500 (job CANCELLED at step 3800)",
    fontsize=11, y=1.005,
)
plt.tight_layout()
out = OUT / "v2GA_K1_convergence.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.close(fig)


# Print numeric summary
print(f"\n--- Train loss MA-200 ---")
sm1_steps = s1[199:]
sm2_steps = s2[199:]
for s in [1000, 2000, 3000, 3500, 3800]:
    if s <= sm1_steps[-1]:
        i1 = int(np.argmin(np.abs(sm1_steps - s)))
        v1str = f"{sm1[i1]:.5f}"
    else:
        v1str = "N/A"
    if s <= sm2_steps[-1]:
        i2 = int(np.argmin(np.abs(sm2_steps - s)))
        v2str = f"{sm2[i2]:.5f}"
    else:
        v2str = "N/A"
    print(f"  step {s:>5}: v1={v1str}  v2={v2str}")

print(f"\n--- Eval overall Δ% ---")
print(f"{'step':>5}  {'v1':>10}  {'v2':>10}  {'gap':>10}")
common_steps = sorted(set(es1.tolist()) & set(es2.tolist()))
for s in common_steps:
    i1 = int(np.where(es1 == s)[0][0])
    i2 = int(np.where(es2 == s)[0][0])
    print(f"{s:>5}  {ed1[i1]:+9.3f}%  {ed2[i2]:+9.3f}%  {ed2[i2]-ed1[i1]:+9.3f}pp")

print(f"\n--- Convergence check (last 1000 train steps) ---")
last_window = 1000
sm2_last = sm2[-last_window:] if len(sm2) > last_window else sm2
delta = sm2_last[-1] - sm2_last[0]
print(f"v2-GA train MA-200 over last 1000 steps: {sm2_last[0]:.5f} -> {sm2_last[-1]:.5f}  (Δ={delta:+.5f})")
print(f"v2-GA eval Δ% last 5 evals (delta vs first):")
for i, s in enumerate(es2[-5:]):
    print(f"  step {s:>5}: Δ%={ed2[len(ed2)-5+i]:+.3f}%")
