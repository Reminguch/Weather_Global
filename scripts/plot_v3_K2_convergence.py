#!/usr/bin/env python3
"""Plot v3 K=2 phase train loss + eval loss + eval Δ% (latw) — diagnose
whether K=2 phase had converged before OOM at step 5800."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
           "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)

K_PHASE_START = 4000


def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


tr = json.load(open(RUN / "train_log.json"))
tr.sort(key=lambda e: e["step"])
ev = json.load(open(RUN / "eval_log.json"))
ev.sort(key=lambda e: e["step"])

# Filter to K=2 phase only
tr_k2 = [e for e in tr if e["step"] > K_PHASE_START]
ev_k2 = [e for e in ev if e["step"] > K_PHASE_START]

train_steps = np.array([e["step"] for e in tr_k2])
train_losses = np.array([e["loss"] for e in tr_k2])

eval_steps = np.array([e["step"] for e in ev_k2])
eval_losses = np.array([e["total_loss"] for e in ev_k2])
eval_dpct_latw = np.array(
    [
        100 * (e["baseline_overall_MAE_latw"] - e["corrected_overall_MAE_latw"])
        / e["baseline_overall_MAE_latw"]
        for e in ev_k2
    ]
)
eval_z_latw = np.array(
    [
        100 * (e["baseline_geopotential_RMSE_latw"] - e["corrected_geopotential_RMSE_latw"])
        / e["baseline_geopotential_RMSE_latw"]
        for e in ev_k2
    ]
)
eval_mslp_latw = np.array(
    [
        100 * (e["baseline_mean_sea_level_pressure_RMSE_latw"] - e["corrected_mean_sea_level_pressure_RMSE_latw"])
        / e["baseline_mean_sea_level_pressure_RMSE_latw"]
        for e in ev_k2
    ]
)

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

# Panel 1: train loss raw (faded) + MA-50 + eval loss
ax = axes[0]
ax.plot(train_steps, train_losses, color="C0", alpha=0.2, linewidth=0.5,
        label="Train loss (raw, per step)")
sm = smooth(train_losses, 50)
sm_steps = train_steps[49:]
ax.plot(sm_steps, sm, color="C0", linewidth=2.2, label="Train loss (MA-50)")
ax.plot(eval_steps, eval_losses, color="C3", linewidth=2.5, marker="o",
        markersize=6, label="Eval total_loss (val 2022, TF mode)")
ax.set_ylabel("Loss (normalized residual MSE)")
ax.set_title("v3 K=2 phase: train loss vs eval loss\n"
             "Phase: step 4001 → 5800 (OOM at step 5800)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

# Panel 2: eval Δ% (latw) overall
ax = axes[1]
ax.plot(eval_steps, eval_dpct_latw, color="C2", linewidth=2.5, marker="s",
        markersize=6, label="Eval overall Δ% (lat-weighted)")
ax.set_ylabel("Eval Δ% overall MAE (latw)")
ax.set_title("Eval improvement (lat-weighted, paper-comparable)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

# Panel 3: per-variable Δ% (Z + MSLP)
ax = axes[2]
ax.plot(eval_steps, eval_z_latw, color="C0", linewidth=2.2, marker="o",
        markersize=5, label="Z RMSE Δ% (latw)")
ax.plot(eval_steps, eval_mslp_latw, color="C3", linewidth=2.2, marker="s",
        markersize=5, label="MSLP RMSE Δ% (latw)")
ax.set_xlabel("Training step")
ax.set_ylabel("Per-variable Δ% (latw)")
ax.set_title("Z (climb) vs MSLP (退化) trajectory")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

plt.tight_layout()
out = OUT / "v3_K2_convergence.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.close(fig)


# Numerical summary
print(f"\n--- Train loss MA-50 trajectory ---")
for s in [4100, 4500, 5000, 5500, 5800]:
    if s in sm_steps:
        i = int(np.argmin(np.abs(sm_steps - s)))
        print(f"  step {s:>5}: {sm[i]:.5f}")
print(f"\n--- Eval trajectory ---")
print(f"{'step':>5}  {'eval_loss':>10}  {'overall':>9}  {'Z':>9}  {'MSLP':>9}")
for i, s in enumerate(eval_steps):
    print(f"{s:>5}  {eval_losses[i]:>10.5f}  {eval_dpct_latw[i]:+8.3f}%  "
          f"{eval_z_latw[i]:+8.3f}%  {eval_mslp_latw[i]:+8.3f}%")

print(f"\n--- Convergence diagnosis (last 1000 K=2 steps: 4800 → 5800) ---")
last_n = 6  # eval every 200, 1000 steps = 5 evals + start eval
recent_steps = eval_steps[-last_n:]
print(f"Train MA-50 over last {recent_steps[-1] - recent_steps[0]} steps:")
sm_last = sm[(sm_steps >= recent_steps[0]) & (sm_steps <= recent_steps[-1])]
if len(sm_last) > 0:
    print(f"  start (step ~{recent_steps[0]}): {sm_last[0]:.5f}")
    print(f"  end (step ~{recent_steps[-1]}):    {sm_last[-1]:.5f}")
    print(f"  Δ_train_MA: {sm_last[-1] - sm_last[0]:+.5f}")

print(f"Eval overall Δ% trajectory (last {last_n} evals):")
for i, s in enumerate(recent_steps):
    j = len(eval_dpct_latw) - last_n + i
    print(f"  step {s:>5}: {eval_dpct_latw[j]:+.3f}%")
print(f"  net climb (last {recent_steps[-1] - recent_steps[0]} steps): "
      f"{eval_dpct_latw[-1] - eval_dpct_latw[-last_n]:+.3f}pp")
