#!/usr/bin/env python3
"""Plot v2-GA K=1 training loss (MA-100) and eval loss in one figure.
Helps visualise whether eval has plateaued vs train (overfit signal)."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
           "mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


tr = json.load(open(RUN / "train_log.json"))
tr.sort(key=lambda e: e["step"])
ev = json.load(open(RUN / "eval_log.json"))
ev.sort(key=lambda e: e["step"])

train_steps = np.array([e["step"] for e in tr])
train_losses = np.array([e["loss"] for e in tr])

eval_steps = np.array([e["step"] for e in ev])
eval_losses = np.array([e["total_loss"] for e in ev])
eval_dpct_latw = np.array(
    [
        100 * (e["baseline_overall_MAE_latw"] - e["corrected_overall_MAE_latw"])
        / e["baseline_overall_MAE_latw"]
        for e in ev
    ]
)

# --- Two panels: (1) loss curves, (2) eval Δ% latw trajectory
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Panel 1: train MA-100 + eval total_loss
sm100 = smooth(train_losses, 100)
sm100_steps = train_steps[99:]
ax1.plot(sm100_steps, sm100, color="C0", linewidth=1.8, label="Train loss (MA-100)")
ax1.plot(eval_steps, eval_losses, color="C3", linewidth=2.2, marker="o",
         markersize=5, label="Eval loss (validation 2022, TF mode)")

ax1.set_ylabel("Loss (normalized residual MSE)")
ax1.set_title(
    "v2-GA K=1 training: train loss (MA-100, blue) vs eval loss (red)\n"
    "Run: mz_fullmamba_v2_GA_paperckpt_..._K1_4k (cancelled at step 3800)"
)
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")

# Panel 2: eval Δ% (latw) — paper-comparable improvement
ax2.plot(eval_steps, eval_dpct_latw, color="C2", linewidth=2.2, marker="s",
         markersize=5, label="Eval Δ% overall MAE (lat-weighted)")
ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
ax2.set_xlabel("Training step")
ax2.set_ylabel("Eval Δ% overall MAE (latw)")
ax2.set_title("Validation improvement over baseline (paper-comparable, lat-weighted)")
ax2.grid(alpha=0.3)
ax2.legend(loc="lower right")

plt.tight_layout()
out = OUT / "v2GA_K1_train_eval_loss.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.close(fig)


# Numerical summary
print(f"\n--- Train loss MA-100 (sampled) ---")
for s in [200, 1000, 2000, 3000, 3500, 3800]:
    if s in sm100_steps:
        i = int(np.where(sm100_steps == s)[0][0])
        print(f"  step {s:>5}: {sm100[i]:.5f}")

print(f"\n--- Eval loss (val 2022) trajectory ---")
for i, s in enumerate(eval_steps):
    if s in [200, 1000, 2000, 3000, 3400, 3600, 3800]:
        print(f"  step {s:>5}: eval_loss={eval_losses[i]:.5f}  Δ%(latw)={eval_dpct_latw[i]:+.3f}%")

print(f"\n--- Convergence diagnosis (last 1000 steps eval) ---")
last_n = 6  # last 6 evals = 1000 steps (eval every 200)
ev_recent = eval_losses[-last_n:]
print(f"  eval_loss  step {eval_steps[-last_n]:>5} -> {eval_steps[-1]:>5}: "
      f"{ev_recent[0]:.5f} -> {ev_recent[-1]:.5f}  Δ={ev_recent[-1]-ev_recent[0]:+.5f}")
dpct_recent = eval_dpct_latw[-last_n:]
print(f"  Δ%(latw)   step {eval_steps[-last_n]:>5} -> {eval_steps[-1]:>5}: "
      f"{dpct_recent[0]:.3f}% -> {dpct_recent[-1]:.3f}%  Δ={dpct_recent[-1]-dpct_recent[0]:+.3f}pp")
