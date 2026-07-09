#!/usr/bin/env python3
"""Simple plot: K=1 best training loss + eval loss together."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
               "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont")

def smooth(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# Load
train = json.load(open(RUN_DIR / "train_log.json"))
train.sort(key=lambda e: e["step"])
ev = json.load(open(RUN_DIR / "eval_log.json"))
ev.sort(key=lambda e: e["step"])

train_steps = np.array([e["step"] for e in train])
train_losses = np.array([e.get("loss", e.get("total_loss", 0)) for e in train])

ev_steps = np.array([e["step"] for e in ev])
ev_corrected_loss = np.array([e["total_loss"] for e in ev])  # eval loss (TF mode loss)
ev_baseline_mae = np.array([e["baseline_overall_MAE"] for e in ev])
ev_corrected_mae = np.array([e["corrected_overall_MAE"] for e in ev])


fig, ax = plt.subplots(figsize=(10, 6))

# Training loss (raw faded + smoothed)
ax.plot(train_steps, train_losses, color="C0", alpha=0.2, linewidth=0.5,
        label="Train loss (raw, per step)")
sm = smooth(train_losses, 50)
sm_steps = train_steps[49:]
ax.plot(sm_steps, sm, color="C0", linewidth=2.5, label="Train loss (MA-50)")

# Eval loss
ax.plot(ev_steps, ev_corrected_loss, color="C3", linewidth=2.5,
        marker="o", markersize=6, label="Eval loss (validation 2022, TF mode)")

ax.set_xlabel("Training step")
ax.set_ylabel("Loss (normalized residual MSE)")
ax.set_title("K=1 best (FullMamba S6, segment=16, 4000 steps)\n"
             "Training loss vs Eval loss")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "k1_best_train_eval_simple.png"
plt.tight_layout()
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"saved: {out_path}")
plt.close()


# Print summary
print(f"\nK=1 best summary:")
print(f"  Train loss: start={train_losses[0]:.4f}  end (last 100 avg)={train_losses[-100:].mean():.4f}")
print(f"  Train loss MA(50): start={sm[0]:.4f}  end={sm[-1]:.4f}")
print(f"  Eval loss: start={ev_corrected_loss[0]:.4f}  end={ev_corrected_loss[-1]:.4f}")
print(f"\n  Eval baseline_overall_MAE: {ev_baseline_mae[0]:.4f} (constant)")
print(f"  Eval corrected_overall_MAE: start={ev_corrected_mae[0]:.4f}  end={ev_corrected_mae[-1]:.4f}")
print(f"  Improvement Δ%: {100 * (ev_baseline_mae[-1] - ev_corrected_mae[-1]) / ev_baseline_mae[-1]:+.2f}%")
