"""Training loss curves for the res=2 K=18 arch screen (7 configs).

Combines step 1-2000 (arch-screen initial run) with step 2001+ (extension).
- Config train_log.json is overwritten by extension; for configs with intact
  step 1-2000 log, use the backup; for others (inner256, inner512), salvage
  step 1-2000 from stdout of job 10719736.
- Then append current extension entries (step 2001-...) from the live
  train_log.json.

x-axis: epoch (= step / 424 for res=2 K=18 arch-screen; 424 steps/epoch was
        reported in the arch-screen stdout).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS_PER_EPOCH = 424
BAK = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/_arch_step0-2000_logs")
LOG_DIR = Path("/home/lm8598/Weather_Global_experiments/logs")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-v22cl-res2-arch-screen/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("inner128",   "K18_res2_closedloop_sg_fresh",  0),
    ("inner256",  "K18_res2_inner256",             1),
    ("inner512",  "K18_res2_inner512",             2),
    ("inner1024", "K18_res2_inner1024",            3),
    ("state32",   "K18_res2_state32",              4),
    ("depth4",    "K18_res2_depth4",               5),
    ("conv8",     "K18_res2_conv8",                6),
]
BASE_JOBID = "10719736"  # arch-screen job (writes step 1-2000)


def parse_stdout_losses(stdout_path: Path):
    """Extract (step, loss) pairs from training stdout `step N/M loss ...`."""
    if not stdout_path.exists():
        return []
    out = []
    pat = re.compile(r"step (\d+)/\d+ loss ([\d.]+)")
    for m in pat.finditer(stdout_path.read_text()):
        out.append((int(m.group(1)), float(m.group(2))))
    return out


def load_config_loss_curve(name, dir_name, task_idx):
    """Return list of (step, loss) covering step 1..current, combining sources."""
    steps_losses = []

    bak = BAK / f"{name}_step1-2000_train_log.json"
    if bak.exists():
        d = json.load(open(bak))
        for entry in d:
            steps_losses.append((entry["step"], entry["loss"]))
    else:
        # Salvage from stdout
        stdout = LOG_DIR / f"v22cl_res2_arch_t{task_idx}_{BASE_JOBID}.out"
        salvage = parse_stdout_losses(stdout)
        # keep only step ≤ 2000
        salvage = [(s, l) for s, l in salvage if s <= 2000]
        steps_losses.extend(salvage)

    # Now append extension log entries (step > 2000)
    if name == "inner128":
        cont_dir = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/K18_res2_inner128")
    else:
        cont_dir = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop") / dir_name
    live_log = cont_dir / "train_log.json"
    if live_log.exists():
        d = json.load(open(live_log))
        for entry in d:
            s = entry["step"]
            if s > 2000:
                steps_losses.append((s, entry["loss"]))

    # Deduplicate + sort
    steps_losses = sorted(set(steps_losses), key=lambda x: x[0])
    return steps_losses


def smooth(vals, window):
    if len(vals) < window: return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid")


fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
cmap = plt.get_cmap("tab10")
color_map = {name: cmap(i) for i, (name, *_) in enumerate(CONFIGS)}

for name, dir_name, task_idx in CONFIGS:
    curve = load_config_loss_curve(name, dir_name, task_idx)
    if not curve:
        print(f"{name}: NO DATA"); continue
    steps = np.asarray([s for s, _ in curve])
    losses = np.asarray([l for _, l in curve])
    epochs = steps / STEPS_PER_EPOCH

    # Left panel: raw
    axes[0].plot(epochs, losses, "-", color=color_map[name], lw=0.8, alpha=0.4)
    # Smoothed
    win = 21
    sm = smooth(losses, win)
    sm_x = epochs[(win-1)//2 : (win-1)//2 + len(sm)]
    axes[0].plot(sm_x, sm, "-", color=color_map[name], lw=2.0,
                 label=f"{name}  (last={losses[-1]:.3f}, n={len(steps)})")

    # Right panel: log-scale smooth only, zoom
    axes[1].plot(sm_x, sm, "-", color=color_map[name], lw=2.0, label=name)

    print(f"{name}: n={len(steps)}  first_step={steps[0]}  last_step={steps[-1]}  "
          f"first_loss={losses[0]:.3f}  last_smoothed_loss={sm[-1]:.3f}")

axes[0].set_ylim(4.5, 7.5)
axes[1].set_ylim(4.5, 6.0)
for ax, title in zip(axes, [
    "Full training loss vs epoch (all configs; raw thin, 21-step-smoothed thick)",
    "Smoothed only — zoom to compare final trends"
]):
    ax.set_xlabel(f"epoch (= step / {STEPS_PER_EPOCH})")
    ax.set_ylabel("training loss")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(2000 / STEPS_PER_EPOCH, ls=":", color="k", alpha=0.5)
    ymax = ax.get_ylim()[1]
    ax.text(2000 / STEPS_PER_EPOCH, ymax * 0.98, " arch→extend", rotation=90,
            fontsize=8, va="top", ha="left", color="k", alpha=0.6)
axes[0].legend(fontsize=8, ncol=1, loc="upper right")
axes[1].legend(fontsize=8, ncol=1, loc="upper right")

fig.suptitle("v22cl res=2 K=18 arch-screen: training loss (7 configs)", fontsize=13)
plt.tight_layout()
out = OUT_DIR / "training_loss_vs_epoch.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {out}")
