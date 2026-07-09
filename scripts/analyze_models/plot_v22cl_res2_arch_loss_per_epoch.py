"""Per-epoch mean training loss for res=2 K=18 arch-screen.

One line per config. x = epoch, y = mean of that epoch's step losses.
No raw traces — clean comparison across configs.
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
BASE_JOBID = "10719736"


def parse_stdout_losses(stdout_path: Path):
    if not stdout_path.exists(): return []
    out = []
    for m in re.finditer(r"step (\d+)/\d+ loss ([\d.]+)", stdout_path.read_text()):
        out.append((int(m.group(1)), float(m.group(2))))
    return out


def load_all_losses(name, dir_name, task_idx):
    """Return sorted list of (step, loss) covering step 1..latest across sources."""
    all_pairs = []
    bak = BAK / f"{name}_step1-2000_train_log.json"
    if bak.exists():
        for e in json.load(open(bak)): all_pairs.append((e["step"], e["loss"]))
    else:
        stdout = LOG_DIR / f"v22cl_res2_arch_t{task_idx}_{BASE_JOBID}.out"
        for s, l in parse_stdout_losses(stdout):
            if s <= 2000: all_pairs.append((s, l))

    if name == "inner128":
        cont_dir = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/K18_res2_inner128")
    else:
        cont_dir = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop") / dir_name
    live = cont_dir / "train_log.json"
    if live.exists():
        for e in json.load(open(live)):
            if e["step"] > 2000: all_pairs.append((e["step"], e["loss"]))
    return sorted(set(all_pairs), key=lambda x: x[0])


def epoch_avg(pairs):
    """Return list of (epoch_idx, mean_loss_this_epoch)."""
    if not pairs: return []
    epochs: dict[int, list[float]] = {}
    for step, loss in pairs:
        ep = (step - 1) // STEPS_PER_EPOCH + 1     # step 1..424 → epoch 1
        epochs.setdefault(ep, []).append(loss)
    return sorted([(ep, float(np.mean(v))) for ep, v in epochs.items()])


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap = plt.get_cmap("tab10")

for i, (name, dir_name, task_idx) in enumerate(CONFIGS):
    pairs = load_all_losses(name, dir_name, task_idx)
    if not pairs: continue
    ep_avg = epoch_avg(pairs)
    if not ep_avg: continue
    xs = [e for e, _ in ep_avg]
    ys = [l for _, l in ep_avg]
    ax.plot(xs, ys, "-o", ms=6, lw=2, color=cmap(i),
            label=f"{name}  (last epoch loss={ys[-1]:.3f})")
    print(f"{name:<10s} epochs 1..{max(xs)}  final_ep_avg={ys[-1]:.4f}  min={min(ys):.4f} at epoch {xs[ys.index(min(ys))]}")

ax.set_xlabel("epoch (= 424 training steps)")
ax.set_ylabel("mean training loss over epoch")
ax.set_title("v22cl res=2 K=18 arch-screen: per-epoch mean training loss (7 configs)",
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="best")
plt.tight_layout()
out = OUT_DIR / "loss_per_epoch.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {out}")
