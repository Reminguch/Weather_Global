"""Analyze Mamba SSM parameter drift for v22 paper ckpts (2026-05-23-v22).
v22 paper: cascade-trained K=1 for 20k steps, then each K resumes for 3k steps
(step 23500..26000, saved every 500).

Compares late-training drift (6 ckpts × 500 step interval) to v22cl_res1
(fresh, 26 ckpts × step 500..20000).
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

V22_ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-02-mamba-convergence-v22paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def ckpt_dir(K):
    if K == 1:
        return V22_ROOT / "K1_from_v15v2_20k"
    return V22_ROOT / f"K{K}_from_v22K1_23k"


def available_steps(K):
    d = ckpt_dir(K)
    files = list(d.glob("v13_residual_step*.pkl"))
    return sorted(int(f.stem.split("step")[1]) for f in files)


def is_mamba(name):
    return ("mesh_interleaved_temporal" in name or "mamba_block" in name
            or "temporal_residual_head" in name)


def submodule(path):
    if "A_log" in path: return "A_log"
    if path.endswith("|D"): return "D"
    if "dt_proj" in path: return "dt_proj"
    if "x_proj" in path: return "x_proj"
    if "in_proj" in path: return "in_proj"
    if "out_proj" in path: return "out_proj"
    if "conv1d" in path: return "conv1d"
    if "layer_norm" in path: return "layer_norm"
    if "temporal_residual_head" in path: return "residual_head"
    return "other"


def flatten_by_sub(params):
    out = {}
    for m, inner in params.items():
        if not is_mamba(m): continue
        if not isinstance(inner, dict): continue
        for k, v in inner.items():
            path = f"{m}|{k}"
            sub = submodule(path)
            arr = np.asarray(v).astype(np.float32).flatten()
            out.setdefault(sub, []).append(arr)
    return {k: np.concatenate(v) for k, v in out.items()}


def flatten_all(params):
    arrs = []
    for m, inner in params.items():
        if not is_mamba(m): continue
        if isinstance(inner, dict):
            for k, v in inner.items():
                arrs.append(np.asarray(v).astype(np.float32).flatten())
    return np.concatenate(arrs)


def load(K, step):
    p = ckpt_dir(K) / f"v13_residual_step{step}.pkl"
    with p.open("rb") as f:
        return pickle.load(f)["residual_params"]


def analyze():
    cmap = cm.get_cmap("viridis")
    norm_c = plt.Normalize(min(KS), max(KS))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    ax_tot, ax_dinit, ax_dprev = axes

    summary = {}
    for K in KS:
        steps = available_steps(K)
        if len(steps) < 2: continue
        flats = {}
        for s in steps:
            flats[s] = flatten_all(load(K, s))
        init = flats[steps[0]]
        norms = [np.linalg.norm(flats[s]) for s in steps]
        dinit = [np.linalg.norm(flats[s] - init) for s in steps]
        dprev = [np.nan] + [np.linalg.norm(flats[steps[i]] - flats[steps[i-1]])
                            for i in range(1, len(steps))]
        summary[K] = (steps, norms, dinit, dprev)

        color = cmap(norm_c(K))
        ax_tot.plot(steps, norms, "-o", ms=4, color=color, lw=1.6, label=f"K={K}")
        ax_dinit.plot(steps, dinit, "-o", ms=4, color=color, lw=1.6, label=f"K={K}")
        ax_dprev.plot(steps, dprev, "-o", ms=4, color=color, lw=1.6, label=f"K={K}")

    for ax, ttl, yl in [
        (ax_tot,   "Mamba params — total L2 norm",         "‖θ_mamba‖₂"),
        (ax_dinit, "‖θ(step) − θ(first ckpt)‖",           "L2 delta"),
        (ax_dprev, "‖θ(step) − θ(prev ckpt)‖ (500-step Δ)","L2 delta"),
    ]:
        ax.set_xlabel("training step"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig.suptitle("v22 PAPER ckpts — Mamba param convergence (steps 23500-26000, K-cascade)", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "mamba_convergence_v22paper.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # Text summary
    print(f"\n{'='*95}")
    print("v22 paper — per-ckpt Δ (500 step interval) & total norm")
    print(f"{'='*95}")
    print(f"  {'K':>3s}  {'steps':>18s}  {'‖θ(first)‖':>10s}  {'‖θ(last)‖':>10s}  {'‖Δ(last-first)‖':>15s}  {'avg Δ/500step':>13s}")
    for K in KS:
        if K not in summary: continue
        steps, norms, dinit, dprev = summary[K]
        finite_dprev = [x for x in dprev if np.isfinite(x)]
        avg = np.mean(finite_dprev) if finite_dprev else np.nan
        step_range = f"{steps[0]}..{steps[-1]}"
        print(f"  K={K:<2d}  {step_range:>18s}  {norms[0]:>10.3f}  {norms[-1]:>10.3f}  {dinit[-1]:>15.3f}  {avg:>13.3f}")

    # Per-submodule Δ for K=22 (last minus first)
    print(f"\n{'='*90}")
    print("K=22 per-submodule drift (step 23500 → 26000, 2500 steps)")
    print(f"{'='*90}")
    if 22 in summary:
        steps = available_steps(22)
        s0 = flatten_by_sub(load(22, steps[0]))
        s1 = flatten_by_sub(load(22, steps[-1]))
        print(f"  {'submodule':>15s}  {'‖init‖':>10s}  {'Δ 2500 steps':>13s}  {'rel %':>8s}")
        for sub in sorted(s0.keys()):
            i = np.linalg.norm(s0[sub])
            d = np.linalg.norm(s1[sub] - s0[sub])
            print(f"  {sub:>15s}  {i:>10.3f}  {d:>13.3f}  {d/i*100:>7.2f}%")

    return summary


if __name__ == "__main__":
    analyze()
