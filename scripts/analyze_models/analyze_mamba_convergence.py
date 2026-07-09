"""Analyze Mamba SSM parameter convergence across training checkpoints.

For each K, we load every saved ckpt (step 500, 1000, ..., 20000) and compute:
  - Total Mamba param L2 norm         → shows if magnitudes grow / stabilize
  - Δ from previous ckpt (relative)   → shows when training slows to a crawl
  - Per-submodule L2 (A_log, dt_proj, x_proj, in/out_proj, conv1d)
      → shows which SSM component is still moving late in training

Overlays across K to find each K's convergence step.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

CKPT_ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res1_closedloop_fresh")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-02-mamba-convergence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

def ckpt_dir(K):
    return CKPT_ROOT / f"K{K}_closedloop_sg_fresh_dmckpt"

def available_steps(K):
    d = ckpt_dir(K)
    files = list(d.glob("v13_residual_step*.pkl"))
    steps = sorted(int(f.stem.split("step")[1]) for f in files)
    return steps

def is_mamba_module(name):
    return ("mesh_interleaved_temporal" in name or "mamba_block" in name
            or "temporal_residual_head" in name)

def submodule(leaf_path):
    """Return submodule label from full leaf path (module_name + leaf_key)."""
    # leaf_path like "mesh_interleaved_temporal_r0_s0/~_run_sequence/mamba_block_0/in_proj|w"
    if "A_log" in leaf_path: return "A_log"
    if "/D" in leaf_path or leaf_path.endswith("|D"): return "D"
    if "dt_proj" in leaf_path: return "dt_proj"
    if "x_proj" in leaf_path: return "x_proj"
    if "in_proj" in leaf_path: return "in_proj"
    if "out_proj" in leaf_path: return "out_proj"
    if "conv1d" in leaf_path: return "conv1d"
    if "layer_norm" in leaf_path: return "layer_norm"
    if "temporal_residual_head" in leaf_path: return "residual_head"
    return "other_mamba"

def flatten_mamba(params):
    """Return dict[submodule_label] → concatenated flat array."""
    out = {}
    for mod_name, inner in params.items():
        if not is_mamba_module(mod_name): continue
        if not isinstance(inner, dict): continue
        for leaf_key, val in inner.items():
            path = f"{mod_name}|{leaf_key}"
            sub = submodule(path)
            arr = np.asarray(val).astype(np.float32).flatten()
            if sub not in out: out[sub] = []
            out[sub].append(arr)
    return {k: np.concatenate(v) for k, v in out.items()}

def load_flat_mamba(K, step):
    path = ckpt_dir(K) / f"v13_residual_step{step}.pkl"
    with path.open("rb") as f:
        ck = pickle.load(f)
    return flatten_mamba(ck["residual_params"])

def analyze_K(K, steps):
    print(f"K={K}  loading {len(steps)} ckpts...")
    per_step = {}
    reference = None
    for i, step in enumerate(steps):
        f = load_flat_mamba(K, step)
        per_step[step] = f
        if i == 0: reference = f
    # Compute norms + deltas
    submods = sorted(reference.keys())
    total_norm = {}
    delta_from_prev = {}
    delta_from_init = {}
    prev = None
    for step in steps:
        s = per_step[step]
        total = np.sqrt(sum(np.sum(v ** 2) for v in s.values()))
        total_norm[step] = float(total)
        if prev is not None:
            diff = np.sqrt(sum(np.sum((s[k] - prev[k]) ** 2) for k in s))
            delta_from_prev[step] = float(diff)
        # Delta from step-500 (or first available step)
        init = per_step[steps[0]]
        diff_init = np.sqrt(sum(np.sum((s[k] - init[k]) ** 2) for k in s))
        delta_from_init[step] = float(diff_init)
        prev = s
    # Per-submodule delta from init
    per_sub_delta = {sub: [] for sub in submods}
    for step in steps:
        s = per_step[step]
        for sub in submods:
            per_sub_delta[sub].append(
                float(np.sqrt(np.sum((s[sub] - per_step[steps[0]][sub]) ** 2))))
    return total_norm, delta_from_prev, delta_from_init, per_sub_delta

def plot_convergence():
    cmap = cm.get_cmap("viridis")
    norm_c = plt.Normalize(min(KS), max(KS))

    # Figure 1: total norm + delta-from-init across K
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    ax_tot, ax_dinit, ax_dprev = axes

    all_data = {}
    for K in KS:
        steps = available_steps(K)
        if not steps: continue
        tot, d_prev, d_init, per_sub = analyze_K(K, steps)
        all_data[K] = (steps, tot, d_prev, d_init, per_sub)

        color = cmap(norm_c(K))
        s_arr = np.array(steps)
        ax_tot.plot(s_arr, [tot[s] for s in steps], "-", color=color, lw=1.8, label=f"K={K}")
        ax_dinit.plot(s_arr, [d_init[s] for s in steps], "-", color=color, lw=1.8, label=f"K={K}")
        d_prev_vals = [d_prev[s] if s in d_prev else np.nan for s in steps]
        ax_dprev.plot(s_arr, d_prev_vals, "-o", ms=3, color=color, lw=1.4, label=f"K={K}")

    for ax, ttl, yl in [
        (ax_tot,   "Mamba params — total L2 norm",              "‖θ_mamba‖₂"),
        (ax_dinit, "‖θ(step) − θ(step 500)‖ (drift from init)", "L2 delta"),
        (ax_dprev, "‖θ(step) − θ(prev ckpt)‖ (per-ckpt Δ, log)","L2 delta"),
    ]:
        ax.set_xlabel("training step"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8)
    ax_dprev.set_yscale("log")
    fig.suptitle("v22cl res=1 — Mamba SSM parameter convergence per K", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "mamba_convergence_summary.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # Figure 2: per-submodule drift-from-init for K=22 (representative)
    if 22 in all_data:
        steps, tot, d_prev, d_init, per_sub = all_data[22]
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap2 = cm.get_cmap("tab10")
        sorted_sub = sorted(per_sub.keys())
        for i, sub in enumerate(sorted_sub):
            ax.plot(steps, per_sub[sub], "-o", ms=3, color=cmap2(i % 10), lw=1.6, label=sub)
        ax.set_xlabel("training step"); ax.set_ylabel("‖θ_sub(step) − θ_sub(step 500)‖")
        ax.set_title("K=22 — per-submodule drift from init (step 500)")
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        out = OUT_DIR / "mamba_per_submodule_K22.png"
        plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")

    # Text summary — where does per-ckpt Δ drop below N% of peak Δ?
    print(f"\n{'='*90}")
    print("Convergence step (first step where Δ per ckpt < 25% of peak Δ)")
    print(f"{'='*90}")
    for K in KS:
        if K not in all_data: continue
        steps, tot, d_prev, d_init, per_sub = all_data[K]
        dp_arr = np.array([d_prev.get(s, np.nan) for s in steps])
        finite = dp_arr[np.isfinite(dp_arr)]
        if len(finite) < 4: continue
        peak = finite.max()
        thresh = 0.25 * peak
        # Find first step (ignoring first ~3 to skip warmup spike) where Δ < thresh
        for idx, s in enumerate(steps):
            if idx < 3: continue
            if np.isfinite(dp_arr[idx]) and dp_arr[idx] < thresh:
                print(f"  K={K:<2d}   converged at step {s}  (peak Δ={peak:.3g}, now={dp_arr[idx]:.3g})")
                break
        else:
            print(f"  K={K:<2d}   did NOT drop below 25% peak within 20k steps")

if __name__ == "__main__":
    plot_convergence()
