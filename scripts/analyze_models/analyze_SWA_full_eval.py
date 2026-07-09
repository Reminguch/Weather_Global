"""Full per-K SWA analysis.
For each K's SWA ckpt (all-K EMA eval JSONs), compute:
  1. Paper-weighted MSE improvement over all leads (curve per K)
  2. Count of channels with positive improvement at each lead (heatmap)
  3. Per-variable improvement% at 240h (multi-var K-scan)
  4. Comparison table: SWA vs step 6k peak vs step 20k final

Same output structure as 2026-07-01-v22cl-r1-step14k-eval + adds channel counts.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr

ROOT = Path("/home/lm8598/Weather_Global_experiments")
D_SWA = ROOT / "results/2026-07-02-v22cl-r1-allK-EMA-eval"
D_STEP6K = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep"
D_STEP14K = ROOT / "results/2026-07-01-v22cl-r1-step14k-eval"
D_STEP20K = ROOT / "results/2026-6-29-v22cl-r1-step20k-eval"
OUT_BASE = D_SWA / "plots"
OUT_BASE.mkdir(parents=True, exist_ok=True)
PY_BIN = ROOT / ".conda/envs/graphcast311/bin/python"

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
MODES = ["cold_bp", "cold_full"]

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
SIG_DS = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
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


def channel_improve_count(ev, k_idx, exclude_precip=True):
    """Number of channels with positive improvement at this lead.
    Uses ONLY per_channel_per_step (83 channels: 5 surface + 6 upper × 13 levels).
    Paper convention: exclude total_precipitation_6hr (idiosyncratic, often noisy).
    → 82 or 81 depending on exclusions (paper plot shows 81)."""
    total, positive = 0, 0
    for c, cd in ev.get("per_channel_per_step", {}).items():
        if exclude_precip and "precipitation" in c: continue
        arr = cd.get("improvement_pct_rmse") or cd.get("improvement_pct")
        if arr and k_idx < len(arr):
            total += 1
            if arr[k_idx] > 0: positive += 1
    return positive, total


def swa_path(K, mode):
    return D_SWA / f"v22cl_r1_K{K}_EMA_{mode}_zero.json"


def load(K, mode):
    p = swa_path(K, mode)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    # 1. Overlay + K-scan plots (regen 2026-05-23-v22 style)
    plot_script = ROOT / "scripts/training/full_mamba_v23/plot_v23_kscan.py"
    for mode in MODES:
        # Overlay
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        cmap = cm.get_cmap("viridis")
        norm = plt.Normalize(min(KS), max(KS))
        Ks_valid = []; evs = {}
        for K in KS:
            ev = load(K, mode)
            if ev is None: continue
            Ks_valid.append(K); evs[K] = ev
        if not Ks_valid: continue
        n_lead = evs[Ks_valid[0]]["target_steps"]
        lead_h = [6*(i+1) for i in range(n_lead)]
        SELECTED = [6, 24, 48, 72, 120, 168, 240]

        ax = axes[0]
        for K in Ks_valid:
            imps = [mse_imp(evs[K], k) for k in range(n_lead)]
            ax.plot(lead_h, imps, "-o", ms=4, lw=1.5, color=cmap(norm(K)), label=f"K={K}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lead time (h)"); ax.set_ylabel("paper-weighted MSE improvement (%)")
        ax.set_title(f"v22cl r1 SWA {mode} — improvement vs lead, all K")
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8, loc="upper left")

        ax = axes[1]
        cmap2 = cm.get_cmap("tab10")
        for i, lh in enumerate(SELECTED):
            k = lh // 6 - 1
            imps = [mse_imp(evs[K], k) for K in Ks_valid]
            ax.plot(Ks_valid, imps, "-o", ms=5, lw=1.8, color=cmap2(i), label=f"lead {lh}h")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("K"); ax.set_ylabel("paper-weighted MSE improvement (%)")
        ax.set_title(f"v22cl r1 SWA {mode} K-scan by lead")
        ax.set_xticks(Ks_valid); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"v22cl res=1 SWA (all K) — {mode.upper()}", fontsize=12)
        plt.tight_layout()
        out = OUT_BASE / f"{mode}_vs_baseline.png"
        plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")

        # Per-K standard v22-style plots
        for K in Ks_valid:
            jp = swa_path(K, mode)
            out_dir = OUT_BASE / mode / f"K{K}"; out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [str(PY_BIN), str(plot_script), str(jp), str(out_dir),
                   f"v22cl_r1_SWA_{mode}", "4"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  K={K} {mode}: STDERR {res.stderr[-300:]}")

    # 2. Channel-improve-count heatmap per mode
    for mode in MODES:
        counts = np.zeros((len(KS), 40))  # rows K, cols lead
        totals = np.zeros((len(KS), 40))
        for i, K in enumerate(KS):
            ev = load(K, mode)
            if ev is None: continue
            for k in range(min(40, ev["target_steps"])):
                p, t = channel_improve_count(ev, k)
                counts[i, k] = p; totals[i, k] = t

        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        # Absolute count
        im0 = axes[0].imshow(counts, aspect="auto", cmap="RdYlGn", origin="lower",
                             vmin=0, vmax=totals.max())
        axes[0].set_yticks(range(len(KS))); axes[0].set_yticklabels([f"K={K}" for K in KS])
        axes[0].set_xticks([3, 7, 11, 19, 27, 39])
        axes[0].set_xticklabels(["24h", "48h", "72h", "120h", "168h", "240h"])
        axes[0].set_xlabel("lead time"); axes[0].set_title(f"# channels with positive improvement — {mode}")
        plt.colorbar(im0, ax=axes[0], shrink=0.8, label="# improving channels")
        for i in range(len(KS)):
            for j in [3, 11, 27, 39]:
                axes[0].text(j, i, f"{int(counts[i,j])}", ha="center", va="center",
                             color="black", fontsize=8)

        # Fraction
        frac = np.divide(counts, np.maximum(totals, 1))
        im1 = axes[1].imshow(frac, aspect="auto", cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
        axes[1].set_yticks(range(len(KS))); axes[1].set_yticklabels([f"K={K}" for K in KS])
        axes[1].set_xticks([3, 7, 11, 19, 27, 39])
        axes[1].set_xticklabels(["24h", "48h", "72h", "120h", "168h", "240h"])
        axes[1].set_xlabel("lead time"); axes[1].set_title(f"fraction of channels improving — {mode}")
        plt.colorbar(im1, ax=axes[1], shrink=0.8, label="fraction")
        for i in range(len(KS)):
            for j in [3, 11, 27, 39]:
                axes[1].text(j, i, f"{frac[i,j]:.2f}", ha="center", va="center",
                             color="black", fontsize=8)
        fig.suptitle(f"v22cl r1 SWA — channels improving vs GC baseline ({mode})", fontsize=12)
        plt.tight_layout()
        out = OUT_BASE / f"channels_improve_{mode}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")

    # 3. Comparison table: SWA vs step 6k / 20k for cold_full
    print(f"\n{'='*100}")
    print("Full-eval comparison: SWA vs step 6000 vs step 20000  (cold_full paper-weighted MSE)")
    print(f"{'='*100}")
    leads_show = [3, 11, 19, 27, 39]; leads_lbl = ["24h", "72h", "120h", "168h", "240h"]
    for K in KS:
        ev_swa = load(K, "cold_full")
        ev_6k = json.loads((D_STEP6K / f"v22cl_r1_K{K}_step6000_cold_full_zero.json").read_text()) if (D_STEP6K / f"v22cl_r1_K{K}_step6000_cold_full_zero.json").exists() else None
        ev_20k = json.loads((D_STEP20K / f"v22cl_r1_K{K}_step20000_cold_full_zero.json").read_text()) if (D_STEP20K / f"v22cl_r1_K{K}_step20000_cold_full_zero.json").exists() else None
        if ev_swa is None: continue
        print(f"\n  K={K}")
        for src, ev in [("6k", ev_6k), ("20k", ev_20k), ("SWA", ev_swa)]:
            if ev is None:
                print(f"    {src:>4s}: MISSING"); continue
            imps = [mse_imp(ev, k) for k in range(ev["target_steps"])]
            n_lead = ev["target_steps"]
            # count total channels improving @240h
            p240, t240 = channel_improve_count(ev, 39)
            mean_val = np.mean([x for x in imps[3:40] if x==x])
            row = f"    {src:>4s}: "
            for k, lbl in zip(leads_show, leads_lbl):
                row += f"{lbl}: {imps[k]:+6.2f}%  "
            row += f"| mean {mean_val:+6.2f}%  | @240h {p240}/{t240} channels + "
            print(row)


if __name__ == "__main__":
    main()
