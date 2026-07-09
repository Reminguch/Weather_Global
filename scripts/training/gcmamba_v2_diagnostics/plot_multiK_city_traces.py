"""Multi-K city trace overlay: truth (ERA5) vs baseline vs gc_mamba_K∈{1,2,4,8,12}.
For each big city, plot 2m_temperature + total_precipitation_6hr trajectories
over the 40-step closed-loop rollout.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    p.add_argument("--cities", nargs="+",
                   default=["NYC","LA","Paris","London","Tokyo","Beijing","Mumbai","Sydney"])
    p.add_argument("--prefix", default="eval_res1_2way",
                   help="JSON filename prefix; K=1 looks for <prefix>.json, K>1 looks for <prefix>_K{K}_step*.json")
    p.add_argument("--baseline-name", default="DeepMind small (res=1)")
    p.add_argument("--title-suffix", default="res=1")
    return p.parse_args()


def find_json(in_dir, k, prefix="eval_res1_2way"):
    if k == 1:
        p = Path(in_dir) / f"{prefix}.json"
        if p.exists():
            return p
        # Fallback: explicit K=1 named file (exclude W-sweep)
        gs = sorted(glob.glob(str(Path(in_dir) / f"{prefix}_K1_step*.json")))
        gs = [g for g in gs if "_W" not in Path(g).stem.split("_K1_step")[1]]
        return Path(gs[-1]) if gs else None
    gs = sorted(glob.glob(str(Path(in_dir) / f"{prefix}_K{k}_step*.json")))
    # Exclude warmup-sweep files (which have _W{N} in name)
    gs = [g for g in gs if "_W" not in Path(g).stem.split(f"_K{k}_step")[1]]
    return Path(gs[-1]) if gs else None


def main():
    cfg = parse_args()
    OUT = Path(cfg.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    k_color = {k: cmap(i / max(1, len(cfg.ks) - 1)) for i, k in enumerate(cfg.ks)}

    data = {}
    for k in cfg.ks:
        p = find_json(cfg.in_dir, k, prefix=cfg.prefix)
        if p is None:
            print(f"[skip] K={k}: no JSON in {cfg.in_dir}"); continue
        data[k] = json.load(p.open())
        print(f"K={k}: loaded {p.name}")
    if not data:
        print("no data"); return

    # Use first K's run for n_anchors/K/lead_days/baseline trace
    ref_k = sorted(data.keys())[0]
    ref = data[ref_k]
    n_anch = ref["n_anchors"]; K_t = ref["target_steps"]
    lead_days = (np.arange(1, K_t + 1) * 6) / 24.0

    var_temp = "2m_temperature"
    var_prec = "total_precipitation_6hr"

    cities = [c for c in cfg.cities if c in ref["cities"]]
    n_c = len(cities)
    fig, axes = plt.subplots(n_c, 2, figsize=(13, 2.4 * n_c), squeeze=False)

    for ci, city in enumerate(cities):
        for vi, var in enumerate([var_temp, var_prec]):
            ax = axes[ci, vi]
            scale = 1000.0 if var == var_prec else 1.0
            ylabel = var + (" (mm/6h)" if var == var_prec else " (K)")
            # truth — from ref (same anchors used for all K)
            truth = np.asarray(ref["city_traces"]["truth"][city][var]).reshape(n_anch, K_t).mean(0) * scale
            ax.plot(lead_days, truth, "k-", lw=2.2, label="ERA5 truth" if (ci==0 and vi==0) else None)
            # baseline — from ref
            base = np.asarray(ref["city_traces"]["baseline"][city][var]).reshape(n_anch, K_t).mean(0) * scale
            ax.plot(lead_days, base, "--", color="C0", lw=1.4,
                    label=f"baseline ({cfg.baseline_name})" if (ci==0 and vi==0) else None)
            # gc_mamba per K
            for k in cfg.ks:
                if k not in data: continue
                d = data[k]
                gm = np.asarray(d["city_traces"]["gc_mamba"][city][var]).reshape(d["n_anchors"], d["target_steps"]).mean(0) * scale
                ax.plot(lead_days, gm, "-", color=k_color[k], lw=1.5,
                        label=f"gc_mamba K={k}" if (ci==0 and vi==0) else None)

            # per-K imp annotation (lead-mean)
            imps = []
            for k in cfg.ks:
                if k in data:
                    pv = data[k]["per_variable_per_lead"][var]
                    imps.append((k, float(np.mean(pv["improvement_gc_mamba_pct"]))))
            ann = "  ".join([f"K={k}: {i:+.1f}%" for k, i in imps])
            ax.set_title(f"{city} — {var}\nglobal lat-w RMSE imp: {ann}", fontsize=8)
            ax.set_xlabel("lead (days)"); ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.3)
            if ci == 0 and vi == 0:
                ax.legend(fontsize=8, loc="best", ncol=2)

    fig.suptitle(
        f"Big-city traces ({cfg.title_suffix}) — truth (ERA5) vs baseline ({cfg.baseline_name}) "
        f"vs gc_mamba K∈{{{','.join(str(k) for k in cfg.ks)}}}\n"
        f"({n_anch} anchors, val_year=2022, 24-step truth warmup + 40-step closed-loop rollout)",
        fontsize=11)
    plt.tight_layout()
    p_out = OUT / "city_traces_multiK_temp_precip.png"
    plt.savefig(p_out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p_out}")

    # ---- improvement vs lead, multi-K, surface vars (incl temp+precip) ----
    surface_vars = ["2m_temperature","total_precipitation_6hr","mean_sea_level_pressure",
                    "10m_u_component_of_wind","10m_v_component_of_wind"]
    fig, axes = plt.subplots(1, len(surface_vars), figsize=(4*len(surface_vars), 4))
    for vi, var in enumerate(surface_vars):
        ax = axes[vi]
        for k in cfg.ks:
            if k not in data: continue
            pv = data[k]["per_variable_per_lead"].get(var)
            if pv is None: continue
            ax.plot(lead_days, pv["improvement_gc_mamba_pct"], "-o", lw=1.2, markersize=3,
                    color=k_color[k], label=f"K={k}")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(var.replace("_component_of_wind", "_wind"), fontsize=10)
        ax.set_xlabel("lead (days)")
        if vi == 0:
            ax.set_ylabel("global lat-w RMSE improvement % (vs baseline)")
            ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"Per-K improvement vs lead (global lat-w RMSE) — {cfg.title_suffix}", fontsize=11)
    plt.tight_layout()
    p2 = OUT / "improvement_vs_lead_multiK.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")


if __name__ == "__main__":
    main()
