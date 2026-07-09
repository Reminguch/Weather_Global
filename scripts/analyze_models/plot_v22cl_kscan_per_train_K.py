"""Per-train_K detailed plots + matched-K summary for closed-loop K-scan res=2.

Mirrors the 2026-05-23-v22/plots/K{K}/4metrics_K40.png style but overlays multiple
deployment modes (open cold_bp / cl cold_bp / cl cold_full / cl warm_full).

Outputs to /home/lm8598/.../results/2026-06-28-v22cl-kscan/plots/K{K}/
  - lead_curve_4modes.png:  4 modes overlaid, paper-weighted MSE imp% vs lead

Also produces:
  - matched_K_summary.png:  imp% at lead = train_K * 6h, all modes vs train_K
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
SIGMA_DS = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(SIGMA_DS[v].values if SIGMA_DS[v].values.ndim == 0
                  else SIGMA_DS[v].values.mean())
         for v in W_VAR if v in SIGMA_DS}

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v22cl-kscan/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CL_DIR     = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-kscan-eval")
CL_WARM_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-28-v22cl-kscan-warm")
OPEN_BASE  = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba")
OPEN_FILL  = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22-open-kfill-eval")


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev.get("per_variable_per_step", {}):
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


def load_eval(p):
    if p is None or not Path(p).exists(): return None
    return json.loads(Path(p).read_text())


def open_loop_path(K):
    candidates = [
        OPEN_BASE / f"K{K}_step20000_cold_bp.json",
        OPEN_BASE / f"K{K}_step6000_cold_bp.json",
        OPEN_BASE / f"K{K}_step5000_cold_bp.json",
        OPEN_BASE / f"K{K}_step3000_cold_bp.json",
        OPEN_BASE / f"K{K}_step2000_cold_bp.json",
        OPEN_FILL / f"v22_open_K{K}_step2000_cold_bp_zero.json",
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def cl_path(K, mode):
    if mode in ("warm_bp", "warm_full"):
        return CL_WARM_DIR / f"v22cl_K{K}_step2000_{mode}_W24_zero.json"
    return CL_DIR / f"v22cl_K{K}_step2000_{mode}_zero.json"


MODES = [
    ("open cold_bp",   "k",  "-",  "o", lambda K: open_loop_path(K)),
    ("cl cold_bp",     "C0", "-",  "s", lambda K: cl_path(K, "cold_bp")),
    ("cl cold_full",   "C2", "-",  "D", lambda K: cl_path(K, "cold_full")),
    ("cl warm_full",   "C3", "--", "^", lambda K: cl_path(K, "warm_full")),
]


def plot_lead_curve_for_K(train_K: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_lead = None
    for label, color, ls, marker, path_fn in MODES:
        ev = load_eval(path_fn(train_K))
        if ev is None: continue
        n_lead = ev["target_steps"]
        lead_h = np.array([6 * (i + 1) for i in range(n_lead)])
        imps = [mse_imp(ev, k) for k in range(n_lead)]
        ax.plot(lead_h, imps, ls, color=color, marker=marker, ms=4, lw=1.8, label=label)
    # Mark train_K boundary (lead = train_K * 6h)
    train_lead = train_K * 6
    ax.axvline(train_lead, color="gray", lw=1.5, ls=":", alpha=0.7,
               label=f"train_K boundary (lead {train_lead}h)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("paper-weighted MSE imp% vs GC baseline")
    ax.set_title(f"v22 train_K={train_K} res=2 step2000 — imp% vs lead")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    if n_lead:
        ax.set_xticks([24, 48, 72, 96, 120, 144, 168, 192, 216, 240])
        ax.set_xlim(0, 6 * n_lead + 6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_matched_K_summary(out_path: Path):
    """For each train_K, imp% at lead = train_K * 6h. Tests 'does model work at trained depth'."""
    fig, ax = plt.subplots(figsize=(12, 6))
    bw = 0.18
    x = np.arange(len(KS))
    offsets = np.arange(len(MODES)) - (len(MODES) - 1) / 2.0
    for i, (label, color, ls, marker, path_fn) in enumerate(MODES):
        vals = []
        for K in KS:
            ev = load_eval(path_fn(K))
            if ev is None:
                vals.append(np.nan); continue
            n_lead = ev["target_steps"]
            target_idx = min(K - 1, n_lead - 1)  # lead = K * 6h, index = K-1
            vals.append(mse_imp(ev, target_idx))
        ax.bar(x + offsets[i] * bw, vals, bw, color=color, edgecolor="black", lw=0.5, label=label)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[j] + offsets[i] * bw, v + (0.3 if v >= 0 else -1.2),
                        f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}\nL{K*6}h" for K in KS])
    ax.set_ylabel("paper-weighted MSE imp% vs GC baseline\n@ lead = train_K × 6h")
    ax.set_title("v22 K-scan res=2 step2000 — MATCHED-K eval (improvement at trained AR depth)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def print_matched_K_table():
    print()
    print("=" * 110)
    print("MATCHED-K table: improvement_pct (paper-weighted MSE) at lead = train_K × 6h")
    print("=" * 110)
    print(f"  {'train_K':>8s}  {'lead':>5s}  " + "  ".join(f"{m[0]:>16s}" for m in MODES))
    for K in KS:
        cells = []
        for label, _color, _ls, _marker, path_fn in MODES:
            ev = load_eval(path_fn(K))
            if ev is None:
                cells.append("n/a")
            else:
                n_lead = ev["target_steps"]
                target_idx = min(K - 1, n_lead - 1)
                imp = mse_imp(ev, target_idx)
                cells.append(f"{imp:+.2f}%")
        print(f"  K={K:<6d}  L{K*6:>3d}h  " + "  ".join(f"{c:>16s}" for c in cells))


def main():
    print_matched_K_table()
    # Per-train_K lead curves
    for K in KS:
        d = OUT_DIR / f"K{K}"
        d.mkdir(parents=True, exist_ok=True)
        plot_lead_curve_for_K(K, d / "lead_curve_4modes.png")
    # Matched-K bar summary
    plot_matched_K_summary(OUT_DIR / "matched_K_summary.png")


if __name__ == "__main__":
    main()
