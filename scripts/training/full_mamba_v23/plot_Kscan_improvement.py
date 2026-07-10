"""Plot improvement% vs K for res=2 K-scan: 2 configs (inner128 vs inner1024).

For each (config, K), read SWA eval JSON, extract lat-weighted improvement% at:
  - step 17 (108h — 18*6h)
  - step 39 (240h — 40*6h)
Aggregate over headline variables (z500, t850, t2m, u850, msl).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/Kscan_SWA_eval")
KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
CONFIGS = ["inner128", "inner1024"]

# Headline variables — 2D and (proxy for) selected pressure-level.
# per_variable_per_step lumps all pressure levels together (rmse averaged across
# vertical). We use 2m_temperature, mean_sea_level_pressure, geopotential (~z),
# temperature (~t), u_component_of_wind, v_component_of_wind.
HEADLINE = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "geopotential", "temperature",
            "u_component_of_wind", "v_component_of_wind", "specific_humidity"]

# Lead index → hours: eval steps are 6h apart; step i (0-indexed) = (i+1)*6h.
# Wait, first entry [0] is the first prediction. If step_size = 6h, then step 0 = 6h.
# So 108h = index 17 (18*6h), 240h = index 39 (40*6h).
LEAD_INDICES = {"108h": 17, "240h": 39}

def load_pct(config: str, K: int, lead_idx: int) -> tuple[float | None, float | None]:
    """Return (mean_rmse_improvement%, mean_mae_improvement%) across headline vars,
    or (None, None) if missing."""
    # K=18 comes from the arch-screen eval (different dir).
    if K == 18:
        arch_name = "inner128" if config == "inner128" else "inner1024"
        p = Path(
            "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/"
            f"arch_eval_SWA/{arch_name}_SWA_step3k-8k_cold_full_zero.json"
        )
    else:
        p = EVAL_DIR / f"K{K}_{config}_SWA_step3k-8k_cold_full_zero.json"
    if not p.exists():
        return None, None
    with p.open() as f: j = json.load(f)
    pv = j["per_variable_per_step"]
    rmse_pcts, mae_pcts = [], []
    for var in HEADLINE:
        if var not in pv: continue
        v = pv[var]
        if lead_idx >= len(v["improvement_pct_rmse"]): continue
        r = v["improvement_pct_rmse"][lead_idx]
        m = v["improvement_pct_mae"][lead_idx]
        if r is None or m is None: continue
        rmse_pcts.append(r); mae_pcts.append(m)
    if not rmse_pcts: return None, None
    return float(np.mean(rmse_pcts)), float(np.mean(mae_pcts))


def main():
    data = {cfg: {"K": [], "rmse_108h": [], "rmse_240h": [], "mae_108h": [], "mae_240h": []}
            for cfg in CONFIGS}
    for cfg in CONFIGS:
        for K in KS:
            r108, m108 = load_pct(cfg, K, LEAD_INDICES["108h"])
            r240, m240 = load_pct(cfg, K, LEAD_INDICES["240h"])
            if r108 is None and r240 is None:
                print(f"  K={K} {cfg}: NO DATA"); continue
            data[cfg]["K"].append(K)
            data[cfg]["rmse_108h"].append(r108 if r108 is not None else np.nan)
            data[cfg]["rmse_240h"].append(r240 if r240 is not None else np.nan)
            data[cfg]["mae_108h"].append(m108 if m108 is not None else np.nan)
            data[cfg]["mae_240h"].append(m240 if m240 is not None else np.nan)
            print(f"  K={K:2d} {cfg:<10s}: 108h RMSE +{r108:+.2f}% MAE +{m108:+.2f}%  "
                  f"240h RMSE +{r240:+.2f}% MAE +{m240:+.2f}%")

    # Plot: 2 panels (108h, 240h). RMSE improvement% vs K, one line per config.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, lead in zip(axes, ["108h", "240h"]):
        for cfg, color, marker in [("inner128", "#d62728", "o"), ("inner1024", "#1f77b4", "s")]:
            d = data[cfg]
            ax.plot(d["K"], d[f"rmse_{lead}"], f"-{marker}", color=color, label=f"{cfg} RMSE", lw=2)
            ax.plot(d["K"], d[f"mae_{lead}"], f"--{marker}", color=color, label=f"{cfg} MAE", lw=1, alpha=0.7)
        ax.axhline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("AR tail K")
        ax.set_ylabel("Improvement over baseline (%)")
        ax.set_title(f"Lead time {lead} — SWA(step 3k-8k), res=2, cold_full/zero-init")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("v22 residual: inner128 vs inner1024 across K (headline-var-averaged, lat-weighted)",
                 fontsize=11)
    fig.tight_layout()
    out = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-08-Kscan-inner128-vs-inner1024/v22_res2_Kscan_inner128_vs_inner1024.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Also dump CSV for records
    import csv
    csv_out = out.with_suffix(".csv")
    with csv_out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["config", "K", "rmse_108h_%", "rmse_240h_%", "mae_108h_%", "mae_240h_%"])
        for cfg in CONFIGS:
            d = data[cfg]
            for i, K in enumerate(d["K"]):
                w.writerow([cfg, K, d["rmse_108h"][i], d["rmse_240h"][i], d["mae_108h"][i], d["mae_240h"][i]])
    print(f"CSV: {csv_out}")


if __name__ == "__main__":
    main()
