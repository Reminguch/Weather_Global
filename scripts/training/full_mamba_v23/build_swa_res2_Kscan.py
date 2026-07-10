"""Build SWA ckpts for res=2 K-scan: 2 configs (inner128, inner1024) × 10 K values.
Uniform-average residual_params over step {3000, 4000, 5000, 6000, 7000, 8000}.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import jax

ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop")
KS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22]  # K=18 already SWA'd in arch-screen
STEPS = [3000, 4000, 5000, 6000, 7000, 8000]
CONFIGS = [("inner128", "K{K}_res2_inner128"), ("inner1024", "K{K}_res2_inner1024")]

for tag, dir_template in CONFIGS:
    for K in KS:
        cfg_dir = ROOT / dir_template.format(K=K)
        paths = [cfg_dir / f"v13_residual_step{s}.pkl" for s in STEPS]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"K={K} {tag}: MISSING {missing}"); continue
        params_list = []
        residual_state_first = None
        for p in paths:
            with p.open("rb") as f: ck = pickle.load(f)
            params_list.append(ck["residual_params"])
            if residual_state_first is None and "residual_state" in ck:
                residual_state_first = ck["residual_state"]
        n = len(params_list)
        swa = jax.tree.map(lambda *ps: sum(ps) / n, *params_list)
        out_p = cfg_dir / f"v13_residual_SWA_step3k-8k.pkl"
        with out_p.open("wb") as f:
            pickle.dump({"residual_params": swa,
                         "residual_state": residual_state_first,
                         "swa_source_steps": STEPS,
                         "swa_source_ckpts": [str(p) for p in paths]}, f)
        print(f"K={K:2d} {tag:<10s}: SWA of {n} ckpts → {out_p.name}")
print("Done.")
