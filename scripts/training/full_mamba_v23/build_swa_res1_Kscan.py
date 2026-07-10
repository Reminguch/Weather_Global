"""Build SWA ckpts for res=1 H=256 K-scan.
Window: step 6k-16k every 2k = 6 ckpts uniformly averaged.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import jax

BASE = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed")
KS = [4, 8, 12, 16, 20, 22]
STEPS = [6000, 8000, 10000, 12000, 14000, 16000]

for K in KS:
    ckpt_dir = BASE / f"K{K}_H256_fresh_20k" / f"v22closed_H256_K{K}_fresh_20k"
    paths = [ckpt_dir / f"v13_residual_step{s}.pkl" for s in STEPS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"K={K}: MISSING {missing}"); continue
    params_list, rs_first = [], None
    for p in paths:
        with p.open("rb") as f: ck = pickle.load(f)
        params_list.append(ck["residual_params"])
        if rs_first is None and "residual_state" in ck:
            rs_first = ck["residual_state"]
    n = len(params_list)
    swa = jax.tree.map(lambda *ps: sum(ps) / n, *params_list)
    out = ckpt_dir / "v13_residual_SWA_step6k-16k.pkl"
    with out.open("wb") as f:
        pickle.dump({"residual_params": swa, "residual_state": rs_first,
                     "swa_source_steps": STEPS,
                     "swa_source_ckpts": [str(p) for p in paths]}, f)
    print(f"K={K:2d}: SWA of {n} ckpts → {out.name}")
print("Done.")
