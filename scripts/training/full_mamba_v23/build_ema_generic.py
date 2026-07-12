#!/usr/bin/env python3
"""Generic EMA builder: exponential moving average over ordered step ckpts.
EMA_t = beta*EMA_{t-1} + (1-beta)*param_t, iterating steps ascending.
Same interface as build_swa_generic.py. EMA is a post-hoc average of ckpts
(no training-time state needed).

Usage:
  build_ema_generic.py --run-glob '<base>/K{K}_*/*' --ks 4,8,... \
      --steps 1000,2000,...,20000 --beta 0.95 [--out-name v13_residual_EMA_b95.pkl]
"""
import argparse, glob, pickle
from pathlib import Path
import jax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-glob", required=True)
    p.add_argument("--ks", default="4,8,12,16,20,22")
    p.add_argument("--steps", default=",".join(str(s) for s in range(1000, 20001, 1000)))
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--out-name", default=None)
    cfg = p.parse_args()
    ks = [int(x) for x in cfg.ks.split(",")]
    steps = [int(x) for x in cfg.steps.split(",")]
    out_name = cfg.out_name or f"v13_residual_EMA_b{int(cfg.beta*100)}.pkl"

    for K in ks:
        dirs = sorted(glob.glob(cfg.run_glob.replace("{K}", str(K))))
        if not dirs:
            print(f"K={K}: NO dir"); continue
        d = Path(dirs[0])
        avail = [s for s in steps if (d / f"v13_residual_step{s}.pkl").exists()]
        if len(avail) < 2:
            print(f"K={K}: only {len(avail)} ckpts, skip"); continue
        ema, rs = None, None
        for s in avail:
            ck = pickle.load(open(d / f"v13_residual_step{s}.pkl", "rb"))
            pp = ck["residual_params"]
            ema = pp if ema is None else jax.tree.map(lambda e, x: cfg.beta * e + (1 - cfg.beta) * x, ema, pp)
            if rs is None and "residual_state" in ck:
                rs = ck["residual_state"]
        out = d / out_name
        with out.open("wb") as f:
            pickle.dump({"residual_params": ema, "residual_state": rs,
                         "ema_beta": cfg.beta, "ema_source_steps": avail}, f)
        print(f"K={K:2d}: EMA(beta={cfg.beta}) of {len(avail)} ckpts -> {out.name}")
    print("Done.")


if __name__ == "__main__":
    main()
