"""Plot train+eval loss curves for the gcmamba_v2 K-sweep (res=1 and/or res=2).

Reads slurm out files and extracts:
  train: lines like  `step <s>/38000 loss <l> ...`
  eval:  lines like  `[eval] step <s> total <l>`
  eval_rot:          `[eval_rotating] step <s> total <l>`
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RE_TRAIN = re.compile(r"^step (\d+)/\d+ loss ([0-9.eE+\-]+)")
RE_EVAL  = re.compile(r"^\[eval\] step (\d+) total ([0-9.eE+\-]+)")
RE_EVALR = re.compile(r"^\[eval_rotating\] step (\d+) total ([0-9.eE+\-]+)")


def parse_log(path):
    tr_s, tr_l = [], []
    ev_s, ev_l = [], []
    er_s, er_l = [], []
    if not path.exists():
        return None
    for line in path.open():
        if (m := RE_TRAIN.match(line)):
            tr_s.append(int(m.group(1))); tr_l.append(float(m.group(2)))
        elif (m := RE_EVAL.match(line)):
            ev_s.append(int(m.group(1))); ev_l.append(float(m.group(2)))
        elif (m := RE_EVALR.match(line)):
            er_s.append(int(m.group(1))); er_l.append(float(m.group(2)))
    return dict(train=(np.array(tr_s), np.array(tr_l)),
                eval=(np.array(ev_s), np.array(ev_l)),
                eval_rot=(np.array(er_s), np.array(er_l)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="/home/lm8598/Weather_Global_experiments/logs")
    p.add_argument("--prefix", default="gcmv2_K-sweep_K", help="log prefix; K is inserted after this")
    p.add_argument("--job-id", default="9781103")
    p.add_argument("--ks", nargs="+", type=int, default=[2,4,8,12])
    p.add_argument("--out-dir", default="/home/lm8598/Weather_Global_experiments/plots/gcmamba_v2/K_sweep_loss")
    p.add_argument("--smooth", type=int, default=10, help="train loss MA window in 200-step units")
    p.add_argument("--title-prefix", default="res=1 gc_mamba K-sweep")
    args = p.parse_args()

    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs_dir)
    cmap = plt.get_cmap("viridis")
    n = len(args.ks)
    colors = {k: cmap(i / max(1, n - 1)) for i, k in enumerate(args.ks)}

    data = {}
    for k in args.ks:
        path = logs / f"{args.prefix}{k}_{args.job_id}.out"
        d = parse_log(path)
        if d is None or len(d["train"][0]) == 0:
            print(f"[skip] K={k}: no log at {path}")
            continue
        data[k] = d
        print(f"K={k}: train {len(d['train'][0])} pts (last step {d['train'][0][-1]} loss {d['train'][1][-1]:.4f}), "
              f"eval_rot {len(d['eval_rot'][0])} pts (last {d['eval_rot'][1][-1]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left: train loss (smoothed) ---
    ax = axes[0]
    for k, d in data.items():
        s, l = d["train"]
        if args.smooth > 1 and len(l) > args.smooth:
            ker = np.ones(args.smooth) / args.smooth
            l_s = np.convolve(l, ker, mode="valid")
            s_s = s[args.smooth-1:]
        else:
            s_s, l_s = s, l
        ax.plot(s_s, l_s, color=colors[k], lw=1.4, label=f"K={k}")
    ax.set_xlabel("training step (cumulative, resumed from 18000)")
    ax.set_ylabel(f"train loss (MA={args.smooth}×200)")
    ax.set_title("Train loss")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    # --- Right: eval losses ---
    ax = axes[1]
    for k, d in data.items():
        s_e, l_e = d["eval"]
        s_r, l_r = d["eval_rot"]
        if len(s_e):
            ax.plot(s_e, l_e, "-o", color=colors[k], lw=1.6, markersize=4,
                    label=f"K={k} eval-fixed")
        if len(s_r):
            ax.plot(s_r, l_r, "--s", color=colors[k], lw=1.2, markersize=4, alpha=0.65,
                    label=f"K={k} eval-rot" if k == args.ks[0] else None)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval loss (total)")
    ax.set_title("Eval loss (fixed=solid, rotating=dashed)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"{args.title_prefix} — train+eval loss vs step (resumed from K=1 best @ step 18000)",
                 fontsize=12)
    plt.tight_layout()
    out_path = OUT / "K_sweep_loss.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")

    # Print summary table
    print()
    print(f"{'K':>3} {'last_step':>10} {'train_last':>12} {'train_best':>12} "
          f"{'eval_last':>12} {'eval_best':>12} {'evalrot_last':>14} {'evalrot_best':>14}")
    for k in args.ks:
        if k not in data: continue
        d = data[k]
        tr_l = d["train"][1]; ev_l = d["eval"][1]; er_l = d["eval_rot"][1]
        last_step = int(d["train"][0][-1]) if len(tr_l) else 0
        print(f"{k:>3} {last_step:>10} "
              f"{tr_l[-1]:>12.4f} {tr_l.min():>12.4f} "
              f"{ev_l[-1] if len(ev_l) else float('nan'):>12.4f} {ev_l.min() if len(ev_l) else float('nan'):>12.4f} "
              f"{er_l[-1] if len(er_l) else float('nan'):>14.4f} {er_l.min() if len(er_l) else float('nan'):>14.4f}")


if __name__ == "__main__":
    main()
