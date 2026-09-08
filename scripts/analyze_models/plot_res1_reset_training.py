"""Compare res1 carry/reset SWAs with approximate GC loss from channel RMSEs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from analyze_v22_final_state_evals_20260815 import graphcast_weighted_normalized_mse_improvement

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "artifacts/checkpoints/v22_final"
NAME = "res1_reset_training_gc_loss_approx"
IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
POLICIES = (
    ("Mamba: carry train / carry eval", "stateful", False, "#28658a", "-"),
    ("Mamba: carry train / reset eval", "stateful", True, "#e76f51", "--"),
    ("Mamba: reset train / reset eval", "reset_every_anchor", True, "#2aaf79", "-."),
)


def main():
    diff_scales = xr.load_dataset(ROOT / "data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    rows, summaries = [], []
    reference = None
    max_baseline_difference = 0.0
    for row, bcg in enumerate((1, 4)):
        for col, (start, end) in enumerate(((2000, 8000), (4000, 10000), (6000, 12000))):
            ax = axes[row, col]
            tag = f"swa_step{start:05d}-{end:05d}"
            for label, train, reset, color, style in POLICIES:
                root_name = ("res1_dm_7yr_k20_di_bcg_20k_20260813" if train == "stateful"
                             else "res1_dm_7yr_k20_di_bcg_reset_every_anchor_20k_20260815")
                run = CHECKPOINTS / root_name / f"di16_bcg{bcg}_closed_sg_{train}_20k"
                mode = "cold_full_reset_every_step_zero" if reset else "cold_full_zero"
                path = run / "eval" / mode / f"{tag}.json"
                data = json.loads(path.read_text())
                config = json.loads((run / "run_config.json").read_text())
                if train == "reset_every_anchor":
                    assert config["sequence"]["temporal_state_policy"] == train
                assert data["rs_reset_every_step"] == reset
                signature = {k: data[k] for k in (
                    "resolution", "chosen_idx", "target_steps", "evaluated_samples",
                    "eval_mode", "eval_feedback", "residual_state_init", "baseline_branch")}
                if reference is None:
                    reference = signature
                assert signature == reference, f"Unmatched protocol: {path}"
                assert data["resolution"] == 1.0 and data["evaluated_samples"] == 32
                metrics = data["per_variable_per_step"]
                baseline = np.array([metrics[v]["rmse_baseline"] for v in sorted(metrics)])
                full = np.array([metrics[v]["rmse_full"] for v in sorted(metrics)])
                assert baseline.shape == full.shape and baseline.shape[1] == 40
                assert np.all(baseline > 0) and np.all(np.isfinite(full))
                if label == POLICIES[0][0]:
                    panel_baseline = baseline
                else:
                    difference = float(np.max(np.abs(baseline / panel_baseline - 1)))
                    max_baseline_difference = max(max_baseline_difference, difference)
                    assert difference < .01, f"Baseline RMSE differs by over 1%: {path}"
                curve = graphcast_weighted_normalized_mse_improvement(data, diff_scales)
                days = np.arange(1, 41) / 4
                ax.plot(days, curve, color=color, ls=style, lw=2.2, label=label)
                summaries.append(dict(bcg=bcg, swa=tag, policy=label,
                                      mean_lead_approx_gc_loss_reduction_pct=float(curve.mean()),
                                      day10_approx_gc_loss_reduction_pct=float(curve[-1])))
                rows.extend(dict(bcg=bcg, swa=tag, policy=label, lead_days=float(day),
                                 approximate_gc_loss_reduction_pct=float(value),
                                 source=str(path.relative_to(ROOT)))
                            for day, value in zip(days, curve))
            ax.axhline(0, color="0.4", lw=1)
            ax.set_title(f"di16 / BCG{bcg} · SWA {start//1000}k–{end//1000}k")
            ax.grid(alpha=.22)
            ax.set_xlim(.25, 10)
            if row == 1:
                ax.set_xlabel("Forecast lead time (days)")
            if col == 0:
                ax.set_ylabel("Approx. GraphCast loss reduction (%)")
    fig.suptitle("Res1: internal-state reset · approximate GraphCast loss", fontsize=17, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.text(.5, .025, "Relative to GraphCast-small baseline · 32 matched cold samples · 40 six-hour leads\n"
             "Reset training = reset_every_anchor; reset evaluation = every step. Separate training runs.\n"
             f"Each curve uses its saved baseline (maximum difference {max_baseline_difference:.2%}). "
             "Channel-MSE reconstruction; exact pole weighting unavailable.",
             ha="center", fontsize=10, color="0.3")
    fig.tight_layout(rect=(0, .11, 1, .89))
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(IMAGE_DIR / f"{NAME}.{suffix}", dpi=180)
    plt.close(fig)
    for filename, records in (("curves.csv", rows), ("summary.csv", summaries)):
        with (DATA_DIR / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"Validated {len(summaries)} evaluations; wrote {IMAGE_DIR / (NAME + '.png')}")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
