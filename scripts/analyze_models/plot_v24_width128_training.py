"""Plot the completed fresh-width128 run and its available exact evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / (
    "artifacts/checkpoints/v24_Ilya/res1_width128_mamba1_20260909/runs/"
    "w128_fresh_mamba1_di16_bcg1_rank32_seed22"
)
INK, BLUE, ORANGE = "#213547", "#207c9d", "#cf713d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    image_dir = ROOT / "plots/analyze_models/images/resolution_eval" / output.stem
    data_dir = ROOT / "plots/analyze_models/data/resolution_eval" / output.stem
    for directory in (output.parent, image_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sources = []

    def read(path, *, jsonl=False):
        raw = path.read_bytes()
        sources.append({"path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        return ([json.loads(line) for line in raw.splitlines() if line]
                if jsonl else json.loads(raw))

    config = read(RUN / "run_config.json")
    records = read(RUN / "train_metrics.jsonl", jsonl=True)
    steps = np.array([r["step"] for r in records])
    loss = np.array([r["loss"] for r in records])
    lr = np.array([r["learning_rate"] for r in records])
    grad = np.array([r["gradient_norm"] for r in records])
    assert np.array_equal(steps, np.arange(1, 10001))
    assert np.isfinite([loss, lr, grad]).all()
    assert config["validation"]["enabled"] is False
    by_epoch = defaultdict(list)
    for record in records:
        by_epoch[record["epoch"]].append(record)
    keys = lambda rows: {(r["segment_index"], r["segment_offset"]) for r in rows}
    common = set.intersection(*(keys(rows) for rows in by_epoch.values()))
    pass_size = max(map(len, by_epoch.values()))
    assert pass_size == 424 and len(common) == 248
    expected_keys = keys(by_epoch[0])
    rows = []
    for epoch, items in sorted(by_epoch.items()):
        assert len(keys(items)) == len(items)
        if len(items) == pass_size:
            assert keys(items) == expected_keys
        matched = [r["loss"] for r in items
                   if (r["segment_index"], r["segment_offset"]) in common]
        rows.append({"epoch": epoch, "first_step": items[0]["step"],
                     "last_step": items[-1]["step"], "samples": len(items),
                     "complete_pass": len(items) == pass_size,
                     "mean_loss": float(np.mean([r["loss"] for r in items])),
                     "matched_samples": len(matched),
                     "matched_mean_loss": float(np.mean(matched))})

    evaluation_dir = RUN / "eval/cold_full_zero_exact_gc/matched32"
    evals = {}
    for tag in ("swa_step02000-08000", "step10000"):
        d = read(evaluation_dir / f"{tag}.json")
        assert d["evaluation_status"] == "complete" and d["evaluated_samples"] == 32
        assert d["target_steps"] == 40 and d["eval_mode"] == "cold_full"
        assert d["residual_state_init"] == "zero"
        g = d["original_graphcast_loss"]
        b, f = np.array(g["baseline_per_step"]), np.array(g["full_per_step"])
        assert b.shape == f.shape == (40,) and np.isfinite([b, f]).all()
        assert np.isclose(g["baseline_rollout"], b.mean())
        assert np.isclose(g["full_rollout"], f.mean())
        assert np.isclose(g["improvement_pct_rollout"], 100 * (1 - f.mean() / b.mean()))
        evals[tag] = d
    swa, final = evals.values()
    for key in ("chosen_idx", "ckpt_in", "eval_feedback", "anchor_history_steps",
                "anchor_index_semantics", "warmup_steps", "resolution"):
        assert swa[key] == final[key], key
    g_swa, g_final = swa["original_graphcast_loss"], final["original_graphcast_loss"]
    baseline_difference = float(max(abs(
        np.array(g_final["baseline_per_step"]) / np.array(g_swa["baseline_per_step"]) - 1)))
    assert baseline_difference < .01

    with (data_dir / "training_passes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = [r for r in rows if r["complete_pass"]]
    last_change = 100 * (rows[-1]["matched_mean_loss"] / rows[-2]["matched_mean_loss"] - 1)
    summary = {
        "run": str(RUN.relative_to(ROOT)), "training_steps": len(records),
        "complete_passes": len(complete), "updates_per_complete_pass": pass_size,
        "last_partial_pass_updates": len(by_epoch[max(by_epoch)]),
        "matched_last_pass_loss_change_pct": last_change,
        "last_complete_pass_loss_change_pct": 100 * (
            complete[-1]["mean_loss"] / complete[-2]["mean_loss"] - 1),
        "epoch11_to_epoch22_loss_change_pct": 100 * (
            rows[22]["mean_loss"] / rows[11]["mean_loss"] - 1),
        "learning_rate_at_steps": {str(s): float(lr[s - 1]) for s in (200, 5000, 8000, 10000)},
        "gradient_norm_max": float(grad.max()),
        "gradient_clipping_fraction": float(np.mean(grad > config["optimizer"]["grad_clip"])),
        "exact_eval_rollout_reduction_pct": {
            tag: d["original_graphcast_loss"]["improvement_pct_rollout"] for tag, d in evals.items()},
        "exact_eval_full_rollout_loss": {
            tag: d["original_graphcast_loss"]["full_rollout"] for tag, d in evals.items()},
        "max_per_lead_baseline_relative_difference": baseline_difference,
        "limitations": ["No inline validation history; validation was disabled.",
                        "SWA is an averaged model, not an intermediate checkpoint.",
                        "Training losses are online closed-loop losses, not fixed-checkpoint validation.",
                        "The final pass is incomplete; use matched data positions for its comparison."],
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources.append({"path": str(Path(__file__).relative_to(ROOT)),
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    output.with_suffix(".sources.json").write_text(json.dumps(sources, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": INK,
                         "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.1))
    fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.84, wspace=.24, hspace=.40)
    fig.suptitle("Fresh width128: training is nearly flat; evaluation saturation is unmeasured",
                 x=.08, ha="left", y=.965, fontsize=16, fontweight="bold")
    fig.text(.08, .922, "1° resolution · residual MP2 · Mamba1 di16 / BCG1 · batch 1 · seed 22 · 10,000 updates",
             fontsize=10.5)
    fig.text(.08, .887, "Late training improves slightly. The final model beats early SWA, but two artifacts cannot establish an eval plateau.",
             fontsize=10.5)

    ax = axes[0, 0]
    ax.plot(steps, loss, color=BLUE, alpha=.17, lw=.45, rasterized=True, label="Each update")
    smooth = np.convolve(loss, np.ones(pass_size) / pass_size, mode="valid")
    ax.plot(steps[pass_size - 1:], smooth, color=INK, lw=1.6, label="424-update trailing mean")
    ax.set(title="A  Recorded training loss", ylabel="Training loss (lower is better)")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot([r["last_step"] for r in complete], [r["mean_loss"] for r in complete],
            color=BLUE, marker="o", ms=3.5, lw=1.7, label="All 424 chunks per complete pass")
    ax.plot([r["last_step"] for r in rows], [r["matched_mean_loss"] for r in rows],
            color=ORANGE, marker="o", ms=3.5, lw=1.7, label="Same 248 chunks in every pass")
    ax.set(title="B  Compare repeated data positions", ylabel="Mean training loss (zoomed scale)",
           ylim=(3.742, 3.809))
    ax.text(.98, .12, f"Last matched pass: {last_change:+.3f}% loss\nFinal pass contains only 248 / 424 chunks",
            transform=ax.transAxes, ha="right", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    ax = axes[1, 0]
    ax.plot(steps, lr / 1e-4, color=BLUE, lw=2)
    ax.axvspan(2000, 8000, color=INK, alpha=.055)
    ax.text(5000, .13, "SWA members: 2k, 3k, …, 8k", ha="center", fontsize=9)
    ax.annotate("Ends at 1e−5", xy=(10000, lr[-1] / 1e-4), xytext=(6200, .37),
                arrowprops={"arrowstyle": "->", "color": INK}, fontsize=9)
    ax.set(title="C  Actual learning-rate schedule", ylabel="Learning rate (× 10⁻⁴)", ylim=(0, 1.12))

    ax = axes[1, 1]
    values = [g_swa["improvement_pct_rollout"], g_final["improvement_pct_rollout"]]
    bars = ax.bar([0, 1], values, color=["#7a929c", ORANGE], width=.54)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=5, fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["SWA 2k–8k\n(averaged weights)", "Step 10k\n(single checkpoint)"])
    ax.set(title="D  Only two completed evaluation artifacts", ylabel="Exact GraphCast loss reduction (%)",
           ylim=(0, 8.8))
    ax.text(.5, .93, "32 matched starts · 40 six-hour leads\nHigher is better; no temporal eval curve available",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(0, 10300)
        ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2k", "4k", "6k", "8k", "10k"])
        ax.set_xlabel("Optimizer updates")
    for ax in axes.flat:
        ax.grid(axis="y", color="#dbe2e6", alpha=.7, lw=.65)
        ax.set_axisbelow(True)
    fig.text(.08, .083, "Training loss averages all 24 BPTT steps on evolving closed-loop trajectories; it is distinct from 10-day evaluation loss.", fontsize=9)
    fig.text(.08, .060, "Panel B controls data-position coverage; the last partial pass must not be compared directly with a full pass.", fontsize=9)
    fig.text(.08, .037, "No evaluation at individual 2k–9k checkpoints was found. Better final performance than SWA does not establish continuing late eval gains.", fontsize=9)
    fig.savefig(output)
    png = image_dir / "training_and_evaluation.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(json.dumps({"pdf": str(output), "png": str(png), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
