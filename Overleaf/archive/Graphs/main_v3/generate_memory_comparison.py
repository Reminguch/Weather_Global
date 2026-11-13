#!/usr/bin/env python3
"""Plot the principal model's exact loss alongside the saved reset controls.

Run from the repository root after sourcing scripts/graphcast_env.sh.
"""
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "main_v2_figures", HERE.parent / "main_v2/generate_figures.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    best = source.read_json(source.BEST)
    source.verify_exact(best, source.read_json(source.REF)["chosen_idx"])
    days = source.np.arange(1, 41) / 4
    exact = best["original_graphcast_loss"]["improvement_pct_per_step"]
    rows = source.read_csv(source.MEM_CSV)
    curves = [("Carry / carry (main model)", exact, source.BEST,
               "exact", "#0072B2", "-")]
    for policy, label, color, linestyle in source.POLICIES[1:]:
        subset = sorted((r for r in rows if int(r["bcg"]) == 1 and
                         r["swa"] == source.WINDOWS[0] and r["policy"] == policy),
                        key=lambda r: float(r["lead_days"]))
        source.np.testing.assert_allclose(
            [float(r["lead_days"]) for r in subset], days)
        paths = {r["source"] for r in subset}
        assert len(paths) == 1
        path = source.ROOT / paths.pop()
        assert source.read_json(path)["chosen_idx"] == best["chosen_idx"]
        values = [float(r["approximate_gc_loss_reduction_pct"]) for r in subset]
        assert source.np.isfinite(values).all()
        curves.append((label, values, path, "approximate", color, linestyle))

    fig, ax = source.plt.subplots(figsize=(6.5, 3.25))
    fig.subplots_adjust(left=.12, right=.975, bottom=.29, top=.89)
    for label, values, path, metric, color, linestyle in curves:
        ax.plot(days, values, label=label, color=color, linestyle=linestyle, linewidth=1.6)
    source.format_leads(ax)
    ax.axhline(0, color=".5", linewidth=.8)
    ax.set_ylim(-22, 27)
    ax.set_ylabel("GraphCast loss\nreduction (%)")
    ax.set_title(r"Temporal memory comparison ($d_{\mathrm{inner}}=16$)", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, .01),
               columnspacing=1.2, handletextpad=.6,
               title="Training/forecasting memory policy")
    stem = HERE / "memory_reset_main_model"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Principal model and temporal memory controls",
        "Subject": "Exact principal-model loss; approximate controls from separate runs.",
        "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=170)
    source.plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["policy", "metric", "lead_days", "loss_reduction_pct"])
        for label, values, path, metric, color, linestyle in curves:
            writer.writerows((label, metric, day, value) for day, value in zip(days, values))
    provenance = {
        "generator": source.rel(Path(__file__).resolve()),
        "generator_sha256": source.sha(Path(__file__).resolve()),
        "control_csv": {"path": source.rel(source.MEM_CSV), "sha256": source.sha(source.MEM_CSV)},
        "matched_32_indices_verified": True,
        "comparison": "Principal model replaces the historical carry/carry curve. Controls use separate training runs and approximate loss; this is not a shared-weight reset ablation of the principal model.",
        "curves": [{"label": label, "metric": metric, "source": source.rel(path),
                    "sha256": source.sha(path), "day10_pct": values[-1]}
                   for label, values, path, metric, color, linestyle in curves],
    }
    stem.with_suffix(".sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
