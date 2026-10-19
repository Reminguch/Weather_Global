"""Plot the 2023 common-grid analogs of manuscript Figs. 5 and 9 (main_v3).

Two normalizations are written:

* RMSE relative to the paired GraphCast (1°) baseline on all 122 starts, from
  the frozen campaign summary. The blue band is the paired five-start-block
  bootstrap 95% interval for GraphCast–Mamba versus that baseline.
* RMSE relative to the IFS ENS mean, as in the archived figures. ENS scores
  come from score_ens_common_grid_2023.py; its source store lacks some chunks,
  so each (diagnostic, lead) pools every model over exactly the starts that
  have ENS data. No interval is drawn for this version.

    python scripts/analyze_models/plot_common_grid_2023.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Overleaf/Graphs"))
from paper_style import apply_style, FIGURE_WIDTH, LABEL_SIZE

CAMPAIGN = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015"
SUMMARY = CAMPAIGN / "report/summary.json"
ENS = CAMPAIGN / "ens/ens_mean_mse.json"
OUT = ROOT / "plots/analyze_models/images/resolution_eval/common_grid_2023_20261015"
BLUE, GRAY, GREEN = "#26609f", "#8b919b", "#1fa450"
SELECTED = ["Z500", "T2m", "U850"]


def load():
    data = json.loads(SUMMARY.read_text())
    days = np.asarray(data["lead_hours"]) / 24
    rmse = {k: np.asarray(v) for k, v in data["rmse"].items()}
    vs_gc1 = data["analyses"]["primary"]["comparisons"]["gc1"]
    curves = {
        "m1": 100 * rmse["m1"] / rmse["gc1"],
        "gc025": 100 * rmse["gc025"] / rmse["gc1"],
        # Reduction intervals map to relative-RMSE intervals as 100 - x.
        "m1_lo": 100 - np.asarray(vs_gc1["ci95_upper"]),
        "m1_hi": 100 - np.asarray(vs_gc1["ci95_lower"]),
    }
    assert np.allclose(curves["m1"], 100 - np.asarray(vs_gc1["rmse_reduction_pct"]))
    return data, days, curves


def per_start_mse(diagnostics, starts):
    """Return {model: array[start, lead, diagnostic]} from the campaign shards."""
    records = {}
    for shard in sorted((CAMPAIGN / "shards").glob("*.json")):
        for record in json.loads(shard.read_text())["per_initialization"]:
            for model, scores in record["mse"].items():
                records[model, record["initialization_time"]] = scores
    ens = {r["initialization_time"]: r["mse"]
           for r in json.loads(ENS.read_text())["per_initialization"]}
    out = {}
    for model in ("m1", "gc1", "gc025", "ens"):
        source = ens if model == "ens" else {s: records[model, s] for s in starts}
        out[model] = np.array([[[np.nan if source[s][d][i] is None else source[s][d][i]
                                 for d in diagnostics] for i in range(40)] for s in starts],
                              dtype=np.float64)
    return out


def load_ens(data):
    """ENS-relative curves, each model pooled over the starts ENS covers."""
    starts = json.loads((CAMPAIGN / "protocol.json").read_text())["initialization_times"]
    assert len(starts) == data["initializations"]
    mse = per_start_mse(data["diagnostics"], starts)
    # The shard pooling must reproduce the campaign summary exactly.
    for model in ("m1", "gc1", "gc025"):
        assert np.allclose(np.sqrt(mse[model].mean(axis=0)), data["rmse"][model], rtol=1e-12)
    covered = np.isfinite(mse["ens"])
    count = covered.sum(axis=0)
    def pooled(model):
        return np.sqrt(np.where(covered, mse[model], 0).sum(axis=0) / count)
    ens = pooled("ens")
    curves = {m: 100 * pooled(m) / ens for m in ("m1", "gc1", "gc025")}
    return curves, count


def plot_panel(ax, days, curves, j, label):
    if "gc1" in curves:
        ax.plot(days, curves["gc1"][:, j], color=GRAY, lw=1.5, ls="--")
    else:
        ax.axhline(100, color=GRAY, lw=1.5, ls="--")
    if "m1_lo" in curves:
        ax.fill_between(days, curves["m1_lo"][:, j], curves["m1_hi"][:, j],
                        color=BLUE, alpha=.18, lw=0)
    ax.plot(days, curves["gc025"][:, j], color=GREEN, lw=1.7)
    if "m1" in curves:
        ax.plot(days, curves["m1"][:, j], color=BLUE, lw=1.7)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.tick_params(length=2.5, width=.65, pad=2.5)
    ax.set_title(label, pad=5)


def shared_legend(fig, y, *, band, mamba=True):
    handles = [Line2D([], [], color=BLUE, lw=1.7),
               Line2D([], [], color=GRAY, lw=1.5, ls="--"),
               Line2D([], [], color=GREEN, lw=1.7)]
    labels = ["GraphCast–Mamba", "GraphCast", "GraphCast (0.25°)"]
    if not mamba:
        handles, labels = handles[1:], labels[1:]
    if band:
        handles.append(Patch(color=BLUE, alpha=.18, lw=0))
        labels.append("95% interval")
    return fig.legend(handles, labels,
        loc="upper center", bbox_to_anchor=(.52, y), ncol=len(labels),
        frameon=False, handlelength=2.1, handletextpad=.5,
        columnspacing=1.1, borderaxespad=0)


def write_figure(path, days, curves, diagnostics, labels, *, full, ylabel):
    rows, cols, height = (3, 4, 5.3) if full else (1, 3, 2.35)
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH, height),
                             squeeze=False)
    if full:
        fig.subplots_adjust(left=.112, right=.988, bottom=.12, top=.845,
                            wspace=.39, hspace=.53)
        fig.suptitle("2023 forecasts on the common 1.5° verification grid",
                     x=.52, y=.994)
        shared_legend(fig, .937, band="m1_lo" in curves, mamba="m1" in curves)
    else:
        fig.subplots_adjust(left=.112, right=.988, bottom=.265, top=.80,
                            wspace=.29)
        shared_legend(fig, .985, band="m1_lo" in curves, mamba="m1" in curves)
    for ax, label in zip(axes.flat, labels, strict=True):
        plot_panel(ax, days, curves, diagnostics.index(label), label)
    fig.supxlabel("Forecast lead time (days)", x=.55,
                  y=.025 if full else .035, fontsize=LABEL_SIZE)
    fig.supylabel(ylabel, x=.015, y=.49, fontsize=LABEL_SIZE)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    apply_style()
    data, days, curves = load()
    diagnostics = data["diagnostics"]
    OUT.mkdir(parents=True, exist_ok=True)
    gc_label = "RMSE vs GraphCast [%]"
    outputs = [
        write_figure(OUT / "common_grid_2023_all_variables.pdf", days, curves,
                     diagnostics, diagnostics, full=True, ylabel=gc_label),
        write_figure(OUT / "common_grid_2023_three_panels.pdf", days, curves,
                     diagnostics, SELECTED, full=False, ylabel=gc_label),
    ]
    ens_curves, count = load_ens(data)
    outputs += [
        write_figure(OUT / "common_grid_2023_all_variables_vs_ens.pdf", days, ens_curves,
                     diagnostics, diagnostics, full=True, ylabel="RMSE vs ENS [%]"),
        write_figure(OUT / "common_grid_2023_three_panels_vs_ens.pdf", days, ens_curves,
                     diagnostics, SELECTED, full=False, ylabel="RMSE vs ENS [%]"),
        # GraphCast-only view: 1° backbone against the 0.25° reference.
        write_figure(OUT / "common_grid_2023_three_panels_vs_ens_gc_only.pdf", days,
                     {k: ens_curves[k] for k in ("gc1", "gc025")},
                     diagnostics, SELECTED, full=False, ylabel="RMSE vs ENS [%]"),
        write_figure(OUT / "common_grid_2023_all_variables_vs_ens_gc_only.pdf", days,
                     {k: ens_curves[k] for k in ("gc1", "gc025")},
                     diagnostics, diagnostics, full=True, ylabel="RMSE vs ENS [%]"),
    ]
    sources = {"summary": str(SUMMARY.relative_to(ROOT)),
               "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
               "ens_scores": str(ENS.relative_to(ROOT)),
               "ens_scores_sha256": hashlib.sha256(ENS.read_bytes()).hexdigest(),
               "protocol_hash": data["protocol_hash"],
               "normalization": {
                   "gc": "RMSE / paired GraphCast (1°) RMSE, same 122 starts",
                   "ens": ("RMSE / IFS ENS-mean RMSE; per (diagnostic, lead) every model "
                           "is pooled over the starts with ENS data"),
               },
               "interval": "gc version only: primary paired 5-start-block bootstrap, M1 vs GC1",
               "ens_starts_per_diagnostic_lead": {
                   d: count[:, j].tolist() for j, d in enumerate(diagnostics)},
               "ens_starts_min_max": [int(count.min()), int(count.max())],
               "ens_vs_gc_values_day_10": {
                   m: {d: round(float(v[-1, j]), 3) for j, d in enumerate(diagnostics)}
                   for m, v in ens_curves.items()},
               "outputs": outputs}
    (OUT / "common_grid_2023_figures.sources.json").write_text(
        json.dumps(sources, indent=2) + "\n")
    for item in outputs:
        print(item["path"])


if __name__ == "__main__":
    main()
