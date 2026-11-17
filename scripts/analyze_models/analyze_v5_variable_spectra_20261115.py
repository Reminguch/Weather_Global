#!/usr/bin/env python3
"""Audit matched annual accuracy and spectra by channel; never run forecasts.

The default input is the pinned four-model v5 comparison on 122 starts.
All rankings are descriptive fixed-checkpoint comparisons, not significance
tests. No manuscript files are changed. With --plot-only, reuse audited exports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "Overleaf/archive/Graphs/v5_spectral_matched122_20261115"
MODELS = ["GC", "M1", "GC_filter", "R1"]
LABELS = ["Raw GraphCast", "M1", "Filtered GraphCast", "R1"]
COLORS = ["#2864A5", "#D97720", "#238B8E", "#9467BD"]
SHORT = {"10m_u_component_of_wind": "U10", "10m_v_component_of_wind": "V10",
         "2m_temperature": "T2m", "geopotential": "Z", "mean_sea_level_pressure": "MSL",
         "specific_humidity": "Q", "temperature": "T", "total_precipitation_6hr": "TP6h",
         "u_component_of_wind": "U", "v_component_of_wind": "V", "vertical_velocity": "W"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def channel_name(channel):
    variable, level = channel
    return SHORT[variable] + (str(int(level)) if level is not None else "")


def analyze(out):
    summary = json.loads((BUNDLE / "summary.json").read_text())
    pins = {row["path"]: row["sha256"] for row in summary["sources"]}
    paths = {
        "spectra": "plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_2023_power_20261113/power_pooled.npz",
        "grid": "artifacts/evaluations/v24_optimal_smoothing_v4_20261112/annual_pooled.npz",
        "reset": "artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/power_pooled.npz",
        "filters": "artifacts/evaluations/v24_optimal_smoothing_v4_20261112/filters.json",
    }
    for path in paths.values():
        if sha(ROOT / path) != pins[path]:
            raise ValueError(f"Pinned input changed: {path}")
    layout = json.loads((ROOT / paths["filters"]).read_text())["layout"]
    channels = layout["channels"]
    scales, weights = np.asarray(layout["scales"]), np.asarray(layout["weights"])
    with np.load(ROOT / paths["spectra"], allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        assert meta["branches"] == ["GC", "M1"] and meta["variants"] == ["raw", "projected", "optimal"]
        assert meta["initialization_ids"] == summary["cohort"]["initialization_ids"]
        truth = data["truth_power"]
        pred = np.stack([data["prediction_power"][:, 0, 0], data["prediction_power"][:, 1, 0], data["prediction_power"][:, 0, 2]], axis=1)
        cross = np.stack([data["cross_power"][:, 0, 0], data["cross_power"][:, 1, 0], data["cross_power"][:, 0, 2]], axis=1)
        exact = np.stack([data["exact_gc_loss"][:, 0, 0], data["exact_gc_loss"][:, 1, 0], data["exact_gc_loss"][:, 0, 2]], axis=1)
    with np.load(ROOT / paths["grid"], allow_pickle=False) as data:
        gridmeta = json.loads(str(data["metadata"]))
        assert gridmeta["initialization_ids"] == meta["initialization_ids"]
        grid = np.stack([data["grid_mse"][:, 0, 0], data["grid_mse"][:, 1, 0], data["grid_mse"][:, 0, 2]], axis=1)
    with np.load(ROOT / paths["reset"], allow_pickle=False) as data:
        resetmeta = json.loads(str(data["metadata"]))
        assert resetmeta["initialization_ids"] == meta["initialization_ids"] and resetmeta["layout"] == layout
        assert resetmeta["branches"] == ["GC", "R_matched"]
        np.testing.assert_allclose(data["truth_power"], truth, rtol=1e-9, atol=1e-12)
        pred = np.concatenate([pred, data["prediction_power"][:, 1:2]], axis=1)
        cross = np.concatenate([cross, data["cross_power"][:, 1:2]], axis=1)
        grid = np.concatenate([grid, data["normalized_grid_mse"][:, 1:2] * scales[None, None, :] ** 2], axis=1)
        exact = np.concatenate([exact, data["exact_gc_loss"][:, 1:2]], axis=1)
    for array in [pred, truth, cross, grid, exact]:
        assert np.isfinite(array).all()
    assert (pred > 0).all() and (truth > 0).all() and (grid > 0).all()
    reconstructed = np.sum(grid * (weights / scales**2)[None, None], axis=-1)
    np.testing.assert_allclose(reconstructed, exact, rtol=1e-12, atol=1e-12)
    ratio = pred[..., 1:] / truth[:, None, :, 1:]
    amplitude = np.sqrt(ratio)
    coherence = (cross / np.sqrt(pred * truth[:, None]))[..., 1:]
    assert grid.shape == (40, 4, 83) and ratio.shape == (40, 4, 83, 179)
    degree_mismatch = 100 * np.mean(np.abs(ratio - 1), axis=-1)
    physical_diff = np.abs(pred[..., 1:] - truth[:, None, :, 1:])
    truth_sum = truth[..., 1:].sum(axis=-1)
    power_weighted = 100 * physical_diff.sum(axis=-1) / truth_sum[:, None]
    coherence_weighted = np.sum(coherence * truth[:, None, :, 1:], axis=-1) / truth_sum[:, None]
    for row in summary["table1_rows"]:
        model = "R1" if row["model"] == "reset" else row["model"]
        mi = MODELS.index(model)
        idx = np.arange(40) if row["window"] == "rollout" else np.array([19 if row["window"] == "day5" else 39])
        np.testing.assert_allclose(exact[idx, mi].mean(), row["exact_gc_loss"], rtol=1e-12)
        np.testing.assert_allclose(degree_mismatch[idx, mi].mean(), row["power_mismatch_pct"], rtol=1e-12)
    names = [channel_name(c) for c in channels]
    rows, contrasts, lead_rows = [], [], []
    for window, idx in [("day5", np.array([19])), ("day10", np.array([39])), ("rollout", np.arange(40))]:
        rmse = np.sqrt(grid[idx].mean(axis=0))
        dp = degree_mismatch[idx].mean(axis=0)
        pw = 100 * physical_diff[idx].sum(axis=(0, 3)) / truth_sum[idx].sum(axis=0)[None]
        coh = (coherence[idx] * truth[idx, None, :, 1:]).sum(axis=(0, 3)) / truth_sum[idx].sum(axis=0)[None]
        for ci, (variable, level) in enumerate(channels):
            for mi, model in enumerate(MODELS):
                rows.append(dict(window=window, channel=names[ci], variable=variable, pressure_hpa=level,
                                 model=model, physical_grid_rmse=float(rmse[mi, ci]),
                                 equal_degree_power_mismatch_pct=float(dp[mi, ci]),
                                 truth_power_weighted_mismatch_pct=float(pw[mi, ci]),
                                 truth_power_weighted_coherence=float(coh[mi, ci]),
                                 equal_degree_coherence=float(coherence[idx, mi, ci].mean())))
            for ri in [0, 2, 3]:
                reduction = 100 * (1 - rmse[1, ci] / rmse[ri, ci])
                delta = pw[1, ci] - pw[ri, ci]
                label = ("lower RMSE / lower mismatch" if reduction > 0 and delta < 0 else
                         "lower RMSE / higher mismatch" if reduction > 0 else
                         "higher RMSE / lower mismatch" if delta < 0 else "higher RMSE / higher mismatch")
                contrasts.append(dict(window=window, channel=names[ci], variable=variable, pressure_hpa=level,
                                      reference=MODELS[ri], rmse_reduction_pct=float(reduction),
                                      power_weighted_mismatch_delta_pp=float(delta),
                                      equal_degree_mismatch_delta_pp=float(dp[1, ci] - dp[ri, ci]),
                                      power_weighted_coherence_delta=float(coh[1, ci] - coh[ri, ci]),
                                      equal_degree_coherence_delta=float(coherence[idx, 1, ci].mean() - coherence[idx, ri, ci].mean()),
                                      category=label))
    for li in range(40):
        for ci, name in enumerate(names):
            for mi, model in enumerate(MODELS):
                lead_rows.append(dict(lead_days=(li + 1) / 4, channel=name, model=model,
                                      physical_grid_rmse=float(np.sqrt(grid[li, mi, ci])),
                                      equal_degree_power_mismatch_pct=float(degree_mismatch[li, mi, ci]),
                                      truth_power_weighted_mismatch_pct=float(power_weighted[li, mi, ci]),
                                      truth_power_weighted_coherence=float(coherence_weighted[li, mi, ci])))
    # Exact per-channel power-ratio criterion. Store first crossings and recrosses.
    horizon_rows = []
    for deficit in [0.20, 0.25, 0.30]:
        for degree in [4, 10, 20, 40, 80, 160]:
            for ci, name in enumerate(names):
                for mi, model in enumerate(MODELS):
                    curve = ratio[:, mi, ci, degree - 1]
                    ix = np.flatnonzero(curve < 1 - deficit)
                    first = int(ix[0]) if ix.size else None
                    horizon_rows.append(dict(channel=name, model=model, degree=degree, wavelength_km=40030/degree,
                                             power_deficit=deficit, first_crossing_day=(first+1)/4 if first is not None else None,
                                             below_at_first_evaluation=first == 0,
                                             recrosses=bool(np.any(curve[first:] >= 1-deficit)) if first is not None else False,
                                             status="not crossed within 10 days" if first is None else "observed"))
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [("channel_metrics.csv", rows), ("channel_contrasts.csv", contrasts),
                         ("lead_metrics.csv", lead_rows), ("scale_crossings.csv", horizon_rows)]:
        write_csv(out/name, values)
    np.savez_compressed(out/"plot_data.npz", channels=np.array(names), variable=np.array([c[0] for c in channels]),
                        rmse=np.sqrt(grid), degree_mismatch=degree_mismatch, power_weighted_mismatch=power_weighted,
                        power_weighted_coherence=coherence_weighted,
                        selected_amplitude=amplitude[[19, 39]], selected_coherence=coherence[[19, 39]])
    counts = {}
    for window in ["day5", "day10", "rollout"]:
        selected = [r for r in contrasts if r["window"] == window and r["reference"] == "R1"]
        counts[window] = {label:sum(r["category"] == label for r in selected) for label in sorted({r["category"] for r in selected})}
    report = dict(cohort_count=122, leads=40, channels=83, models=MODELS, cohort=summary["cohort"],
                  methods={"accuracy":"sqrt(mean over selected leads of date-pooled exact native-grid channel MSE); no spectral truncation",
                           "equal_degree_mismatch":"100*mean_{lead,degree=1..179}|Pf/Py-1|; legacy diagnostic",
                           "truth_power_weighted_mismatch":"100*sum_{lead,degree=1..179}|Pf-Py| / sum_{lead,degree=1..179}Py; within each physical variable/level, not across units",
                           "truth_power_weighted_coherence":"sum Py*rho / sum Py, same selected leads and positive degrees",
                           "scale_crossing":"first six-hour lead with per-channel Pf/Py < 1-deficit; explicitly flags recrossing and first-lead censoring",
                           "classification":"signs of M1 RMSE reduction and truth-power-weighted mismatch difference; descriptive, no acceptance gate",
                           "selection":"post-hoc illustrations of joint changes, tradeoffs and deterioration; full83-channel results retained",
                           "spectral_fields":"physical fields without climatology subtraction; degree0 excluded; large-scale static structure can dominate power weighting",
                           "uncertainty":"fixed checkpoints; pooled-date point estimates, no new date/seed intervals"},
                  checks={"reconstructed_exact_loss_max_abs":float(np.max(np.abs(reconstructed-exact))),
                          "reproduces_all_annual_table_metrics":True, "source_hashes_verified":True},
                  counts_vs_R1=counts, retained_numerical_diagnostics=summary.get("numerical_diagnostics"),
                  sources=[dict(path=p,sha256=pins[p]) for p in paths.values()],
                  code_sha256=sha(__file__))
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    shutil.copy2(__file__, out/Path(__file__).name)
    print(json.dumps(dict(counts=counts, checks=report["checks"]),indent=2))


def plot(data_dir, image_dir, fields):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.dpi":160})
    image_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir/"plot_data.npz",allow_pickle=False) as archive:
        d = {k:archive[k] for k in archive.files}
    names = list(d["channels"]); idx = [names.index(f) for f in fields]
    days=np.arange(1,41)/4
    handles=[Line2D([],[],color=c,lw=1.6,label=l) for c,l in zip(COLORS,LABELS)]
    def finish(fig, stem, title):
        fig.suptitle(title, fontsize=12)
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0,.045,1,.95))
        for ext in ["png","pdf"]: fig.savefig(image_dir/f"{stem}.{ext}")
        plt.close(fig)
    fig,axes=plt.subplots(3,len(idx),figsize=(3.1*len(idx),8.0),squeeze=False,sharex=True)
    metrics=[("rmse","RMSE / R1"),("power_weighted_mismatch","Power-weighted mismatch (%)"),
             ("power_weighted_coherence","Power-weighted coherence")]
    for col,ci in enumerate(idx):
        axes[0,col].set_title(names[ci])
        for row,(metric,label) in enumerate(metrics):
            ax=axes[row,col]
            for mi in range(4):
                values=d[metric][:,mi,ci]
                if metric=="rmse": values=values/d[metric][:,3,ci]
                ax.plot(days,values,color=COLORS[mi],lw=1.4)
            if metric=="rmse": ax.axhline(1,color=".6",ls=":",lw=.7)
            if col==0: ax.set_ylabel(label)
            ax.set_xlim(.25,10); ax.grid(alpha=.15)
        axes[-1,col].set_xlabel("Forecast lead (days)")
    finish(fig,"selected_accuracy_and_spectral_metrics","Selected variables: exact grid accuracy and power-weighted diagnostics | 122 starts")
    fig,axes=plt.subplots(4,len(idx),figsize=(3.1*len(idx),10),squeeze=False,sharex=True)
    wave=40030/np.arange(1,180)
    for col,ci in enumerate(idx):
        axes[0,col].set_title(names[ci])
        for li,day in enumerate([5,10]):
            for kind,key in enumerate(["selected_amplitude","selected_coherence"]):
                ax=axes[2*li+kind,col]
                for mi in range(4): ax.plot(wave,d[key][li,mi,ci],color=COLORS[mi],lw=1.4)
                if kind==0:
                    ax.axhspan(np.sqrt(.7),np.sqrt(.8),color=".5",alpha=.12)
                    ax.axhline(np.sqrt(.75),color=".3",ls="--",lw=.8)
                    ax.axhline(1,color=".6",ls=":",lw=.7)
                    ax.set_ylim(bottom=0)
                else: ax.set_ylim(-.05,1.03)
                if col==0: ax.set_ylabel(f"Day {day}: "+("amplitude ratio" if kind==0 else "coherence"))
                ax.set_xscale("log"); ax.set_xlim(40030,40030/179)
                ax.set_xticks([10000,2000,500]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
                ax.grid(alpha=.15)
        axes[-1,col].set_xlabel("Wavelength (km)")
    finish(fig,"selected_scale_diagnostics","Day-5 and day-10 spectra | Dashed amplitude line: 25% power deficit; band: 20–30%")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
    categories=list(dict.fromkeys(d["variable"])); colors=plt.get_cmap("tab20").colors
    for ax,li,day in zip(axes,[19,39],[5,10]):
        x=100*(1-d["rmse"][li,1]/d["rmse"][li,3])
        y=d["power_weighted_mismatch"][li,1]-d["power_weighted_mismatch"][li,3]
        for j,var in enumerate(categories):
            mask=d["variable"]==var
            ax.scatter(x[mask],y[mask],s=22,color=colors[j],alpha=.85,label=SHORT[var])
        for ci in idx:
            ax.annotate(names[ci],(x[ci],y[ci]),xytext=(4,5),textcoords="offset points",fontsize=8)
        ax.axvline(0,color=".4",lw=.7); ax.axhline(0,color=".4",lw=.7)
        ax.set(title=f"Day {day}: M1 versus R1",xlabel="Native-grid RMSE reduction (%) → better")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Power-weighted mismatch difference (pp)\n↓ better")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=11,frameon=False,fontsize=8)
    fig.suptitle("All 83 variable/pressure channels; point estimates on the matched 122-start cohort",fontsize=11)
    fig.tight_layout(rect=(0,.09,1,.94))
    for ext in ["png","pdf"]: fig.savefig(image_dir/f"all_channels_accuracy_power.{ext}")
    plt.close(fig)
    (image_dir/"selection.json").write_text(json.dumps(dict(fields=fields,selection="exploratory illustrative examples; full channel data retained",data=str(data_dir.relative_to(ROOT))),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115")
    parser.add_argument("--image-dir",type=Path,default=ROOT/"plots/analyze_models/images/resolution_eval/v5_variable_spectral_audit_20261115")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--fields",default="",help="Comma-separated channel labels; leave empty to export metrics only")
    args=parser.parse_args()
    if not args.plot_only: analyze(args.data_dir)
    if args.fields: plot(args.data_dir,args.image_dir,args.fields.split(","))


if __name__=="__main__": main()
