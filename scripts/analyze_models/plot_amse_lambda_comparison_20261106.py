"""Compare completed scalar-AMSE lambda experiments using saved forecasts only.

Run after sourcing scripts/graphcast_env.sh. The two training histories remain
separate; lambda here interpolates spectral MSE and AMSE, not exact grid MSE.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004"
NAME = "amse_lambda_comparison_20261106"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
FIELDS = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850"]
SELECTED_LEADS = [3, 11, 27, 39]
sources, warnings, summaries, plotted_curves = {}, [], [], []


def identity(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory, label, color, linestyle="-"):
    path = directory / "evaluation.json"
    identity(path)
    evaluation = json.loads(path.read_text())
    if evaluation.get("evaluation_status") != "complete" or evaluation["n_samples"] != 8:
        raise ValueError(f"Incomplete evaluation: {path}")
    arrays, dates = [], []
    for path in sorted(directory.glob("*.npz")):
        identity(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint mismatch: {path}")
            if meta["baseline_checkpoint_sha256"] != evaluation["baseline_checkpoint_sha256"]:
                raise ValueError(f"Baseline checkpoint mismatch: {path}")
            dates.append(meta["initialization_time"])
            ix = [z["variables"].tolist().index(f) for f in FIELDS]
            arrays.append({key: z[key][:, ix, 1:].copy() for key in
                           ("truth_power", "baseline_power", "full_power",
                            "baseline_error_power", "full_error_power")})
            units = z["units"][ix].tolist()
    if len(dates) != 8 or len(set(dates)) != 8 or dates != sorted(evaluation["global_initialization_times"]):
        raise ValueError(f"Initialization mismatch: {directory}")
    pooled = {key: np.mean([a[key] for a in arrays], axis=0) for key in arrays[0]}
    if not all(np.isfinite(a).all() and (a >= 0).all() for a in pooled.values()):
        raise ValueError(f"Invalid spectral inputs: {directory}")
    loss = evaluation["original_graphcast_loss"]
    return dict(label=label, color=color, ls=linestyle, dates=dates, units=units,
                checkpoint_sha256=evaluation["checkpoint_sha256"], directory=str(directory.relative_to(ROOT)),
                mse=np.asarray(loss["full_per_step"]), gc_mse=np.asarray(loss["baseline_per_step"]),
                **pooled)


def phase_model(phase, model, *args):
    return load(CAMPAIGN / phase / "evaluation/results" / model / "spectral", *args)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"{stem}.{ext}", dpi=170, facecolor="white")
    plt.close(fig)


def style(ax):
    ax.grid(alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    groups = {
        "pilot": dict(title="Matched 500-update continuation", subtitle="Same mature MSE parent; K=20; seed 22", models=[
            phase_model("pilot", "mature_mse", "Parent before continuation", "#999999", ":"),
            phase_model("pilot", "mse_step00500", "Exact-grid MSE control", "#222222"),
            phase_model("pilot", "spectral_mse_step00500", "Spectral MSE: λ=0", "#0072B2", "--"),
            phase_model("pilot", "amse_l025_step00500", "AMSE λ=0.25", "#009E73"),
            phase_model("pilot", "amse_l050_step00500", "AMSE λ=0.5", "#D55E00")]),
        "historical": dict(title="Earlier 2,000-update training; prescribed SWA", subtitle="Shared warm-up parent; separate frozen runtime versions", models=[
            phase_model("trajectory", "mature_mse", "MSE control (λ=0.1 campaign)", "#222222"),
            load(ROOT / "artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919/eval/spectral/mse_fp32/swa_step00500-02000",
                 "MSE control (λ=1 campaign)", "#777777", ":"),
            phase_model("trajectory", "amse_lambda0p1_swa", "AMSE λ=0.1", "#0072B2"),
            phase_model("trajectory", "full_amse_swa", "AMSE λ=1", "#CC79A7")])}
    days = np.arange(1, 41) / 4
    k = np.arange(1, 181)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("GC–Mamba: existing AMSE λ experiments", fontsize=17, y=.985)
    fig.text(.5, .95, "Both rows: the same eight 2023 starts · each row uses one common original-GC denominator", ha="center")
    for row, (name, group) in enumerate(groups.items()):
        parent = group["models"][0]
        gc = dict(label="Original GC", color="#777777", ls="--", mse=parent["gc_mse"],
                  full_power=parent["baseline_power"], full_error_power=parent["baseline_error_power"])
        truth = parent["truth_power"]
        group["gc"] = gc
        group["truth"] = truth
        for model in group["models"]:
            if model["dates"] != parent["dates"]:
                raise ValueError("Cohorts differ")
            # Repeated baseline/truth drift is retained, never an acceptance gate.
            warnings.append(dict(group=name, model=model["label"],
                gc_rollout_drift_pct=float(100*(model["gc_mse"].mean()/gc["mse"].mean()-1)),
                truth_power_max_abs_difference=float(np.max(np.abs(model["truth_power"]-truth)))))
        all_models = [gc] + group["models"]
        for model in all_models:
            power_error = 100*np.mean(np.abs(model["full_power"]/truth-1), axis=(1, 2))
            model["power_error"] = power_error
            mse_reduction = 100*(1-model["mse"]/gc["mse"])
            summaries.append(dict(group=name, model=model["label"],
                exact_gc_rollout_mse=float(model["mse"].mean()),
                mse_reduction_pct=float(100*(1-model["mse"].sum()/gc["mse"].sum())),
                day10_mse_reduction_pct=float(mse_reduction[-1]),
                power_error_pct=float(power_error[SELECTED_LEADS].mean()),
                power_error_reduction_vs_gc_pct=float(100*(1-power_error[SELECTED_LEADS].mean()/gc["power_error"][SELECTED_LEADS].mean()))))
            for li, day in enumerate(days):
                plotted_curves.append(dict(group=name, model=model["label"], lead_days=float(day),
                    exact_gc_mse=float(model["mse"][li]), mse_reduction_pct=float(mse_reduction[li]),
                    relative_power_error_pct=float(power_error[li])))
            kwargs = dict(color=model["color"], ls=model["ls"], lw=1.8, label=model["label"])
            axes[row, 0].plot(days, mse_reduction, **kwargs)
            axes[row, 1].plot(days[SELECTED_LEADS], power_error[SELECTED_LEADS], marker="o", ms=4, **kwargs)
            axes[row, 2].semilogx(k, model["full_power"][39, FIELDS.index("T850")]/truth[39, FIELDS.index("T850")], **kwargs)
        axes[row, 0].set(title=group["title"], xlabel="Forecast lead (days)", ylabel="Exact GC MSE reduction (%)\nHigher is better")
        axes[row, 0].text(.02, .97, group["subtitle"], transform=axes[row, 0].transAxes, va="top", fontsize=8)
        axes[row, 0].axhline(0, color="#777777", lw=.7)
        axes[row, 1].set(title="Power mismatch across nine fields", xlabel="Forecast lead (days)",
                         ylabel="Mean relative power error (%)\nLower is better", xticks=[1, 3, 7, 10])
        axes[row, 2].set(title="T850 power spectrum at day 10", xlabel="Zonal wavenumber", ylabel="Forecast power / ERA5 power\nCloser to 1 is better", xlim=(1, 180))
        axes[row, 2].axhline(1, color="#111111", ls=":", lw=.8)
        axes[row, 2].set_xticks([1, 10, 100]); axes[row, 2].xaxis.set_major_formatter(ScalarFormatter())
        for ax in axes[row]:
            style(ax)
        axes[row, 1].legend(fontsize=8, loc="upper left", frameon=False)
        # Preserve the exact plotted spectral arrays and model/checkpoint identity.
        np.savez_compressed(DATA/f"{name}_pooled.npz", fields=FIELDS, units=parent["units"], lead_days=days, wavenumber=k,
            labels=[m["label"] for m in all_models], truth_power=truth,
            forecast_power=np.stack([m["full_power"] for m in all_models]),
            forecast_error_power=np.stack([m["full_error_power"] for m in all_models]))
        for metric in ("power_ratio", "absolute_power", "error_ratio", "power_mismatch_vs_gc"):
            panel, axs = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
            if metric == "power_ratio":
                title, ylabel = "Forecast power / ERA5 power", "Power ratio (ideal = 1)"
            elif metric == "absolute_power":
                title, ylabel = "Forecast and ERA5 power spectra", "Power (field units squared)"
            elif metric == "error_ratio":
                title, ylabel = "Forecast-error power / original GC error power", "Error ratio (lower is better)"
            else:
                title, ylabel = "Absolute power mismatch / original GC mismatch", "Mismatch ratio (lower is better)"
            for vi, field in enumerate(FIELDS):
                ax = axs.flat[vi]
                for model in all_models:
                    power = model["full_power"][39, vi]
                    if metric == "power_ratio":
                        values = power/truth[39, vi]
                    elif metric == "absolute_power":
                        values = power
                    elif metric == "error_ratio":
                        values = model["full_error_power"][39, vi]/gc["full_error_power"][39, vi]
                    else:
                        den = np.abs(gc["full_power"][39, vi]-truth[39, vi])
                        values = np.divide(np.abs(power-truth[39, vi]), den,
                                           out=np.full_like(den, np.nan), where=den != 0)
                    ax.plot(k, values, color=model["color"], ls=model["ls"], lw=1.5, label=model["label"])
                ax.set(xscale="log", xlim=(1, 180), title=field)
                if metric == "absolute_power":
                    ax.plot(k, truth[39, vi], color="#111111", ls="-.", lw=1, label="ERA5 truth")
                    ax.set_yscale("log")
                else:
                    ax.axhline(1, color="#111111", ls=":", lw=.8)
                    ax.set_ylim(bottom=0)
                    if metric == "power_mismatch_vs_gc":
                        ax.set_yscale("symlog", linthresh=1, linscale=.5)
                if vi % 3 == 0: ax.set_ylabel(ylabel)
                if vi >= 6: ax.set_xlabel("Zonal wavenumber")
                ax.set_xticks([1, 10, 100]); ax.xaxis.set_major_formatter(ScalarFormatter())
                style(ax)
            panel.suptitle(f"{title} — day 10\n{group['title']} · eight 2023 starts · 30–60° N/S zonal spectra", fontsize=14)
            panel.legend(*axs.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .025))
            panel.text(.5, .009, "Powers pooled across starts before ratios; unsmoothed curves. Training histories differ between pilot and historical groups.", ha="center", fontsize=8)
            panel.tight_layout(rect=(0, .10, 1, .93))
            save(panel, f"{name}_{metric}_day10")
    fig.text(.5, .025, "Power diagnostics: zonal spectra at 30–60° N/S; positive modes 1–180. Exact MSE covers all fields globally and all 40 leads.", ha="center", fontsize=9)
    fig.text(.5, .007, "λ interpolates spectral MSE and full AMSE. The rows are separate experiments, not a single controlled λ sweep. Point estimates on an explored sample.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .93), h_pad=3)
    save(fig, "mse_and_power_overview")
    # Include the saved pilot development checkpoints, without mixing raw losses
    # from objectives whose lambda changes during the first 100 updates.
    report_path = CAMPAIGN / "pilot/evaluation/report/joint_selection.json"
    identity(report_path)
    report = json.loads(report_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for family, label, color in [("mse", "Exact-grid MSE", "#222222"),
            ("spectral_mse", "Spectral MSE: λ=0", "#0072B2"),
            ("amse_l025", "AMSE λ=0.25", "#009E73"), ("amse_l050", "AMSE λ=0.5", "#D55E00")]:
        records = sorted([r for r in report["records"] if r["family"] == family], key=lambda r:r["step"])
        x = [r["step"] for r in records]
        axes[0].plot(x, [100*(1-r["score"]["exact_gc_loss_ratio_vs_gc"]) for r in records], "o-", color=color, label=label)
        axes[1].plot(x, [r["score"]["mean_relative_power_error_pct"] for r in records], "o-", color=color)
    for ax in axes:
        ax.set_xlabel("Completed continuation updates"); style(ax)
    axes[0].set_ylabel("Exact GC rollout MSE reduction (%)\nHigher is better"); axes[0].legend(fontsize=9, frameon=False)
    axes[1].set_ylabel("Mean relative power error (%)\nLower is better")
    fig.suptitle("Evolution across saved checkpoints: matched eight-start 2023 forecast evaluation", fontsize=13)
    fig.text(.5, .025, "Each point is a complete 10-day forecast evaluation, not minibatch training loss. Positive λ ramps up over the first 100 updates.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .92)); save(fig, "pilot_checkpoint_tradeoff")
    for filename, rows in [("summary.csv", summaries), ("rollout.csv", plotted_curves)]:
        with (DATA/filename).open("w") as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest=dict(sources=sources, finite_drift_warnings=warnings,
        loss_definition="spectral_MSE + lambda * (full_AMSE - spectral_MSE); exact-grid MSE is a separate control",
        power_error_definition="100*mean(abs(pooled_forecast_power/pooled_truth_power-1)); nine primitive fields, k1:180, days1/3/7/10",
        groups={name:dict(title=g["title"], models=[{k:m[k] for k in ["label","checkpoint_sha256","directory","dates"]} for m in g["models"]]) for name,g in groups.items()})
    (DATA/"provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
