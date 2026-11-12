#!/usr/bin/env python3
"""Reuse saved non-SWA AMSE evaluations without running new forecasts.

Three training histories remain separate. The main figures use exact GC MSE
and zonal powers from the same eight 2023 starts. Development32 is a separate
CSV/table and never supplies the MSE axis for those spectral comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "amse_lambda_raw_cached_20261110"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
BASE = ROOT / "artifacts/checkpoints/v24_Ilya"
FIELDS = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850"]
POWER_KEYS = ("truth_power", "baseline_power", "full_power", "baseline_error_power", "full_error_power")
LEADS = list(range(6, 241, 6))
SOURCES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = digest
    return digest


def read(path):
    pin(path)
    return json.loads(path.read_text())


def load(directory, label, color, step):
    evaluation = read(directory / "evaluation.json")
    require(evaluation["evaluation_status"] == "complete" and evaluation["metrics_complete"]
            and evaluation["evaluated_samples"] == evaluation["n_samples"] == 8
            and evaluation["remaining_samples"] == 0 and evaluation["target_steps"] == 40,
            f"Incomplete spectral evaluation: {directory}")
    require(Path(evaluation["ckpt"]).name == f"checkpoint_step{step:08d}.pkl", "Expected raw endpoint")
    require(pin(Path(evaluation["ckpt"])) == evaluation["checkpoint_sha256"], "Checkpoint hash differs")
    require(pin(Path(evaluation["ckpt_in"])) == evaluation["baseline_checkpoint_sha256"], "Baseline hash differs")
    paths = sorted(directory.glob("*.npz"))
    require(len(paths) == 8, "Missing or extra spectrum dates")
    arrays, dates, units = [], [], None
    for path in paths:
        pin(path)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            for key in ("checkpoint_sha256", "baseline_checkpoint_sha256", "protocol_hash", "model_id"):
                require(meta[key] == evaluation[key], f"Spectrum {key} differs")
            require(meta["status"] == "complete" and meta["initialization_id"] == path.stem
                    and meta["lead_hours"] == LEADS, "Spectrum metadata coverage differs")
            require(meta["spectral_metadata"]["diagnostic"] == "zonal_fourier_spectrum"
                    and meta["spectral_metadata"]["latitude_band_degrees_absolute"] == [30., 60.],
                    "Unexpected spectrum definition")
            require(archive["lead_hours"].tolist() == LEADS and archive["wavenumber"].tolist() == list(range(181)),
                    "Spectrum axes differ")
            indices = [archive["variables"].tolist().index(field) for field in FIELDS]
            current_units = archive["units"][indices].tolist()
            require(units is None or current_units == units, "Spectrum units differ")
            units = current_units
            values = {key: np.asarray(archive[key][:, indices, 1:], dtype=float) for key in POWER_KEYS}
            require(all(value.shape == (40, 9, 180) and np.isfinite(value).all() and (value >= 0).all()
                        for value in values.values()), "Corrupt/nonfinite spectrum")
            arrays.append(values)
            dates.append(meta["initialization_time"])
    require(len(set(dates)) == 8 and dates == sorted(evaluation["global_initialization_times"]), "Spectrum dates differ")
    loss = evaluation["original_graphcast_loss"]
    for branch in ("baseline", "full"):
        value = np.asarray(loss[branch + "_per_step"])
        require(value.shape == (40,) and np.isfinite(value).all() and (value >= 0).all(), "Invalid exact GC MSE")
        require(np.isfinite(loss[branch + "_rollout"]) and loss[branch + "_rollout"] >= 0, "Invalid rollout MSE")
    return {"label": label, "color": color, "step": step, "dates": dates, "units": units,
            "directory": str(directory.relative_to(ROOT)), "checkpoint": evaluation["ckpt"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"], "evaluation": evaluation,
            "mse": np.asarray(loss["full_per_step"]), "gc_mse": np.asarray(loss["baseline_per_step"]),
            "rollout": loss["full_rollout"], "gc_rollout": loss["baseline_rollout"],
            **{key: np.mean([value[key] for value in arrays], axis=0) for key in POWER_KEYS}}


def groups():
    pilot = BASE / "res1_amse_joint_regime_20261004/pilot/evaluation/results"
    september24 = BASE / "res1_amse_blend_lambda0p1_20260924/eval/spectral"
    september19 = BASE / "res1_amse_pair_20260919/eval/spectral"
    return {
        "october500": {"title": "October continuation · raw step 500", "models": [
            load(pilot / "mse_step00500/spectral", "Exact-grid MSE control", "#222222", 500),
            load(pilot / "spectral_mse_step00500/spectral", "Spectral MSE λ=0", "#0072B2", 500),
            load(pilot / "amse_l025_step00500/spectral", "AMSE λ=0.25", "#009E73", 500),
            load(pilot / "amse_l050_step00500/spectral", "AMSE λ=0.5", "#D55E00", 500)]},
        "september24": {"title": "September 24 campaign · raw step 2000", "models": [
            load(september24 / "mse_fp32/step2000", "Exact-grid MSE control", "#222222", 2000),
            load(september24 / "amse/step2000", "AMSE λ=0.1", "#0072B2", 2000)]},
        "september19": {"title": "September 19 campaign · raw step 2000", "models": [
            load(september19 / "mse_fp32/step2000", "Exact-grid MSE control", "#222222", 2000),
            load(september19 / "amse/step2000", "AMSE λ=1", "#CC79A7", 2000)]},
    }


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def save(figure, stem):
    for extension in ("png", "pdf"):
        figure.savefig(IMAGES / f"{stem}.{extension}", dpi=170, facecolor="white")
    plt.close(figure)


def style(axis):
    axis.grid(alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True); IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    all_groups = groups()
    days, wave = np.arange(1, 41) / 4, np.arange(1, 181)
    summary, curves, development, differences = [], [], [], []
    overview, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for row, (group_id, group) in enumerate(all_groups.items()):
        canonical = group["models"][0]
        truth = canonical["truth_power"]
        mask = np.logical_and.reduce([model["truth_power"] > 0 for model in group["models"]])
        require(mask.all(axis=(1, 2)).any() and mask[-1].any(), "Missing truth power")
        gc = {"label": "Public GC", "color": "#888888", "mse": canonical["gc_mse"],
              "rollout": canonical["gc_rollout"], "full_power": canonical["baseline_power"], "step": None}
        spread = max(abs(model["gc_rollout"] - gc["rollout"]) for model in group["models"])
        for model in group["models"]:
            require(model["dates"] == canonical["dates"] and model["units"] == canonical["units"], "Group cohort/units differ")
            for field, actual, reference in (("truth_power", model["truth_power"], truth),
                    ("baseline_power", model["baseline_power"], canonical["baseline_power"]),
                    ("baseline_mse", model["gc_mse"], canonical["gc_mse"])):
                delta = float(np.max(np.abs(actual - reference)))
                bound = 1e-6 * float(np.max(np.abs(reference)))
                differences.append({"group": group_id, "model": model["label"], "field": field,
                    "max_absolute_difference": delta, "reference_absolute_bound": bound,
                    "exceeds_reference": delta > bound, "policy": "finite differences are warnings only"})
            source_dir = ROOT / model["directory"]
            if group_id == "october500":
                exact = source_dir.parent / "exact/evaluation.json"
            else:
                exact = Path(str(source_dir).replace("/eval/spectral/", "/eval/exact/")) / "evaluation.json"
            saved = read(exact)
            require(saved["evaluation_status"] == "complete" and saved["evaluated_samples"] == 32
                    and saved["checkpoint_sha256"] == model["checkpoint_sha256"], "Development evidence differs")
            specification = read(exact.parent / "specification.json")
            require(specification["evaluation_year"] == 2022 and specification["n_samples"] == 32
                    and specification["chosen_idx"] == saved["chosen_idx"],
                    "Development cohort is not the prescribed32 starts in2022")
            loss = saved["original_graphcast_loss"]
            require(all(np.isfinite(loss[key]) and loss[key] >= 0 for key in ("full_rollout", "baseline_rollout")), "Invalid development MSE")
            development.append({"group": group_id, "model": model["label"], "samples": 32, "year": 2022,
                "exact_gc_rollout_mse": loss["full_rollout"], "paired_gc_rollout_mse": loss["baseline_rollout"],
                "paired_mse_reduction_pct": 100 * (1-loss["full_rollout"]/loss["baseline_rollout"]),
                "checkpoint": model["checkpoint"], "source": str(exact.relative_to(ROOT))})
        models = [gc] + group["models"]
        for model in models:
            error = 100 * np.abs(ratio(model["full_power"], truth)-1)
            power_curve = np.asarray([error[i][mask[i]].mean() for i in range(40)])
            summary.append({"group": group_id, "model": model["label"], "step": model["step"],
                "samples": 8, "year": 2023, "exact_gc_rollout_mse": model["rollout"],
                "mse_reduction_pct": 100 * (1-model["rollout"]/gc["rollout"]),
                "day10_mse": model["mse"][-1], "day10_mse_reduction_pct": 100*(1-model["mse"][-1]/gc["mse"][-1]),
                "power_error_rollout_pct": error[mask].mean(), "power_error_day10_pct": power_curve[-1],
                "decrease_exceeds_observed_gc_repeat_spread": bool(gc["rollout"]-model["rollout"] > spread),
                "gc_repeat_spread": spread, "zero_truth_bins_excluded": int(mask.size-mask.sum())})
            for i, day in enumerate(days):
                curves.append({"group": group_id, "model": model["label"], "lead_days": day,
                               "exact_gc_mse": model["mse"][i], "power_error_pct": power_curve[i]})
            style_args = {"color": model["color"], "ls": "--" if model is gc else "-", "label": model["label"]}
            axes[row, 0].plot(days, model["mse"], **style_args)
            axes[row, 1].plot(days, power_curve, **style_args)
            axes[row, 2].plot(wave, ratio(model["full_power"][39, 5], truth[39, 5]), **style_args)
        axes[row, 0].set(title=group["title"], xlabel="Forecast lead (days)", ylabel="Exact GraphCast weighted MSE")
        axes[row, 1].set(title="Zonal power mismatch: nine primitive fields", xlabel="Forecast lead (days)", ylabel="Mean absolute relative power error (%)")
        axes[row, 2].set(title="T850 power ratio · day 10", xlabel="Zonal wavenumber", ylabel="Forecast power / ERA5 power", xscale="log")
        axes[row, 2].axhline(1, color="black", ls=":", lw=1)
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]: style(ax)
        np.savez_compressed(DATA / f"{group_id}_pooled.npz", fields=FIELDS, units=canonical["units"],
            lead_hours=LEADS, wavenumber=wave, labels=[model["label"] for model in models],
            truth_power=truth, forecast_power=np.stack([model["full_power"] for model in models]),
            common_truth_mask=mask, exact_gc_mse=np.stack([model["mse"] for model in models]),
            initialization_times=canonical["dates"])
        for kind in ("power_spectra", "power_ratios"):
            figure, panels = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
            for i, field in enumerate(("T2m", "T850", "U850")):
                channel = FIELDS.index(field)
                for j, lead in enumerate((19, 39)):
                    ax = panels[i, j]
                    for model in models:
                        values = model["full_power"][lead, channel]
                        if kind == "power_ratios": values = ratio(values, truth[lead, channel])
                        ax.plot(wave, values, color=model["color"], ls="--" if model is gc else "-", label=model["label"])
                    if kind == "power_spectra":
                        ax.plot(wave, truth[lead, channel], color="black", ls="-.", label="ERA5 truth")
                        ax.set_yscale("log")
                    else: ax.axhline(1, color="black", ls=":", label="ERA5 truth = 1")
                    ax.set(xscale="log", title=f"{field} · day {(lead+1)/4:g}", xlabel="Zonal wavenumber",
                           ylabel=f"Power ({canonical['units'][channel]} squared)" if kind == "power_spectra" else "Forecast power / ERA5 power")
                    style(ax)
            panels[0, 0].legend(fontsize=8)
            figure.suptitle(f"{group['title']} · eight 2023 starts\nZonal spectra at 30–60° N/S; powers pooled before ratios", fontsize=13)
            save(figure, f"{group_id}_{kind}")
    overview.suptitle("Historical AMSE checkpoints without SWA · cached eight-start 2023 evaluation\nSeparate training histories; exact MSE and zonal power use the same forecasts", fontsize=15)
    save(overview, "mse_and_power_overview")
    csv_file(DATA / "summary_2023_eight_starts.csv", summary)
    csv_file(DATA / "lead_curves_2023_eight_starts.csv", curves)
    csv_file(DATA / "development_2022_32_starts.csv", development)
    provenance = {"status": "complete", "sources": SOURCES, "finite_reference_differences": differences,
        "groups": {key: {"title": group["title"], "models": [{name:model[name] for name in ("label", "checkpoint", "checkpoint_sha256", "directory", "dates")} for model in group["models"]]} for key,group in all_groups.items()},
        "summary": summary, "development": development,
        "power_definition": "Zonal Fourier spectra, both hemispheres at30–60deg latitude; modes1–180; nine primitive fields, all40leads. Pool date powers before ratios; equally average valid bins.",
        "mse_definition": "Exact original GraphCast weighted grid MSE across all variables globally; common GC denominator within each training-history group.",
        "limitations": "Eight previously explored2023 dates, no intervals or training-seed uncertainty. Separate histories are not one controlled lambda sweep. Development32 results remain separate. Zonal spectra differ from the new pilot's spherical T179 spectra. No principal-M1 raw spectra are claimed here."}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    lines = ["# Historical AMSE without SWA: cached MSE and power spectra", "",
             "All main figures use the same eight2023 starts and40 six-hour leads for exact GC MSE and zonal power. The three training histories are kept separate; no new forecasts were run.", "",
             "| Training group | Model | Exact rollout MSE | Reduction vs group GC | Zonal power error |",
             "| --- | --- | ---: | ---: | ---: |"]
    lines += [f"| {row['group']} | {row['model']} | {row['exact_gc_rollout_mse']:.6f} | {row['mse_reduction_pct']:+.3f}% | {row['power_error_rollout_pct']:.3f}% |" for row in summary]
    lines += ["", provenance["power_definition"], "", provenance["limitations"], "",
              "The October λ0→0.25→0.5 series improves zonal power matching as exact MSE increases. "
              "The September λ0.1 arm has modestly better power matching and slightly higher MSE than its own control. "
              "The September λ1 arm has much better power matching but higher MSE than both its control and public GC. "
              "These observations concern these fixed checkpoints and eight dates.", "",
              "Finite repeated-baseline/truth differences are recorded with1e-6 relative reference bounds in provenance.json. The CSV separately records whether MSE decreases exceed the observed GC repeat spread. Neither is an acceptance gate.", "",
              "The separate `development_2022_32_starts.csv` contains existing32-start2022 exact-loss results; those values do not enter the eight-start spectral figures.", ""]
    for path in sorted(IMAGES.glob("*.png")):
        lines += [f"![{path.stem.replace('_',' ')}](../../../images/resolution_eval/{NAME}/{path.name})", ""]
    lines += ["Checkpoint paths and hashes are in provenance.json. All displayed model checkpoints are raw step500 or2000; no SWA checkpoint is displayed."]
    (DATA / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report": str(DATA / 'report.md'), "overview": str(IMAGES / 'mse_and_power_overview.png'), "summary": summary},indent=2))


if __name__ == "__main__":
    main()
