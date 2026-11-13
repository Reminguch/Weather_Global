#!/usr/bin/env python3
"""Report matched raw-checkpoint pilot MSE and physical spherical spectra."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal
from scripts.experiments.frozen_runtime import _digest
from scripts.analyze_models.comparison_policy import POLICY, partition_models, presentation_directory

POWER_KEYS = ("truth_power", "prediction_power", "error_power", "cross_power")
REFERENCE_RELATIVE = 1e-6
FIELDS = (("2m_temperature", None, "T2m", "K²"),
          ("geopotential", 500, "Z500 (geopotential)", "m⁴ s⁻⁴"),
          ("u_component_of_wind", 850, "U850", "m² s⁻²"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def divide(numerator, denominator):
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def power_error(power, truth, mask):
    """Pool date powers upstream, then equally average retained bins."""
    errors = 100 * np.abs(divide(power[..., 1:], truth[..., 1:]) - 1)
    return float(errors[mask].mean()), np.array([
        errors[i][mask[i]].mean() if mask[i].any() else np.nan
        for i in range(len(errors))])


def finite_difference(model, field, actual, reference):
    delta = float(np.max(np.abs(actual - reference)))
    scale = float(np.max(np.abs(reference)))
    bound = REFERENCE_RELATIVE * scale
    return {"model": model, "field": field, "maximum_absolute_difference": delta,
            "reference_maximum_absolute": scale, "reference_relative": REFERENCE_RELATIVE,
            "reference_absolute_bound": bound, "exceeds_reference": delta > bound,
            "policy": "Finite differences are advisory; no acceptance gate."}


def verify_computational_sources(parent, manifest):
    """Reused forecasts must have the same model, metric and spectral runtime."""
    snapshots = []
    shared_scripts = {"scripts/experiments/run_v24_full_gc_ar.py",
                      "scripts/experiments/run_v24_smoothing.py", "scripts/graphcast_env.sh"}
    for campaign in (parent, manifest):
        source = Path(campaign["source_root"])
        snapshot = read(source / "snapshot.json")
        require(snapshot["source_digest"] == campaign["source_digest"] == _digest(snapshot["files"]),
                "Reused source snapshot digest differs")
        selected = {path: checksum for path, checksum in snapshot["files"].items()
                    if path.startswith(("src/", "third_party/")) or path in shared_scripts}
        require(selected and "third_party/graphcast/graphcast/graphcast.py" in selected,
                "Reused computational source snapshot is incomplete")
        for relative, checksum in selected.items():
            check({"path": str(source / relative), "sha256": checksum})
        snapshots.append(selected)
    require(snapshots[0] == snapshots[1], "Reused computational source differs")


def evaluation_context(root, manifest, model, verified_parents):
    reuse = model.get("reuse")
    if reuse is None:
        return root / "evaluation" / model["id"], manifest, []
    check(reuse["campaign"])
    parent = unseal(reuse["campaign"]["path"])
    require(parent["record_hash"] == reuse["manifest_hash"], "Reused campaign hash differs")
    require(parent["source_digest"] == reuse["source_digest"], "Reused source digest differs")
    require(parent["cohort"] == manifest["cohort"] and parent["target_steps"] == manifest["target_steps"],
            "Reused cohort or leads differ")
    require(parent["baseline"] == manifest["baseline"], "Reused baseline differs")
    candidates = [item for item in parent["models"] if item["id"] == model["id"]]
    require(len(candidates) == 1, "Reused model identity is missing or duplicated")
    original = candidates[0]
    require(all(original[key] == model[key] for key in ("checkpoint", "run_config", "model_family")),
            "Reused checkpoint, config or model family differs")
    expected = Path(reuse["campaign"]["path"]).parent / "evaluation" / model["id"]
    directory = Path(reuse["directory"])
    require(directory.resolve() == expected.resolve(), "Reused evaluation directory differs")
    if parent["record_hash"] not in verified_parents:
        verify_computational_sources(parent, manifest)
        verified_parents.add(parent["record_hash"])
    return directory, parent, [reuse["campaign"]]


def load_models(root, manifest):
    check(manifest["cohort"])
    cohort = read(manifest["cohort"]["path"])
    dates = [v["initialization_time"] if isinstance(v, dict) else v
             for v in cohort["initializations"]]
    require(len(set(dates)) == len(dates) and dates == sorted(dates), "Invalid pilot dates")
    require(manifest["target_steps"] == 40, "Pilot requires 40 six-hour leads")
    models = manifest["models"]
    require(models and models[0]["id"] == "M1_raw", "Canonical reference must be M1_raw")
    require(len({m["id"] for m in models}) == len(models), "Duplicate model identity")
    items, layout, baseline_path, sources = {}, None, None, [pinned(root / "manifest.json")]
    verified_parents = set()
    for model in models:
        identity = model["id"]
        directory, evaluation_manifest, reused_sources = evaluation_context(root, manifest, model, verified_parents)
        sources.extend(reused_sources)
        record_path = directory / "complete.json"
        record = unseal(record_path)
        require(record.get("status", "complete") == "complete", "Incomplete model record")
        require(record["manifest_hash"] == evaluation_manifest["record_hash"], "Campaign hash differs")
        require(record["checkpoint"] == model["checkpoint"], "Checkpoint record differs")
        for key in ("checkpoint", "evaluation", "pooled_spectra"):
            check(record[key])
        check(model["run_config"])
        result = read(record["evaluation"]["path"])
        require(result["evaluation_status"] == "complete" and result["metrics_complete"]
                and result["remaining_samples"] == 0, "Incomplete evaluator result")
        require(result["model_id"] == identity, "Evaluation model identity differs")
        require(result["evaluated_samples"] == record["count"] == len(dates), "Sample count differs")
        require(result["target_steps"] == 40, "Lead coverage differs")
        require(result["initialization_manifest_sha256"] == manifest["cohort"]["sha256"],
                "Truth cohort identity differs")
        baseline_path = baseline_path or result["ckpt_in"]
        require(result["ckpt_in"] == baseline_path, "Public GC checkpoint differs")
        per_start = result["original_graphcast_loss_per_initialization"]
        require([r["initialization_time"] for r in per_start] == dates, "Evaluation dates differ")
        require(all(r["lead_hours"] == list(range(6, 241, 6)) for r in per_start),
                "Per-initialization lead identities differ")
        losses = {}
        for branch in ("baseline", "full"):
            values = np.asarray(result["original_graphcast_loss"][branch + "_per_step"], dtype=float)
            require(values.shape == (40,) and np.isfinite(values).all() and (values >= 0).all(),
                    "Invalid exact GC losses")
            scalar = float(result["original_graphcast_loss"][branch + "_rollout"])
            require(np.isfinite(scalar) and scalar >= 0, "Invalid rollout GC loss")
            losses[branch] = values
            losses[branch + "_rollout"] = scalar
        with np.load(record["pooled_spectra"]["path"], allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            current_layout = metadata["layout"]
            if layout is None:
                layout = current_layout
            require(current_layout == layout, "Channel/normalization layouts differ")
            require(metadata["count"] == len(dates), "Spectrum count differs")
            require(metadata["initialization_ids"] == [r["initialization_id"] for r in per_start],
                    "Spectrum initialization identities differ")
            require(metadata["lead_hours"] == list(range(6, 241, 6)), "Spectrum leads differ")
            require(metadata["campaign_hash"] == evaluation_manifest["record_hash"], "Spectrum campaign differs")
            require(metadata["cohort_sha256"] == manifest["cohort"]["sha256"], "Spectrum truth cohort differs")
            require(metadata["source_digest"] == evaluation_manifest["source_digest"], "Spectrum source differs")
            require(metadata["lmax"] == 179, "Spectrum degree range differs")
            require(metadata["checkpoint_sha256"] == model["checkpoint"]["sha256"],
                    "Spectrum checkpoint identity differs")
            require(metadata["branches"] == ["public_gc", identity], "Spectrum branches differ")
            powers = {key: np.asarray(saved[key], dtype=float) for key in POWER_KEYS}
        require(len(layout["channels"]) == 83, "Expected all 83 primitive channels")
        for key, values in powers.items():
            expected = (40, 83, 180) if key == "truth_power" else (40, 2, 83, 180)
            require(values.shape == expected and np.isfinite(values).all(), f"Invalid {key} shape/values")
            if key != "cross_power":
                require((values >= 0).all(), f"Corrupt negative {key}")
        items[identity] = {"model": model, "record": record, "result": result,
                           "loss": losses, "power": powers}
        sources.extend([pinned(record_path), record["checkpoint"], record["evaluation"],
                        record["pooled_spectra"], model["run_config"]])
    return items, layout, dates, sources


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(models):
    groups = {}
    for model in models:
        if model["id"].startswith("M1_"):
            continue
        group = model.get("plot_group", model["model_family"])
        groups.setdefault(group, []).append(model["id"])
    return groups or {"gc_mamba": []}


def figure_group_name(group):
    return re.sub(r"[^a-z0-9]+", "_", group.lower().replace("λ", "lambda")).strip("_")


def checkpoint_pairs(models):
    """Validate declared raw/SWA pairs without selecting them by their scores."""
    pairs = {}
    for model in models:
        pair = model.get("comparison_pair")
        if pair is None:
            continue
        kind = model.get("checkpoint_kind")
        require(kind in ("raw", "swa"), "Comparison pair requires raw/SWA checkpoint kind")
        members = pairs.setdefault(pair, {})
        require(kind not in members, "Duplicate checkpoint kind in comparison pair")
        members[kind] = model
    for members in pairs.values():
        require(set(members) == {"raw", "swa"}, "Incomplete raw/SWA comparison pair")
        raw, swa = members["raw"], members["swa"]
        require(all(raw.get(key) == swa.get(key) for key in (
            "model_family", "lambda", "run_config", "training_updates")),
            "Raw/SWA comparison pair has different training identities")
    return pairs


def paired_summary(models, series, spread, day10_spread):
    """Pair model scores after pooled losses/powers have been reduced."""
    rows = []
    for pair, members in checkpoint_pairs(models).items():
        raw, swa = members["raw"], members["swa"]
        row = {"comparison_pair": pair, "model_family": raw["model_family"],
               "lambda": raw["lambda"], "raw_model": raw["id"], "swa_model": swa["id"],
               "training_updates": raw.get("training_updates"),
               "swa_steps": ",".join(str(step) for step in swa.get("swa_steps", []))}
        for scope, noise in (("rollout", spread), ("day10", day10_spread)):
            raw_score = series[raw["id"]]["summary"]
            swa_score = series[swa["id"]]["summary"]
            raw_loss, swa_loss = [score[f"exact_gc_mse_{scope}"] for score in (raw_score, swa_score)]
            row.update({
                f"raw_exact_gc_mse_{scope}": raw_loss,
                f"swa_exact_gc_mse_{scope}": swa_loss,
                f"swa_mse_reduction_vs_raw_{scope}_pct": 100 * (1 - swa_loss / raw_loss) if raw_loss > 0 else None,
                f"swa_minus_raw_power_error_{scope}_pp": (
                    swa_score[f"power_error_{scope}_pct"] - raw_score[f"power_error_{scope}_pct"]),
                f"swa_mse_decrease_exceeds_baseline_repeat_spread_{scope}": bool(raw_loss - swa_loss > noise)})
        rows.append(row)
    return rows


def figure_styles(models, series):
    """Use the same color for a paired training run, solid raw/dashed SWA."""
    palette = ["#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "#7A5195"]
    colors = {"public_gc": "#777777", "M1_raw": "#0072B2", "M1_swa": "#222222"}
    by_id = {model["id"]: model for model in models}
    for identities in plot_groups(models).values():
        assigned = {}
        for identity in identities:
            key = by_id[identity].get("comparison_pair", identity)
            assigned.setdefault(key, palette[len(assigned) % len(palette)])
            colors[identity] = assigned[key]
    if any(model.get("comparison_pair") == "M1" for model in models):
        colors["M1_swa"] = colors["M1_raw"]
    return {identity: {"color": colors[identity],
                       "linestyle": "--" if identity in ("public_gc", "M1_swa")
                       or by_id.get(identity, {}).get("checkpoint_kind") == "swa" else "-",
                       "linewidth": 1.6} for identity in series}


def make_figures(image_root, models, series, channels, count, figure_title="Raw-checkpoint pilot"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = plot_groups(models)
    families = list(groups)
    reference_ids = [identity for identity in ("public_gc", "M1_raw", "M1_swa") if identity in series]
    styles = figure_styles(models, series)
    pairs = checkpoint_pairs(models)
    paired = bool(pairs)
    by_id = {model["id"]: model for model in models}
    figures = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            target = image_root / f"{name}.{suffix}"
            fig.savefig(target, dpi=170, bbox_inches="tight")
            figures.append(str(target))
        plt.close(fig)

    def family_ids(family):
        return reference_ids + groups[family]

    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 5), squeeze=False)
    for ax, family in zip(axes[0], families):
        for identity in family_ids(family):
            item = series[identity]
            ax.plot(np.arange(1, 41) / 4, item["loss"], label=item["label"], **styles[identity])
        ax.set(title=family.replace("_", " "), xlabel="Forecast lead (days)",
               ylabel="Exact GraphCast weighted MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"{figure_title} · {count} starts in 2022 · cold full-feedback forecasts")
    fig.tight_layout()
    save(fig, "mse_vs_lead")

    degrees = np.arange(1, 180)
    for family in families:
        for kind in ("power_spectrum", "power_ratio"):
            fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
            for row, (variable, level, label, units) in enumerate(FIELDS):
                channel = channels.index((variable, level))
                for col, lead in enumerate((19, 39)):
                    ax = axes[row, col]
                    truth = series["public_gc"]["truth"][lead, channel, 1:]
                    if kind == "power_spectrum":
                        ax.loglog(degrees, truth, color="black", linewidth=2, label="ERA5 truth")
                    else:
                        ax.axhline(1, color="black", linewidth=1, label="ERA5 truth = 1")
                    for identity in family_ids(family):
                        item = series[identity]
                        values = item["power"][lead, channel, 1:]
                        if kind == "power_ratio":
                            values = divide(values, truth)
                        ax.plot(degrees, values, label=item["label"], **styles[identity])
                    ax.set_xscale("log")
                    if kind == "power_spectrum":
                        ax.set_yscale("log")
                    ax.set(title=f"{label} · day {(lead + 1) / 4:g}", xlabel="Spherical degree",
                           ylabel=f"Physical power ({units})" if kind == "power_spectrum" else "Forecast power / ERA5 power")
                    ax.grid(alpha=.2)
            if paired:
                fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                           bbox_to_anchor=(.5, 0), ncols=3, fontsize=8)
            else:
                axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"{figure_title} · {family.replace('_', ' ')} · powers pooled over {count} dates")
            save(fig, f"{kind}_{figure_group_name(family)}")

    separate_histories = paired or any("plot_group" in model for model in models)
    comparisons = groups if separate_histories else {"all": list(series)}
    for group, identities in comparisons.items():
        selected = family_ids(group) if separate_histories else identities
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        for ax, label, suffix in zip(axes, ("All 40 leads", "Day 10"), ("rollout", "day10")):
            for identity in selected:
                item = series[identity]
                x, y = item["summary"][f"power_error_{suffix}_pct"], item["summary"][f"exact_gc_mse_{suffix}"]
                is_swa = by_id.get(identity, {}).get("checkpoint_kind") == "swa"
                marker = "*" if identity == "public_gc" else "s" if item["family"] == "full_gc" else "o"
                ax.scatter(x, y, edgecolor=styles[identity]["color"],
                           facecolor=styles[identity]["color"] if not is_swa else "none",
                           marker=marker, s=100, label=item["label"])
            for members in pairs.values():
                raw_id, swa_id = members["raw"]["id"], members["swa"]["id"]
                if raw_id in selected and swa_id in selected:
                    scores = [series[identity]["summary"] for identity in (raw_id, swa_id)]
                    ax.plot([score[f"power_error_{suffix}_pct"] for score in scores],
                            [score[f"exact_gc_mse_{suffix}"] for score in scores],
                            color=styles[raw_id]["color"], linewidth=.8, alpha=.6)
            ax.set(title=label, xlabel="Mean absolute relative power error (%)",
                   ylabel="Exact GraphCast weighted MSE")
            ax.grid(alpha=.2)
            note = "Lower is better on both axes"
            if paired:
                note += " · filled: raw; open: SWA"
            ax.text(.02, .02, note, transform=ax.transAxes, fontsize=9)
        axes[1].legend(fontsize=7, loc="best")
        group_title = f"{group} · " if separate_histories else ""
        title_prefix = f"{figure_title} · " if paired else ""
        fig.suptitle(f"{title_prefix}{group_title}MSE / power tradeoff · {count} starts · 83 channels, degrees 1–179")
        save(fig, f"mse_power_tradeoff_{figure_group_name(group)}" if separate_histories else "mse_power_tradeoff")
    return figures


def report(root):
    root = Path(root).resolve()
    manifest = unseal(root / "manifest.json")
    items, layout, dates, sources = load_models(root, manifest)
    models, excluded = partition_models(manifest["models"])
    included_ids = {model["id"] for model in models}
    data_root = presentation_directory(manifest["data_root"], excluded)
    image_root = presentation_directory(manifest["image_root"], excluded)
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    canonical = items["M1_raw"]
    truth = canonical["power"]["truth_power"]
    mask = np.logical_and.reduce([item["power"]["truth_power"][..., 1:] > 0 for item in items.values()])
    require(mask.any() and mask[-1].any(), "No valid truth-power bins")
    baseline = canonical["loss"]["baseline"]
    baseline_rollout = canonical["loss"]["baseline_rollout"]
    require(baseline_rollout > 0 and (baseline > 0).all(), "Public GC denominator must be positive")
    spread = max(abs(item["loss"]["baseline_rollout"] - baseline_rollout) for item in items.values())
    day10_spread = max(abs(item["loss"]["baseline"][-1] - baseline[-1]) for item in items.values())
    diagnostics, series = [], {}
    series["public_gc"] = {"label": "Public GC", "family": "reference", "step": None, "lambda": None,
                           "plot_group": "reference",
                           "loss": baseline, "rollout": baseline_rollout, "truth": truth,
                           "power": canonical["power"]["prediction_power"][:, 0]}
    for identity, item in items.items():
        model = item["model"]
        if identity not in included_ids:
            continue
        series[identity] = {"label": model["label"], "family": model["model_family"],
                            "plot_group": model.get("plot_group", "reference" if identity.startswith("M1_") else model["model_family"]),
                            "step": model["step"], "lambda": model["lambda"],
                            "checkpoint_kind": model.get("checkpoint_kind"),
                            "comparison_pair": model.get("comparison_pair"),
                            "training_updates": model.get("training_updates", model.get("total_training_updates")),
                            "loss": item["loss"]["full"], "rollout": item["loss"]["full_rollout"],
                            "truth": truth, "power": item["power"]["prediction_power"][:, 1]}
        for field, actual, reference in (
                ("truth_power", item["power"]["truth_power"], truth),
                ("public_gc_power", item["power"]["prediction_power"][:, 0], series["public_gc"]["power"]),
                ("public_gc_loss_per_lead", item["loss"]["baseline"], baseline)):
            diagnostics.append(finite_difference(identity, field, actual, reference))
    summary, curves = [], []
    for identity, item in series.items():
        mismatch, per_lead = power_error(item["power"], truth, mask)
        row = {"model": identity, "label": item["label"], "model_family": item["family"],
               "plot_group": item["plot_group"],
               "lambda": item["lambda"], "checkpoint_step": item["step"],
               "checkpoint_kind": item.get("checkpoint_kind"),
               "comparison_pair": item.get("comparison_pair"),
               "training_updates": item.get("training_updates"),
               "exact_gc_mse_rollout": item["rollout"], "exact_gc_mse_day10": float(item["loss"][-1]),
               "mse_reduction_rollout_pct": 100 * (1 - item["rollout"] / baseline_rollout),
               "mse_reduction_day10_pct": float(100 * (1 - item["loss"][-1] / baseline[-1])),
               "power_error_rollout_pct": mismatch, "power_error_day10_pct": float(per_lead[-1]),
               "rollout_decrease_exceeds_baseline_repeat_spread": bool(baseline_rollout - item["rollout"] > spread),
               "day10_decrease_exceeds_baseline_repeat_spread": bool(baseline[-1] - item["loss"][-1] > day10_spread)}
        item["summary"] = row
        summary.append(row)
        for lead in range(40):
            curves.append({"model": identity, "lead_days": (lead + 1) / 4,
                           "exact_gc_mse": float(item["loss"][lead]),
                           "mse_reduction_pct": float(100 * (1 - item["loss"][lead] / baseline[lead])),
                           "power_error_pct": float(per_lead[lead]) if np.isfinite(per_lead[lead]) else None})
    write_csv(data_root / "summary.csv", summary)
    write_csv(data_root / "lead_curves.csv", curves)
    pairs = paired_summary(models, series, spread, day10_spread)
    pair_path = data_root / "raw_vs_swa.csv"
    if pairs:
        write_csv(pair_path, pairs)
    pooled_path = data_root / "pooled_spectra.npz"
    np.savez_compressed(pooled_path, truth_power=truth, common_positive_truth_mask=mask,
                        models=np.asarray(list(series)), lead_hours=np.arange(1, 41) * 6,
                        degree=np.arange(180),
                        prediction_power=np.stack([v["power"] for v in series.values()]),
                        exact_gc_loss=np.stack([v["loss"] for v in series.values()]),
                        metadata=np.asarray(json.dumps({"layout": layout, "initialization_times": dates,
                                                        "canonical_baseline": "M1_raw evaluation"})))
    figures = make_figures(image_root, models, series,
                           [tuple(c) for c in layout["channels"]], len(dates),
                           figure_title=manifest.get("figure_title", "Raw-checkpoint pilot"))
    warnings = [row for row in diagnostics if row["exceeds_reference"]]
    payload = {"status": "complete", "manifest_hash": manifest["record_hash"], "count": len(dates),
               "comparison_policy": POLICY if excluded else None, "excluded_models": excluded,
               "initialization_times": dates, "target_steps": 40, "summary": summary,
               "baseline_reference": "public GC branch from M1_raw; common denominator for all models",
               "baseline_repeat_spread_rollout": spread, "baseline_repeat_spread_day10": float(day10_spread),
               "numerical_reference_differences": diagnostics, "warnings": warnings,
               "noise_scope": "Observed public-GC repeat spread only, not full model repeatability; advisory.",
               "raw_vs_swa": pairs,
               "raw_vs_swa_csv": pinned(pair_path) if pairs else None,
               "raw_vs_swa_definition": "SWA loss reduction uses pooled raw loss as denominator; power-error changes are SWA minus raw in percentage points. Positive loss reduction and negative power-error change favor SWA. Public-GC repeat-spread flags are descriptive, not pairwise model-noise estimates or gates.",
               "power_error_definition": "100*abs(pooled forecast power / pooled ERA5 power - 1), equally averaged over retained channel/degree/lead bins; degrees 1–179.",
               "excluded_zero_truth_bins": int(mask.size - mask.sum()),
               "excluded_zero_truth_bins_per_lead": np.sum(~mask, axis=(1, 2)).tolist(),
               "limits": manifest.get("report_limits", "Small development pilot, not checkpoint selection or a claim about full-year skill. Raw lambda checkpoints at update 1250 are intermediate in planned 2500-update training; M1 update 2000 follows its 500-update warm-up. SWA is a separate historical reference. Full-GC lambda0 used peak LR 3e-7, whereas positive-lambda full-GC runs used 1e-5: this is not a fixed-LR lambda-only comparison. Finite worse metrics do not stop training/evaluation."),
               "figures": figures, "pooled_spectra": pinned(pooled_path), "sources": sources}
    if excluded:
        payload["archived_source_limits"] = payload["limits"]
        payload["limits"] = (
            "Filtered saved-checkpoint development diagnostic, not a full-year skill estimate or checkpoint selection. "
            "The rejected full-GC peak-LR1e-5 training lineages are excluded from every displayed table and figure. "
            "The remaining full-GC3e-7 lambda0 point is not a full-GC lambda sweep. "
            "Training recipes and checkpoint budgets differ; inspect the recorded checkpoint identities. "
            "The original source-protocol limitations remain in archived_source_limits in the JSON. "
            "Finite worse metrics and numerical drift remain advisory.")
    seal(data_root / "report.json", payload)
    lines = [f"# {manifest.get('report_title', 'Raw M1 and lambda checkpoint pilot: MSE and power spectra')}\n",
             f"Matched **{len(dates)} starts in 2022**, 40 six-hour leads, cold full feedback. "
             "MSE is the exact original GraphCast weighted grid objective; spectra are physical T179 powers.\n",
             "| Model | Rollout MSE | MSE reduction vs GC | Day-10 MSE | Power error | Day-10 power error |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['label']} | {row['exact_gc_mse_rollout']:.6f} | {row['mse_reduction_rollout_pct']:+.3f}% | "
                     f"{row['exact_gc_mse_day10']:.6f} | {row['power_error_rollout_pct']:.3f}% | {row['power_error_day10_pct']:.3f}% |")
    lines += ["", "Power error pools powers across dates first, then equally averages absolute relative "
              "errors over 83 channels, degrees 1–179 and the requested leads. Lower is better. "
              f"The common mask excludes {payload['excluded_zero_truth_bins']} zero-truth bins.", "",
              f"Repeated public-GC rollout losses differ by up to **{spread:.6g}** from the canonical reference "
              f"(day 10: {day10_spread:.6g}). These are numerical warnings, not gates. "
              "The CSV separately records whether each decrease exceeds that measured baseline spread; "
              "it does not estimate full model noise.", "", payload["limits"], ""]
    if excluded:
        lines += ["**Comparison policy (2026-11-12):** full-GC peak-LR `1e-5` runs are excluded by user instruction. "
                  "Their raw measurements remain in the archived source report for provenance/debugging only. "
                  "Eligible full-GC `3e-7` and GC–Mamba results remain displayed.", ""]
    if pairs:
        lines += ["## Matched raw versus SWA weights", "",
                  "The same color identifies each training run: solid curves and filled markers are raw "
                  "weights; dashed curves and open markers are SWA weights. These comparisons use unsmoothed "
                  "forecast outputs. SWA uses the declared saved checkpoints, without additional training "
                  "or selection using these evaluation scores.", "",
                  "| Training run | SWA rollout MSE reduction vs raw | SWA day-10 MSE reduction vs raw | "
                  "SWA − raw rollout power error | SWA − raw day-10 power error |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in pairs:
            rollout = row["swa_mse_reduction_vs_raw_rollout_pct"]
            day10 = row["swa_mse_reduction_vs_raw_day10_pct"]
            rollout_text = f"{rollout:+.3f}%" if rollout is not None else "undefined (raw loss zero)"
            day10_text = f"{day10:+.3f}%" if day10 is not None else "undefined (raw loss zero)"
            lines.append(f"| {row['comparison_pair']} | {rollout_text} | {day10_text} | "
                         f"{row['swa_minus_raw_power_error_rollout_pp']:+.3f} pp | "
                         f"{row['swa_minus_raw_power_error_day10_pp']:+.3f} pp |")
        lines += ["", "Positive MSE reductions and negative power-error changes favor SWA. "
                  "The paired CSV retains whether each MSE decrease exceeds the measured public-GC "
                  "repeat spread; this is advisory and does not estimate raw/SWA model repeatability.", ""]
    full_gc_models = [model for model in models if model["model_family"] == "full_gc"]
    zero_rates = {model.get("learning_rate") for model in full_gc_models if model.get("lambda") == 0}
    positive_rates = {model.get("learning_rate") for model in full_gc_models
                      if model.get("lambda") is not None and model["lambda"] > 0}
    if 3e-7 in zero_rates and 1e-5 in positive_rates:
        lines += ["**Learning-rate caveat:** full-GC λ=0 used peak LR `3e-7`; positive-λ full-GC runs used `1e-5`. "
                  "Their differences combine loss weight and learning rate.", ""]
    for figure in figures:
        if figure.endswith(".png"):
            relative = os.path.relpath(figure, data_root)
            lines += [f"![{Path(figure).stem.replace('_', ' ')}]({relative})", ""]
    lines += ["Checkpoint paths and hashes:", ""]
    lines.extend(f"- {model['label']}: `{model['checkpoint']['path']}`; SHA256 `{model['checkpoint']['sha256']}`."
                 for model in models)
    (data_root / "report.md").write_text("\n".join(lines) + "\n")
    return {"status": "complete", "count": len(dates), "models": len(models),
            "excluded_models": excluded,
            "report": str(data_root / "report.md"), "summary": str(data_root / "summary.csv"),
            "figures": figures, "numerical_warnings": len(warnings)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = report(args.root)
    print(json.dumps({"status": result["status"], "count": result["count"], "figures": result["figures"]}, indent=2))
