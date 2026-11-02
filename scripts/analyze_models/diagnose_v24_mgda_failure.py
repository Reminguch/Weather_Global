#!/usr/bin/env python3
"""Summarize saved MGDA failure evidence without replaying or changing training.

Run through scripts/graphcast_env.sh. The CPU toy comparison distinguishes the
implemented stopped-feedback gradient from the derivative of a regenerated
trajectory; its magnitude is not an estimate for the production model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run):
    rows = read_rows(run / "train_metrics.jsonl")
    failure = read_json(run / "failed_update.json")
    values = lambda key: np.asarray([row[key] for row in rows], dtype=float)
    result = {
        "accepted_updates": len(rows),
        "failure": failure,
        "all_recorded_directions_valid": all(row["descent_valid"] for row in rows),
        "stalled_updates": sum(row["stalled"] for row in rows),
        "decreases_above_advisory_noise_margin": sum(row["accepted_decrease_above_noise"] for row in rows),
        "negative_gradient_cosine_updates": int(np.sum(values("gradient_cosine") < 0)),
        "negative_mse_direction_updates": int(np.sum(values("gc_displacement_dot") < 0)),
        "halvings_histogram": {str(int(h)): int(np.sum(values("backtracking_halvings") == h))
                              for h in np.unique(values("backtracking_halvings"))},
    }
    for key in ("mgda_alpha", "gradient_cosine", "gc_gradient_norm", "power_gradient_norm",
                "gc_displacement_dot", "power_displacement_dot", "update_norm",
                "repeatability_span_1", "repeatability_span_2"):
        v = values(key)
        result[key] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
    result["median_within_step_span_to_predicted_decrease_ratio"] = [
        float(np.median(values(span) / np.maximum(np.abs(values(dot)), 1e-30)))
        for span, dot in (("repeatability_span_1", "gc_displacement_dot"),
                          ("repeatability_span_2", "power_displacement_dot"))]
    if run.name == "mgda":
        expected_alpha = np.clip((values("metric_power_power") - values("metric_gc_power")) /
                                 values("metric_difference_squared"), 0, 1)
        result["alpha_formula_max_absolute_error"] = float(np.max(np.abs(expected_alpha - values("mgda_alpha"))))
    before = np.asarray(failure["acceptance"]["before_losses"])
    repeats = np.asarray(failure["repeatability"]["repeated_losses"])
    candidate = np.asarray([trial["losses"] for trial in failure["acceptance"]["trials"]])
    result["failure_analysis"] = {
        "all_candidates_finite": bool(np.isfinite(candidate).all()),
        "candidate_changes_from_mean": (candidate - before).tolist(),
        "nominal_change_from_first_reference": (candidate[0] - repeats[0]).tolist(),
        "nominal_change_from_mean_reference": (candidate[0] - before).tolist(),
        "repeat_span_percent": (100 * np.ptp(repeats, axis=0) / before).tolist(),
        "best_mse_increase": float((candidate - before)[:, 0].min()),
    }
    return result


def toy_feedback_comparison():
    """Compare actual explicit-tape gradients to stopped and full toy AD."""
    import sys
    import jax
    import jax.numpy as jnp
    from graphcast import xarray_jax

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tests/v24_Ilya"))
    from test_mgda_step import setup, reference

    cfg, _, _, params, state, args, step = setup()
    keys, initial_frames, _, truths, _ = args

    def full_reference(p):
        frames = [xarray_jax.unwrap_data(frame["x"])[:, 0] for frame in initial_frames]
        memory = state["s"]
        losses = []
        for i in range(4):
            mean = jnp.mean(jnp.stack((frames[i], frames[i + 1])), axis=(0, 2))
            memory = .7 * memory + p["a"] * mean + .01 * jax.random.normal(keys[i], mean.shape)
            forecast = .5 + p["b"] * memory[:, None] * jnp.ones((4, 2))
            truth = xarray_jax.unwrap_data(truths[i]["x"])[:, 0]
            losses.append(jnp.stack((jnp.mean((forecast - truth) ** 2),
                                     jnp.mean((forecast ** 2 - truth ** 2) ** 2))))
            if 1 <= i < 3:
                frames.append(forecast)
        return jnp.sum(jnp.stack(losses) * jnp.asarray(cfg.normalized_supervised_weights)[:, None], axis=0)

    gradients = step.objective_gradients(params, state, args)
    stopped = tuple(jax.grad(lambda p: reference(p, state, cfg, args)[0][i])(params) for i in range(2))
    full = tuple(jax.grad(lambda p: full_reference(p)[i])(params) for i in range(2))
    flatten = lambda trees: np.asarray([[float(tree[k]) for k in ("a", "b")] for tree in trees])
    np.testing.assert_allclose(flatten(gradients), flatten(stopped), rtol=4e-5, atol=3e-6)
    np.testing.assert_allclose(full_reference(params), reference(params, state, cfg, args)[0], rtol=2e-6)
    return {
        "scope": "four-step CPU toy from test_mgda_step; not the failed production batch",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "parameter_order": ["a", "b"],
        "implemented_gradient": flatten(gradients).tolist(),
        "stopped_feedback_AD": flatten(stopped).tolist(),
        "full_feedback_AD": flatten(full).tolist(),
        "relative_gradient_difference": (np.linalg.norm(flatten(gradients) - flatten(full), axis=1) /
                                         np.linalg.norm(flatten(full), axis=1)).tolist(),
        "forward_losses_unchanged": True,
        "stopped_reference_matches_implemented_gradient": True,
    }


def plot(arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [("mgda", 0, "MGDA: exact GC MSE"), ("mgda", 1, "MGDA: spectral adjustment"),
              ("mse_control", 0, "MSE control: exact GC MSE"), ("fixed_sum", None, "Fixed sum: MSE + adjustment")]
    for ax, (arm, component, title) in zip(axes.flat, panels):
        f = arms[arm]["failure"]
        repeats = np.asarray(f["repeatability"]["repeated_losses"])
        candidates = np.asarray([trial["losses"] for trial in f["acceptance"]["trials"]])
        if component is None:
            repeats, candidates = repeats.sum(axis=1), candidates.sum(axis=1)
        else:
            repeats, candidates = repeats[:, component], candidates[:, component]
        mean = repeats.mean()
        lo, hi = 100 * (repeats.min() - mean) / mean, 100 * (repeats.max() - mean) / mean
        ax.axhspan(lo, hi, color="#bdd7e7", alpha=.65, label="Unchanged model: range of 3 repeats")
        ax.plot(np.arange(len(candidates)), 100 * (candidates - mean) / mean,
                "o-", color="#9c382b", linewidth=1.6, markersize=4, label="Candidate minus reference mean")
        ax.axhline(0, color="0.25", linewidth=.8)
        ax.set(title=f"{title} — attempted update {f['attempted_step']}",
               xlabel="Number of LR halvings (0 = 1e−6; 8 = 3.9e−9)",
               ylabel="Loss change (%) — lower is better", xticks=np.arange(9))
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Failed line searches: shrinking the step did not give a smooth loss response\n"
                 "Shading is an observed repeat range, not a confidence interval.", fontsize=13)
    fig.savefig(output / "failed_line_search.png", dpi=170)
    fig.savefig(output / "failed_line_search.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = {name: summarize(args.experiment / "runs" / name) for name in ("mgda", "mse_control", "fixed_sum")}
    inputs = sorted((args.experiment / "runs").glob("*/*.json*")) + [args.experiment / "manifest.json"]
    evidence = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    result = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "experiment": str(args.experiment), "input_sha256": evidence,
              "arms": arms, "toy_gradient_comparison": toy_feedback_comparison(),
              "limits": ["Production failed-step gradients, parameters and intermediate tensors were not saved.",
                         "Three repeats do not identify the kernel responsible or estimate a confidence interval.",
                         "Accepted updates were selected by the old decrease rule; their losses are selection-biased.",
                         "No forecast skill or intervention benefit is inferred from this postmortem."]}
    (args.output / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(arms, args.output)
    print(json.dumps({"output": str(args.output), "toy": result["toy_gradient_comparison"],
                      "mgda_alpha_error": arms["mgda"]["alpha_formula_max_absolute_error"]}, indent=2))


if __name__ == "__main__":
    main()
