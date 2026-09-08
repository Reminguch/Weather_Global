#!/usr/bin/env python3
"""Summarize matched day-5/day-10 optimizer SWA results versus the incumbent."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUN_RE = re.compile(
    r"^(?P<init>legacy|mamba1)_di16_bcg1_lr(?P<lr>1em4|5em5)_"
    r"b1p(?P<b1>8|9)_b2p(?P<b2>98|999)_cos10k$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--eval-tag", default="swa2k7_matched_legacy32")
    parser.add_argument("--artifact-tag", default="swa_step02000-07000")
    return parser.parse_args()


def metrics(path: Path) -> tuple[dict, dict[str, float]]:
    payload = json.loads(path.read_text())
    loss = payload["original_graphcast_loss"]["improvement_pct_per_step"]
    t2m = payload["per_variable_per_step"]["2m_temperature"]["improvement_pct_rmse"]
    if payload["target_steps"] != 40 or payload["evaluated_samples"] != 32:
        raise ValueError(f"Expected complete 32-anchor x 40-lead evaluation: {path}")
    return payload, {
        "t2m_day5": float(t2m[19]),
        "t2m_day10": float(t2m[39]),
        "allvars_day5": float(loss[19]),
        "allvars_day10": float(loss[39]),
    }


def main() -> None:
    args = parse_args()
    reference_payload, reference = metrics(args.reference_json)
    rows = []
    for run_dir in sorted(path for path in args.experiment_root.iterdir() if path.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        path = (
            run_dir
            / "eval/cold_full_zero_exact_gc"
            / args.eval_tag
            / f"{args.artifact_tag}.json"
        )
        if not path.is_file():
            continue
        payload, values = metrics(path)
        if payload["chosen_idx"] != reference_payload["chosen_idx"]:
            raise ValueError(f"Anchor mismatch: {path}")
        fields = match.groupdict()
        row = {
            "run": run_dir.name,
            "init": fields["init"],
            "lr": {"1em4": "1e-4", "5em5": "5e-5"}[fields["lr"]],
            "beta1": {"8": "0.8", "9": "0.9"}[fields["b1"]],
            "beta2": {"98": "0.98", "999": "0.999"}[fields["b2"]],
            **values,
        }
        row.update({f"delta_vs_reference_{key}": value - reference[key] for key, value in values.items()})
        rows.append(row)
    if len(rows) != 10:
        raise SystemExit(f"Expected 10 optimizer evaluations, found {len(rows)}")

    metric_keys = ("t2m_day5", "t2m_day10", "allvars_day5", "allvars_day10")
    rows.sort(key=lambda row: (-row["allvars_day10"], -row["t2m_day10"], row["run"]))
    for rank, row in enumerate(rows, start=1):
        row["rank_by_allvars_day10"] = rank

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        columns = [
            "rank_by_allvars_day10", "run", "init", "lr", "beta1", "beta2",
            *metric_keys, *(f"delta_vs_reference_{key}" for key in metric_keys),
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        f"{row['init']} lr={row['lr']} b=({row['beta1']},{row['beta2']})"
        for row in rows
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    specs = (
        ("t2m_day5", "2m temperature RMSE improvement, day 5"),
        ("t2m_day10", "2m temperature RMSE improvement, day 10"),
        ("allvars_day5", "Exact allvars loss improvement, day 5"),
        ("allvars_day10", "Exact allvars loss improvement, day 10"),
    )
    x = range(len(rows))
    colors = ["#4c78a8" if row["init"] == "legacy" else "#f58518" for row in rows]
    for ax, (key, title) in zip(axes.flat, specs, strict=True):
        ax.bar(x, [row[key] for row in rows], color=colors)
        ax.axhline(reference[key], color="black", linestyle="--", linewidth=1.5, label="legacy 2k-8k incumbent")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Improvement (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    for ax in axes[-1]:
        ax.set_xticks(list(x), labels, rotation=55, ha="right", fontsize=8)
    fig.suptitle("v24 res1 optimizer SWA(2k-7k), exact matched 32-anchor evaluation")
    fig.tight_layout()
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_image, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_image}")
    print(json.dumps({"reference": reference, "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
