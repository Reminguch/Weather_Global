#!/usr/bin/env python3
"""Snapshot existing K=1 validation results, or redraw the committed snapshot."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


STEM = "k1_improvement_vs_training_time"
RUNS = (
    "r2p8_w128_di16", "r2p8_w128_di32",
    "r2p8_w256_di16", "r2p8_w256_di32",
)
COLUMNS = (
    "run_id", "epoch", "update", "training_seconds", "training_hours",
    "validation_records", "ngcm_loss", "residual_ngcm_loss", "improvement_pct",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root, output):
    """Pair each validation result with its committed checkpoint's update count."""
    manifest = json.loads((root / "manifest.json").read_text())
    provenance = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root.resolve()),
        "source_id": manifest["source_id"],
        "improvement_definition": "100 * (1 - residual_ngcm_loss / ngcm_loss)",
        "time_definition": "sum(metrics.jsonl seconds through checkpoint update) / 3600",
        "baseline_definition": "epoch 0, exactly zero residual output head",
        "runs": {},
    }
    rows = []
    common_identities = None
    common_baseline = None
    for run_id in RUNS:
        stage = root / "runs" / run_id / "seed22" / "pretrain"
        # Ignore archived interrupted validation directories.
        paths = sorted(p for p in (stage / "validation").glob("*/one_step.json")
                       if p.parent.name.isdigit())
        if not paths or int(paths[0].parent.name) != 0:
            raise ValueError(f"Missing initial validation for {run_id}")
        config = json.loads((root / "configs" / f"{run_id}.json").read_text())
        if config["architecture"]["initialization"] != "fresh_zero_head":
            raise ValueError("Initial validation cannot be assumed to equal NGCM")
        points = []
        for path in paths:
            epoch = int(path.parent.name)
            checkpoint = stage / ("checkpoint_initial.json" if epoch == 0
                                  else f"checkpoint_pass_{epoch:02d}.json")
            score_bytes, checkpoint_bytes = path.read_bytes(), checkpoint.read_bytes()
            score, metadata = json.loads(score_bytes), json.loads(checkpoint_bytes)
            cursor = metadata["cursor"]
            if (metadata["stage"] != "pretrain" or cursor["epoch"] != epoch
                    or cursor["chunk"] != 0 or cursor["segment"] != 0):
                raise ValueError(f"Inconsistent checkpoint cursor: {checkpoint}")
            if not math.isfinite(score["loss"]) or score["loss"] <= 0:
                raise ValueError(f"Invalid validation loss: {path}")
            identities = {key: metadata["identities"][key] for key in
                          ("backbone", "cache", "dataset", "normalization", "source")}
            if common_identities is None:
                common_identities = identities
            if identities != common_identities or identities["source"] != manifest["source_id"]:
                raise ValueError("Checkpoints must use the same data, baseline, and loss scales")
            points.append({
                "epoch": epoch, "update": cursor["update"], "score": score,
                "validation_path": str(path.relative_to(root)),
                "validation_sha256": sha256(score_bytes),
                "checkpoint_metadata_path": str(checkpoint.relative_to(root)),
                "checkpoint_metadata_sha256": sha256(checkpoint_bytes),
                "checkpoint_sha256": metadata["sha256"],
            })
        baseline = points[0]["score"]
        if points[0]["update"] != 0:
            raise ValueError("Baseline must precede all training updates")
        if common_baseline is None:
            common_baseline = baseline
        if baseline != common_baseline:
            raise ValueError("Runs do not share an identical NGCM baseline")
        max_update = points[-1]["update"]
        cumulative = [0.0]
        metrics_hash = hashlib.sha256()
        # Read only the validated prefix, even if training is still appending.
        with (stage / "metrics.jsonl").open("rb") as stream:
            for expected in range(1, max_update + 1):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError(f"Missing complete update {expected} in {run_id}")
                record = json.loads(line)
                seconds = float(record["seconds"])
                if record["update"] != expected or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"Invalid timing or update order in {run_id}")
                metrics_hash.update(line)
                cumulative.append(cumulative[-1] + seconds)
        for point in points:
            score, update = point["score"], point["update"]
            if score["records"] != baseline["records"]:
                raise ValueError("Validation sample count changed")
            rows.append(dict(zip(COLUMNS, (
                run_id, point["epoch"], update, cumulative[update], cumulative[update] / 3600,
                score["records"], baseline["loss"], score["loss"],
                100 * (1 - score["loss"] / baseline["loss"]),
            ))))
        provenance["runs"][run_id] = {
            "metrics_path": str((stage / "metrics.jsonl").relative_to(root)),
            "metrics_prefix_updates": max_update,
            "metrics_prefix_sha256": metrics_hash.hexdigest(),
            "validation_points": points,
        }
    provenance["shared_identities"] = common_identities
    with (output / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{STEM}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def draw(output):
    # Avoid creating a font cache outside the writable workspace.
    with tempfile.TemporaryDirectory(prefix="ngcm-plot-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, PercentFormatter

        with (output / f"{STEM}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        styles = (
            ("#0072B2", "o", "-", -14),
            ("#D55E00", "s", "--", -6),
            ("#009E73", "^", "-", -14),
            ("#9B59A1", "D", "--", 8),
        )
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.labelsize": 12, "axes.edgecolor": "#AAB2BC",
            "text.color": "#202A35", "axes.labelcolor": "#202A35",
            "xtick.color": "#46515D", "ytick.color": "#46515D",
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "svg.hashsalt": "ngcm-k1-training-time",
        })
        fig, ax = plt.subplots(figsize=(9, 5.8))
        fig.subplots_adjust(left=0.105, right=0.96, bottom=0.205, top=0.82)
        fig.text(0.105, 0.935, "Residual NGCM over NGCM", fontsize=18, weight="bold")
        records = {int(r["validation_records"]) for r in rows}
        if len(records) != 1:
            raise ValueError("Mixed validation sample counts")
        fig.text(0.105, 0.885,
                 f"K = 1 (6-hour) validation  |  2.8° resolution  |  {records.pop():,} samples from 2022",
                 fontsize=10.5, color="#5B6571")
        xmax, ymax, ymin = 0.0, 0.0, 0.0
        for run_id, (color, marker, linestyle, offset) in zip(RUNS, styles):
            points = sorted((r for r in rows if r["run_id"] == run_id), key=lambda r: int(r["epoch"]))
            if not points:
                raise ValueError(f"Missing series: {run_id}")
            x = [float(r["training_hours"]) for r in points]
            y = [float(r["improvement_pct"]) for r in points]
            for row, value in zip(points, y):
                expected = 100 * (1 - float(row["residual_ngcm_loss"]) / float(row["ngcm_loss"]))
                if not math.isclose(value, expected, abs_tol=1e-10):
                    raise ValueError("CSV improvement is inconsistent with losses")
            if x[0] != 0 or y[0] != 0 or any(b <= a for a, b in zip(x, x[1:])):
                raise ValueError("Expected zero initial improvement and increasing training times")
            width, inner = run_id.split("_")[1:]
            label = f"w = {width[1:]}, di = {inner[2:]}"
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=2.2, markersize=5.8, markeredgecolor="white", markeredgewidth=0.7)
            ax.annotate(f"{y[-1]:.1f}%", xy=(x[-1], y[-1]), xytext=(9, offset),
                        textcoords="offset points", color=color, weight="bold", fontsize=10.5,
                        va="center")
            xmax, ymax, ymin = max(xmax, max(x)), max(ymax, max(y)), min(ymin, min(y))
        ax.axhline(0, color="#8C97A3", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlabel="Training time (hours)", ylabel="Improvement over NGCM (%)",
               xlim=(-0.04, xmax * 1.14), ylim=(min(-0.7, ymin - 1), max(30, ymax + 4)))
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E7EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3.5)
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10,
                  handlelength=2.8, columnspacing=1.8)
        fig.text(0.105, 0.09,
                 "Improvement = 100 × (1 − residual NGCM loss / NGCM loss)", fontsize=10,
                 color="#46515D")
        fig.text(0.105, 0.052,
                 "Cumulative logged training time; validation and queue time excluded. Dots mark completed epochs.",
                 fontsize=9, color="#65717E")
        for extension in ("png", "pdf", "svg"):
            metadata = {"CreationDate": None} if extension == "pdf" else None
            if extension == "svg":
                metadata = {"Date": None}
            path = output / f"{STEM}.{extension}"
            fig.savefig(path, dpi=240, facecolor="white", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    for run_id in RUNS:
        last = max((r for r in rows if r["run_id"] == run_id), key=lambda r: int(r["epoch"]))
        print(f"{run_id}: epoch {last['epoch']}, {float(last['training_hours']):.3f} h, "
              f"{float(last['improvement_pct']):.2f}% improvement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path,
                        help="Refresh CSV and provenance from this experiment before plotting")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment_root:
        snapshot(args.experiment_root, args.output_dir)
    draw(args.output_dir)


if __name__ == "__main__":
    main()
