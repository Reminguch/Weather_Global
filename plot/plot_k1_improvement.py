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


STEM = "k1_improvement_vs_training_step"
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
    """Pair completed validations with their checkpoint's optimizer update count."""
    manifest = json.loads((root / "manifest.json").read_text())
    provenance = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root.resolve()),
        "source_id": manifest["source_id"],
        "improvement_definition": "100 * (1 - residual_ngcm_loss / ngcm_loss)",
        "time_definition": "sum(metrics.jsonl seconds through checkpoint update) / 3600",
        "plotted_x_axis": "checkpoint optimizer update count (training step)",
        "time_scope": "update wall time including data loading and gradient/update work; "
                      "excludes queueing, validation, checkpoint writes, and other startup work",
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
                       if p.parent.name.isdigit() and (p.parent / "COMPLETE.json").is_file())
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
            receipt_path = path.parent / "COMPLETE.json"
            receipt_bytes = receipt_path.read_bytes()
            if json.loads(receipt_bytes)["checkpoint_sha256"] != metadata["sha256"]:
                raise ValueError(f"Validation receipt and checkpoint differ: {path}")
            cursor = metadata["cursor"]
            if (metadata["stage"] != "pretrain" or cursor["epoch"] != epoch
                    or cursor["chunk"] != 0 or cursor["segment"] != 0):
                raise ValueError(f"Inconsistent checkpoint cursor: {checkpoint}")
            if not math.isfinite(score["loss"]) or score["loss"] <= 0:
                raise ValueError(f"Invalid validation loss: {path}")
            identities = {key: metadata["identities"][key] for key in
                          ("backbone", "cache", "dataset", "normalization", "source",
                           "loss", "correction_policy")}
            if (identities["loss"] != config["loss"]
                    or identities["correction_policy"] != config["correction_policy"]):
                raise ValueError(f"Checkpoint objective differs from config: {checkpoint}")
            if common_identities is None:
                common_identities = identities
            if identities != common_identities or identities["source"] != manifest["source_id"]:
                raise ValueError("Checkpoints must use the same data, baseline, and loss scales")
            points.append({
                "epoch": epoch, "update": cursor["update"], "score": score,
                "validation_path": str(path.relative_to(root)),
                "validation_sha256": sha256(score_bytes),
                "completion_receipt_path": str(receipt_path.relative_to(root)),
                "completion_receipt_sha256": sha256(receipt_bytes),
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
        # Keep long-rollout results beside the K=1 figure so its scope is explicit.
        ar_points = [p for p in points
                     if (stage / "validation" / f"{p['epoch']:06d}" / "cold" / "summary.json").is_file()]
        if ar_points:
            latest = ar_points[-1]
            context = {"epoch": latest["epoch"], "update": latest["update"]}
            for mode in ("cold", "warm"):
                path = stage / "validation" / f"{latest['epoch']:06d}" / mode / "summary.json"
                report_bytes = path.read_bytes()
                report = json.loads(report_bytes)
                if report["loss_name"] != config["loss"]:
                    raise ValueError(f"Rollout validation objective differs: {path}")
                context[mode] = {
                    "path": str(path.relative_to(root)), "sha256": sha256(report_bytes),
                    **{key: report[key] for key in (
                        "five_day_loss", "baseline_loss", "five_day_reduction_pct",
                        "completed_origins", "expected_origins", "failures",
                    )},
                }
            provenance["runs"][run_id]["latest_rollout_validation"] = context
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
        from matplotlib.ticker import MaxNLocator, MultipleLocator

        with (output / f"{STEM}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        styles = (
            ("#0072B2", "o", "-"), ("#D55E00", "s", "--"),
            ("#009E73", "^", "-"), ("#9B59A1", "D", "--"),
        )
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
            "mathtext.fontset": "dejavuserif", "axes.labelsize": 10,
            "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
            "svg.fonttype": "none", "svg.hashsalt": "ngcm-k1-training-step",
        })
        fig, ax = plt.subplots(figsize=(4.8, 3.35))
        fig.subplots_adjust(left=0.145, right=0.975, bottom=0.16, top=0.975)
        detail = ax.inset_axes([0.47, 0.15, 0.49, 0.42])
        if len({int(r["validation_records"]) for r in rows}) != 1:
            raise ValueError("Mixed validation sample counts")
        xmax, ymax, ymin = 0, 0.0, 0.0
        detail_x, detail_y = [], []
        for run_id, (color, marker, linestyle) in zip(RUNS, styles):
            points = sorted((r for r in rows if r["run_id"] == run_id), key=lambda r: int(r["epoch"]))
            if not points:
                raise ValueError(f"Missing series: {run_id}")
            x = [int(r["update"]) for r in points]
            y = [float(r["improvement_pct"]) for r in points]
            for row, value in zip(points, y):
                expected = 100 * (1 - float(row["residual_ngcm_loss"]) / float(row["ngcm_loss"]))
                if not math.isclose(value, expected, abs_tol=1e-10):
                    raise ValueError("CSV improvement is inconsistent with losses")
                if not math.isclose(float(row["training_hours"]),
                                    float(row["training_seconds"]) / 3600, abs_tol=1e-10):
                    raise ValueError("CSV training hours are inconsistent with seconds")
            if x[0] != 0 or y[0] != 0 or any(b <= a for a, b in zip(x, x[1:])):
                raise ValueError("Expected zero initial improvement and increasing training steps")
            width, inner = run_id.split("_")[1:]
            label = rf"$w={width[1:]},\ d_{{\mathrm{{inner}}}}={inner[2:]}$"
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.4, markersize=4, markeredgecolor="white", markeredgewidth=0.4)
            detail.plot(x[1:], y[1:], color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.1, markersize=3.4, markeredgecolor="white", markeredgewidth=0.35)
            detail_x.extend(x[1:])
            detail_y.extend(y[1:])
            xmax, ymax, ymin = max(xmax, max(x)), max(ymax, max(y)), min(ymin, min(y))
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlabel="Training step", ylabel="Improvement over NGCM (%)",
               xlim=(-35, max(500, xmax * 1.035)), ylim=(min(-1, ymin - 1), max(90, ymax + 2)))
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.7, direction="out")
        ax.legend(loc="upper left", bbox_to_anchor=(0.26, 0.84), frameon=False, fontsize=8,
                  handlelength=2.3, labelspacing=0.35, borderaxespad=0.35)
        if detail_x:
            detail.set(xlim=(min(detail_x) - 50, max(detail_x) + 50),
                       ylim=(min(detail_y) - 0.3, max(detail_y) + 0.3))
            detail.xaxis.set_major_locator(MaxNLocator(nbins=4))
            detail.yaxis.set_major_locator(MaxNLocator(nbins=4))
            detail.tick_params(labelsize=7, length=2, width=0.5)
            detail.grid(color="#E4E4E4", linewidth=0.4)
            detail.spines[["top", "right"]].set_visible(False)
        else:
            detail.set_visible(False)
        for extension in ("png", "pdf", "svg"):
            metadata = {"CreationDate": None} if extension == "pdf" else None
            if extension == "svg":
                metadata = {"Date": None}
            path = output / f"{STEM}.{extension}"
            fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    for run_id in RUNS:
        last = max((r for r in rows if r["run_id"] == run_id), key=lambda r: int(r["epoch"]))
        print(f"{run_id}: epoch {last['epoch']}, training step {last['update']}, "
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
