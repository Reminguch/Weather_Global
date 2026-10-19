"""Capacity comparisons and measured cost accounting for a completed matrix."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import numpy as np
from .io import read_json, write_json, sha256


def build_report(root, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    rows = []
    for run_id, entry in manifest["configs"].items():
        row = {"run_id": run_id, "resolution": entry["resolution"]}
        for stage in ("pretrain_cold", "pretrain_warm", "finetune_cold", "finetune_warm"):
            summary = read_json(root / "evaluation" / run_id / stage / "summary.json")
            if summary["failures"] or summary["completed_origins"] != 128:
                raise ValueError(f"Incomplete paired test evaluation: {run_id}/{stage}")
            row[stage] = summary
        row["finetune_vs_pretrain_mse_reduction_pct"] = 100 * (1 - row["finetune_cold"]["loss"] / row["pretrain_cold"]["loss"])
        rows.append(row)
    costs = {}
    for rid, resource in manifest["resources"].items():
        cache = Path(resource["cache_root"])
        shard_metadata = [read_json(cache / (s["id"] + ".json")) for s in read_json(cache / "manifest.json")["shards"]]
        costs[rid] = {"cache_storage_bytes": sum(s["stored_bytes"] for s in shard_metadata),
                      "cache_producer_seconds": sum(s["seconds"] for s in shard_metadata),
                      "cache_producer_seconds_per_arm": sum(s["seconds"] for s in shard_metadata) / 4,
                      "profile": read_json(root / "checks" / f"profile_{rid}.json")}
    submissions = [read_json(p) for p in (root / "submissions").glob("*.json")]
    ids = [s["job_id"] for s in submissions]
    accounting = {"job_ids": ids, "status": "no_submitted_jobs"}
    if ids:
        command = ["sacct", "-j", ",".join(ids), "-X", "-P", "--noheader",
                   "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,MaxRSS"]
        result = subprocess.run(command, capture_output=True, text=True)
        accounting = {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    report = {"schema": "neuralgcm_matrix_report_v1", "source_id": manifest["source_id"], "rows": rows,
              "shared_costs": costs, "slurm_accounting": accounting,
              "cost_note": "Measured cache producer time is shared equally across four arms; no inferred GPU-hour total."}
    out = root / "reports"
    write_json(out / "matrix.json", report, immutable=True)
    lines = ["# NeuralGCM residual matrix results", "", "Metric: NeuralGCM-field normalized MSE. Test year: 2023.", "",
             "| Run | Pretrain cold 5d reduction | Fine-tune cold 5d reduction | Fine-tune cold 10d reduction | Fine-tune warm 10d reduction |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["pretrain_cold"]["five_day_reduction_pct"], row["finetune_cold"]["five_day_reduction_pct"],
                  row["finetune_cold"]["reduction_pct"], row["finetune_warm"]["reduction_pct"]]
        lines.append("| " + row["run_id"] + " | " + " | ".join(f"{v:.3f}%" for v in values) + " |")
    lines += ["", "Warm forecasts use four observed branch-history frames. The baseline is initialized at the scored origin.",
              "Per-origin physical metrics, pressure/moisture diagnostics, paired calendar-week bootstrap intervals and spherical spectra are saved with each evaluation.",
              "Cache costs and measured profiles are in matrix.json. Slurm accounting is included without substituting estimates for unavailable measurements."]
    (out / "matrix.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, rid in zip(axes, ("res2p8", "res1p4"), strict=True):
        for row in rows:
            if row["resolution"] == rid:
                for stage, style in (("pretrain_cold", "--"), ("finetune_cold", "-")):
                    summary = row[stage]
                    # Plot aggregate per-lead MSE, never averaged percent reductions.
                    axis.plot(np.arange(1, 41) / 4, summary["lead_loss"], style, label=row["run_id"] + " " + stage)
        axis.set(title=rid, xlabel="Forecast lead (days)", ylabel="Normalized MSE")
        axis.legend(fontsize=6)
    fig.savefig(out / "lead_curves.png", dpi=180)
    fig.savefig(out / "lead_curves.pdf")
    plt.close(fig)
    return {"report": str(out / "matrix.md"), "matrix_sha256": sha256(out / "matrix.json"), "arms": len(rows)}
