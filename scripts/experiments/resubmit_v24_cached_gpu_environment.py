#!/usr/bin/env python3
"""Replace the four cancelled CPU-contaminated cached submissions, preserving snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Remove before any imports; sbatch exports this sanitized environment to the
# unchanged frozen launcher, and its descendants inherit it in turn.
os.environ.pop("JAX_PLATFORMS", None)
from scripts.experiments import run_v24_cached_pair as pair
from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump

BASE = ROOT / "artifacts/checkpoints/v24_Ilya"
SWEEP = BASE / "res1_cached_w128_di_sweep_recovery_20261007"
TARGETS = (
    (BASE / "res1_cached_stepwise_pair_recovery_20261007", "15190847"),
    (SWEEP / "di32_carry", "15190867"),
    (SWEEP / "di64_carry", "15190868"),
    (SWEEP / "di16_reset_each_prediction", "15190869"),
)


def require_cancelled_without_start(job):
    result = subprocess.run(["sacct", "-j", job, "-X", "--noheader", "--parsable2",
                             "--format=JobIDRaw,State,Start"], check=True, capture_output=True, text=True)
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == job]
    if len(rows) != 1 or not rows[0][1].startswith("CANCELLED") or rows[0][2] not in ("Unknown", "None", ""):
        raise RuntimeError(f"Expected cancelled, never-started job {job}; accounting returned {rows!r}")


def main():
    commands = []
    # Verify all original jobs before changing any submission records.
    for root, job in TARGETS:
        require_cancelled_without_start(job)
        record = root / "submission_smoke.json"
        archive = root / f"submission_smoke.superseded_{job}_cpu_environment.json"
        if record.exists() and json.loads(record.read_text())["jobs"].get("smoke") != job and not archive.exists():
            raise RuntimeError(f"Unexpected active submission at {record}")
        if (root / ".submission_smoke_smoke.intent.json").exists():
            raise RuntimeError(f"Unresolved submission intent at {root}; reconcile before retry")

    def sanitized_submit(args):
        environment = os.environ.copy()
        environment.pop("JAX_PLATFORMS", None)
        command = ["sbatch", "--parsable", "--export=ALL", *args]
        print(shlex.join(command), flush=True)
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
        job = completed.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch response: {completed.stdout!r}")
        commands.append({"argv": command, "job_id": job, "JAX_PLATFORMS": "unset"})
        atomic_json_dump(commands, SWEEP / "gpu_environment_resubmission_commands.json")
        return job

    pair.submit_command = sanitized_submit
    sweep_jobs = {}
    summary = {}
    for root, old_job in TARGETS:
        record = root / "submission_smoke.json"
        archive = root / f"submission_smoke.superseded_{old_job}_cpu_environment.json"
        if not archive.exists():
            record.rename(archive)
        manifest = pair.load_experiment(root)
        pair.enable_automation(manifest)
        replacement = pair.submit_stage(manifest, "smoke")
        summary[root.name] = {"old_job": old_job, "jobs": replacement["jobs"], "snapshot_unchanged": True,
                              "source_digest": manifest["source_digest"], "reason": "remove inherited JAX_PLATFORMS=cpu"}
        atomic_json_dump(summary[root.name], root / "gpu_environment_recovery.json")
        if root.parent == SWEEP:
            sweep_jobs[root.name] = replacement
        atomic_json_dump(summary, SWEEP / "gpu_environment_resubmissions.json")
    jobs = SWEEP / "jobs.json"
    original = SWEEP / "jobs.superseded_cpu_environment.json"
    if jobs.exists() and not original.exists():
        jobs.rename(original)
    atomic_json_dump(sweep_jobs, jobs)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
