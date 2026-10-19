"""Repair definite GPU-partition submission rejections without editing a snapshot.

Only successful, finished release parents may advance. Scientific failures,
ambiguous submission intents, existing queued jobs and scientific settings are
never retried, cancelled or changed. Submitted jobs still run the old snapshot.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.experiments.frozen_runtime import verify_source

RELEASE_STAGES = {"prepare_verify", "cache_release", "cache_verify", "training_release"}
POLICY_REJECTION = "This is not allowed. Please remove"


def load_driver(root):
    """Load only scheduler functions; restore import search paths immediately.

    Existing package imports belong to this bridge's verified frozen runtime.
    No scientific producer function is executed here. Absolute submitted wrapper
    paths continue selecting the producer's independently verified old runtime.
    """
    root = Path(root).resolve()
    source = root / "source"
    verify_source(source)
    path = source / "scripts/experiments/run_v24_cached_40yr.py"
    spec = importlib.util.spec_from_file_location("_fresh_scale_original_producer", path)
    driver = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path[:] = before
    campaign = driver.load_campaign(root)
    if Path(campaign["source"]).resolve() != source:
        raise ValueError("Producer source differs from its immutable campaign directory")
    return driver, campaign


def corrected_sbatch_args(args, job):
    """Drop the rejected explicit GPU partition; retain all scientific arguments."""
    corrected = list(args)
    if not job.get("constraint"):
        return corrected
    if corrected.count("--partition=gpu") != 1 or "--gres=gpu:1" not in corrected:
        raise ValueError("Unexpected original GPU submission command")
    corrected.remove("--partition=gpu")
    minutes = job["time_minutes"]
    if minutes > 8640:
        raise ValueError("GPU request exceeds the verified six-day limit")
    qos = "gpu-long" if minutes > 4320 else "gpu-medium" if minutes > 1440 else None
    if qos:
        if any(arg.startswith("--qos=") for arg in corrected):
            raise ValueError("Original command has an unexpected GPU QoS override")
        corrected.insert(1, f"--qos={qos}")
    return corrected


def rejection_evidence(driver, campaign, entry):
    """Tie a definite policy rejection to this exact successful parent attempt."""
    ended = datetime.fromisoformat(entry["ended_at"])
    wrapper = str(Path(campaign["source"]) / "scripts/experiments/v24_cached_40yr.slurm")
    dependency = f"--dependency=afterok:{entry['job_id']}"
    matches = []
    for path in (Path(campaign["root"]) / "intents").glob("*.rejected.json"):
        record = driver.read(path)
        args = record.get("command", [])
        if (POLICY_REJECTION not in record.get("error", "") or "--partition=gpu" not in args
                or dependency not in args or "--gres=gpu:1" not in args
                or len(args) < 4 or args[-4] != wrapper or args[-3] != campaign["root"]
                or args[-1] != campaign["workspace"]):
            continue
        if datetime.fromisoformat(record["created_at"]) < ended:
            continue
        matches.append(dict(path=str(path), sha256=driver.sha(path), **record))
    if not matches:
        raise RuntimeError("Release failure is not a definite current GPU-partition policy rejection")
    return matches


def inactive_completed(job_id):
    command = ["squeue", "-h", "-j", job_id, "-o", "%T"]
    query = subprocess.run(command, check=False, capture_output=True, text=True)
    expired = {"slurm_load_jobs error: Invalid job id specified",
               "squeue: error: slurm_load_jobs error: Invalid job id specified"}
    if query.returncode and query.stderr.strip() not in expired:
        raise subprocess.CalledProcessError(query.returncode, command, output=query.stdout, stderr=query.stderr)
    if query.stdout.strip():
        return False
    accounting = subprocess.run(["sacct", "-X", "-j", job_id, "-n", "-P", "--format=State"],
                                check=True, capture_output=True, text=True).stdout
    states = {row.split("|")[0].split()[0] for row in accounting.splitlines() if row.strip()}
    if not states:
        return False
    if states != {"COMPLETED"}:
        raise RuntimeError(f"Release parent did not finish successfully: {job_id}: {states}")
    return True


def repair_release(driver, campaign, key):
    root = Path(campaign["root"])
    with driver.locked(root, ".resume.lock"):
        # Revalidate immutable producer source/configs and re-read state under the
        # same recovery lock used by the original explicit resume operation.
        campaign = driver.load_campaign(root)
        state = driver.read(root / "state.json")
        entry = state["jobs"][key]
        if entry.get("status") != "release_failed":
            return None
        stage = entry["spec"]["stage"]
        if stage not in RELEASE_STAGES:
            raise RuntimeError(f"Unsupported scheduler release failure: {key}")
        evidence = rejection_evidence(driver, campaign, entry)
        result = driver.require_report(campaign, key)
        if stage == "cache_verify":
            # The historical verifier predates a 'passed' field. Its complete
            # READY/manifest and signed report are checked by its own reader.
            driver.require_cache(campaign)
            if result.get("partial") is not False:
                raise ValueError("Historical cache verification was not complete")
        elif result.get("passed") is not True:
            raise ValueError("Release parent lacks a passing scientific report")
        if not inactive_completed(entry["job_id"]):
            return None
        commands = []
        original = driver.sbatch_args
        def corrected(c, k, job, dependencies=(), condition="afterok", attempt=1):
            before = original(c, k, job, dependencies, condition, attempt)
            after = corrected_sbatch_args(before, job)
            commands.append(dict(key=k, before=before, after=after))
            return after
        driver.sbatch_args = corrected
        try:
            # Never call resume/_resume: their scientific retry/cancellation
            # branch is deliberately unavailable to this compatibility bridge.
            driver.advance(campaign, key, entry["spec"], result)
        finally:
            driver.sbatch_args = original
        proof = dict(passed=True, workflow_id=campaign["workflow_id"], key=key,
                     parent_job_id=entry["job_id"], rejection_evidence=evidence,
                     commands=commands, repaired_at=driver.now(),
                     producer_source_digest=campaign["source_digest"],
                     bridge_sha256=driver.sha(Path(__file__)))
        proof["sha256"] = driver.digest(proof)
        driver.write(root / "reports" / f"scheduler_compat_{key}_{entry.get('attempt', 1)}.json", proof)
        driver.mark_job(campaign, key, expected_job_id=entry["job_id"], status="complete",
                        scheduler_compatibility_repair=proof["sha256"], error=None)
        return proof


def run(producer_root, *, timeout_hours=143, poll_seconds=60):
    if not 0 < timeout_hours <= 143 or not 0 < poll_seconds <= 60:
        raise ValueError("Expected a positive wait no longer than 143 hours and polls up to 60 seconds")
    driver, campaign = load_driver(producer_root)
    manifest_sha = driver.sha(Path(campaign["root"]) / "campaign.json")
    deadline = time.monotonic() + timeout_hours * 3600
    repaired = []
    while True:
        if driver.sha(Path(campaign["root"]) / "campaign.json") != manifest_sha:
            raise ValueError("Producer campaign changed during compatibility monitoring")
        state = driver.read(Path(campaign["root"]) / "state.json")
        if state.get("workflow_id") != campaign["workflow_id"] or not state.get("automation_enabled"):
            raise ValueError("Producer identity/automation changed")
        # Once the final report is accepted, the entire GPU submission chain has
        # been released; its scientific jobs can finish without this monitor.
        report = state["jobs"].get("report")
        if report and str(report.get("job_id", "")).isdigit():
            return dict(passed=True, repaired=repaired, report_job_id=report["job_id"],
                        producer_workflow_id=campaign["workflow_id"],
                        meaning="All original producer jobs submitted; scientific completion remains separate")
        for key, entry in state["jobs"].items():
            if entry.get("status") == "failed" and entry["spec"]["stage"] not in ("cache_pilot", "training_pilot"):
                raise RuntimeError(f"Producer scientific stage failed; bridge will not retry: {key}")
            if entry.get("status") == "release_failed":
                proof = repair_release(driver, campaign, key)
                if proof is not None:
                    repaired.append(dict(key=key, parent_job_id=proof["parent_job_id"], sha256=proof["sha256"]))
        if time.monotonic() >= deadline:
            raise TimeoutError("Producer scheduler compatibility bridge exhausted its wait budget")
        print(json.dumps(dict(waiting_for="original producer GPU releases", repaired=repaired, at=driver.now())), flush=True)
        time.sleep(poll_seconds)
