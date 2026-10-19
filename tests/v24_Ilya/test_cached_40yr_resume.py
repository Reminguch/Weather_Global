"""Explicit campaign recovery must replace only blocked dependent jobs."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v24_cached_40yr as campaign


class Scheduler:
    """In-memory Slurm boundary: no command leaves this test process."""
    def __init__(self):
        self.next_id = 1000
        self.queued = {}
        self.submitted = []
        self.cancelled = []
        self.accounting = {}
        self.failed_cancellations = set()

    def run(self, args, **unused):
        if args[0] == "sbatch":
            self.next_id += 1
            job = str(self.next_id)
            self.submitted.append((job, args))
            self.queued[job] = ("PENDING", "Dependency")
            return SimpleNamespace(stdout=job + "\n", returncode=0)
        if args[0] == "squeue":
            entry = self.queued.get(args[args.index("-j") + 1])
            output = "" if entry is None else ("|".join(entry) if "%r" in args[-1] else entry[0]) + "\n"
            return SimpleNamespace(stdout=output, returncode=0)
        if args[0] == "scancel":
            self.cancelled.append(args[1])
            if args[1] in self.failed_cancellations:
                return SimpleNamespace(stdout="", stderr="Cancellation failed", returncode=1)
            self.queued.pop(args[1], None)
            return SimpleNamespace(stdout="", returncode=0)
        if args[0] == "sacct":
            job = args[args.index("-j") + 1]
            output = self.accounting.get(job, "")
            return SimpleNamespace(stdout=output, returncode=0)
        pytest.fail(f"Unexpected scheduler command: {args}")


@pytest.fixture
def flow(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    for name in ("reports", "intents", "logs"):
        (root / name).mkdir()
    value = {"root": str(root), "workspace": str(tmp_path), "source": str(root / "source"),
             "workflow_id": "test-recovery"}
    campaign.write(root / "state.json", {"workflow_id": value["workflow_id"],
                   "automation_enabled": True, "jobs": {}})
    scheduler = Scheduler()
    monkeypatch.setattr(campaign.subprocess, "run", scheduler.run)
    return value, scheduler


def jobs(value):
    return campaign.read(Path(value["root"]) / "state.json")["jobs"]


def finish(value, scheduler, key, status="complete", result=None):
    entry = jobs(value)[key]
    scheduler.queued.pop(entry["job_id"], None)
    scheduler.accounting[entry["job_id"]] = "FAILED|\n" if status == "failed" else "COMPLETED|\n"
    campaign.mark_job(value, key, status=status)
    if result is not None:
        campaign.save_report(value, key, result)


@pytest.mark.parametrize("stage,key,verifier", [
    ("prepare_year", "prepare_years", "prepare_verify"),
    ("cache_generate", "cache_generate", "cache_verify"),
])
def test_failed_array_replaces_pending_verifier_only(flow, stage, key, verifier):
    value, scheduler = flow
    failed = campaign.submit_job(value, key, campaign.spec(stage, array=41), dependencies=["99"])
    pending = campaign.submit_job(value, verifier, campaign.spec(verifier), dependencies=[failed])
    unrelated = campaign.submit_job(value, "unrelated", campaign.spec("report"), dependencies=["98"])
    scheduler.queued[unrelated] = ("RUNNING", "None")
    finish(value, scheduler, key, "failed")
    original_unrelated = jobs(value)["unrelated"]

    result = campaign.resume(value, key)

    current = jobs(value)
    assert result[key] != failed
    assert current[key]["attempt"] == 2
    assert current[key]["history"][0]["job_id"] == failed
    assert current[verifier]["dependencies"] == [result[key]]
    assert current[verifier]["attempt"] == 2
    assert current["unrelated"] == original_unrelated
    assert scheduler.cancelled == [pending]
    retired = campaign.read(Path(value["root"]) / "state.json")["retired_jobs"]
    assert [row["key"] for row in retired] == [verifier]


@pytest.mark.parametrize("family,width", [("cache", None), ("training", 128)])
def test_failed_pilot_retires_collectors_release_and_reuses_other_gpu(flow, family, width):
    value, scheduler = flow
    campaign.launch_pilots(value, family=family, dependency="99", width=width)
    prefix = family if width is None else f"{family}_w{width}"
    full, full_collect, release = prefix + "_full", prefix + "_full_collect", prefix + "_release"
    mig, mig_collect = prefix + "_mig", prefix + "_mig_collect"
    for key in (mig, mig_collect):
        finish(value, scheduler, key, result={"passed": True})
    # WorkerFailure is deliberately collected as a normal pilot return, so the
    # enclosing Slurm job completes despite a failed scientific qualification.
    finish(value, scheduler, full, "complete", {"passed": False, "failure_kind": "resource"})
    finish(value, scheduler, full_collect, result={"passed": False, "failure_kind": "resource"})
    finish(value, scheduler, release, "failed", {"passed": False})
    before = jobs(value)
    calls = len(scheduler.submitted)

    result = campaign.resume(value, full)

    current = jobs(value)
    assert current[full]["job_id"] == result[full]
    assert current[full_collect]["dependencies"] == [result[full]]
    assert current[full_collect]["condition"] == "afterany"
    assert current[release]["dependencies"] == [current[full_collect]["job_id"], before[mig_collect]["job_id"]]
    assert current[release]["condition"] == "afterok"
    assert current[mig] == before[mig] and current[mig_collect] == before[mig_collect]
    assert len(scheduler.submitted) - calls == 3
    assert set(scheduler.cancelled) == {before[full_collect]["job_id"], before[release]["job_id"]}
    assert not campaign.report_path(value, full).exists()
    assert campaign.report_path(value, full).with_name(full + ".attempt1.retired.json").exists()
    assert campaign.require_report(value, mig_collect)["passed"]


@pytest.mark.parametrize("family,width", [("cache", None), ("training", 128)])
def test_failed_collector_retry_rebuilds_release_without_rerunning_pilots(flow, family, width):
    value, scheduler = flow
    campaign.launch_pilots(value, family=family, dependency="99", width=width)
    prefix = family if width is None else f"{family}_w{width}"
    collector, release = prefix + "_full_collect", prefix + "_release"
    for key in (prefix + "_full", prefix + "_mig", prefix + "_mig_collect"):
        finish(value, scheduler, key, result={"passed": True})
    finish(value, scheduler, collector, "failed", {"passed": False})
    before = jobs(value)
    calls = len(scheduler.submitted)

    result = campaign.resume(value, collector)

    current = jobs(value)
    assert current[collector]["job_id"] == result[collector]
    assert current[collector]["dependencies"] == [before[prefix + "_full"]["job_id"]]
    assert current[release]["dependencies"] == [result[collector], before[prefix + "_mig_collect"]["job_id"]]
    assert len(scheduler.submitted) - calls == 2
    assert scheduler.cancelled == [before[release]["job_id"]]
    for key in (prefix + "_full", prefix + "_mig", prefix + "_mig_collect"):
        assert current[key] == before[key]


def training_branch(value, scheduler, width):
    key = f"training_w{width}_release"
    job = campaign.spec("training_release", width=width)
    campaign.submit_job(value, key, job)
    result = {"passed": True, "budget": {"memory_gib": 64, "constraint": "gpu40"},
              "chunks": [{"start_step": start, "stop_step": start + 250, "time_minutes": 120}
                         for start in (0, 250, 500)]}
    finish(value, scheduler, key, result=result)
    campaign.advance(value, key, job, result)


def test_failed_training_chunk_rewires_its_tail_preserving_other_width(flow):
    value, scheduler = flow
    training_branch(value, scheduler, 128)
    training_branch(value, scheduler, 512)
    finish(value, scheduler, "train_w128_00250")
    finish(value, scheduler, "train_w128_00500", "failed", {"passed": False})
    for key in ("train_w512_00250", "train_w512_00500"):
        finish(value, scheduler, key)
    scheduler.queued[jobs(value)["train_w512_00750"]["job_id"]] = ("RUNNING", "None")
    before = jobs(value)
    calls = len(scheduler.submitted)

    result = campaign.resume(value, "train_w128_00500")

    current = jobs(value)
    assert current["train_w128_00500"]["dependencies"] == [before["train_w128_00250"]["job_id"]]
    assert current["train_w128_00750"]["dependencies"] == [result["train_w128_00500"]]
    assert current["evaluate_w128"]["dependencies"] == [current["train_w128_00750"]["job_id"]]
    assert current["report"]["dependencies"] == [current["evaluate_w128"]["job_id"], before["evaluate_w512"]["job_id"]]
    assert current["train_w128_00250"] == before["train_w128_00250"]
    for key in before:
        if "w512" in key:
            assert current[key] == before[key]
    assert len(scheduler.submitted) - calls == 4
    assert set(scheduler.cancelled) == {before[key]["job_id"] for key in
        ("train_w128_00750", "evaluate_w128", "report")}


@pytest.mark.parametrize("state,reason", [("RUNNING", "None"), ("PENDING", "Resources")])
def test_resume_refuses_active_descendants_without_cancelling_or_submitting(flow, state, reason):
    value, scheduler = flow
    failed = campaign.submit_job(value, "prepare_years", campaign.spec("prepare_year", array=41))
    child = campaign.submit_job(value, "prepare_verify", campaign.spec("prepare_verify"), dependencies=[failed])
    finish(value, scheduler, "prepare_years", "failed")
    scheduler.queued[child] = (state, reason)
    before = jobs(value)
    with pytest.raises(RuntimeError, match="active"):
        campaign.resume(value, "prepare_years")
    assert jobs(value) == before
    assert not scheduler.cancelled
    assert len(scheduler.submitted) == 2


def test_resume_refuses_overwriting_completed_scientific_descendant(flow):
    value, scheduler = flow
    failed = campaign.submit_job(value, "prepare_years", campaign.spec("prepare_year", array=41))
    campaign.submit_job(value, "prepare_verify", campaign.spec("prepare_verify"), dependencies=[failed])
    finish(value, scheduler, "prepare_years", "failed")
    finish(value, scheduler, "prepare_verify", result={"passed": True})
    with pytest.raises(RuntimeError, match="scientific output"):
        campaign.resume(value, "prepare_years")
    assert not scheduler.cancelled and len(scheduler.submitted) == 2


def test_cancellation_failure_keeps_descendant_records_for_next_retry(flow):
    value, scheduler = flow
    failed = campaign.submit_job(value, "prepare_years", campaign.spec("prepare_year", array=41))
    child = campaign.submit_job(value, "prepare_verify", campaign.spec("prepare_verify"), dependencies=[failed])
    finish(value, scheduler, "prepare_years", "failed")
    scheduler.failed_cancellations.add(child)
    before = jobs(value)
    with pytest.raises(RuntimeError, match="has not stopped"):
        campaign.resume(value, "prepare_years")
    assert jobs(value) == before and len(scheduler.submitted) == 2
    assert scheduler.cancelled == [child]


def test_release_failure_advances_only_after_completed_accounting(flow):
    value, scheduler = flow
    original = campaign.submit_job(value, "download_gate", campaign.spec("download_gate"))
    finish(value, scheduler, "download_gate", "release_failed", {"passed": True})
    assert campaign.resume(value, "download_gate") == {"advanced": "download_gate"}
    current = jobs(value)
    assert current["download_gate"]["job_id"] == original and current["download_gate"]["attempt"] == 1
    assert current["download_gate"]["status"] == "complete"
    assert current["prepare_pilot"]["dependencies"] == [original]


def test_success_report_with_failed_accounting_requires_new_job(flow):
    value, scheduler = flow
    original = campaign.submit_job(value, "download_gate", campaign.spec("download_gate"))
    finish(value, scheduler, "download_gate", "complete", {"passed": True})
    scheduler.accounting[original] = "FAILED|\n"
    result = campaign.resume(value, "download_gate")
    assert result["download_gate"] != original
    assert "prepare_pilot" not in jobs(value)
    assert jobs(value)["download_gate"]["attempt"] == 2


def test_missing_final_accounting_never_releases_or_duplicates_work(flow):
    value, scheduler = flow
    original = campaign.submit_job(value, "download_gate", campaign.spec("download_gate"))
    finish(value, scheduler, "download_gate", "complete", {"passed": True})
    scheduler.accounting[original] = ""
    before = jobs(value)
    with pytest.raises(RuntimeError, match="accounting"):
        campaign.resume(value, "download_gate")
    assert jobs(value) == before and len(scheduler.submitted) == 1


@pytest.mark.parametrize("array_index", [None, 3])
def test_retired_attempt_cannot_publish_state_or_array_reports(flow, array_index):
    value, scheduler = flow
    key = "prepare_years"
    job = campaign.spec("prepare_year", array=41)
    old = campaign.submit_job(value, key, job)
    finish(value, scheduler, key, "failed")
    new = campaign.submit_job(value, key, job, retry=True)
    report_key = key if array_index is None else f"{key}_{array_index:03d}"
    campaign.save_report(value, report_key, {"passed": True, "attempt": 2},
                         job_key=key, expected_job_id=new)
    before = jobs(value)
    with pytest.raises(RuntimeError, match="Retired job attempt"):
        campaign.save_report(value, report_key, {"passed": False, "attempt": 1},
                             job_key=key, expected_job_id=old)
    with pytest.raises(RuntimeError, match="Retired job attempt"):
        campaign.mark_job(value, key, status="failed", expected_job_id=old)
    assert jobs(value) == before
    assert campaign.require_report(value, report_key) == {"passed": True, "attempt": 2}


def test_new_concurrent_descendant_aborts_recovery_before_retiring_records(flow, monkeypatch):
    value, scheduler = flow
    failed = campaign.submit_job(value, "prepare_years", campaign.spec("prepare_year", array=41))
    child = campaign.submit_job(value, "prepare_verify", campaign.spec("prepare_verify"), dependencies=[failed])
    finish(value, scheduler, "prepare_years", "failed")
    original_run = scheduler.run
    added = False

    def concurrent_submission(args, **kwargs):
        nonlocal added
        result = original_run(args, **kwargs)
        if args[0] == "scancel" and not added:
            added = True
            campaign.submit_job(value, "late_descendant", campaign.spec("report"), dependencies=[child])
        return result

    monkeypatch.setattr(campaign.subprocess, "run", concurrent_submission)
    with pytest.raises(RuntimeError):
        campaign.resume(value, "prepare_years")
    current = jobs(value)
    assert current["prepare_years"]["job_id"] == failed
    assert current["prepare_verify"]["job_id"] == child
    assert current["late_descendant"]["dependencies"] == [child]
    assert len(scheduler.submitted) == 3
