"""Bounded campaign orchestration and diagnostic numerical regression tests.

No test submits Slurm work or loads production model weights. Workflow tests
replace subprocess execution with small, provenance-bearing local artifacts.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments import run_v24_amse_regime as campaign
from src.models.mamba.v24_Ilya.training.config import load_training_config


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924/configs/mse_fp32.json"
PILOT_DATES = [f"2023-{month:02d}-15T00:00:00Z" for month in (1, 2, 4, 5, 7, 8, 10, 11)]


@pytest.fixture
def template():
    return campaign.read(TEMPLATE)


def file(path, content="fixture"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_confirmation_dates_are_frozen_seasonal_and_disjoint():
    first = campaign.confirmation_dates(PILOT_DATES)
    assert first == campaign.confirmation_dates(list(reversed(PILOT_DATES)))
    assert first == sorted(set(first)) and len(first) == 64
    dates = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in first]
    exploratory = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in PILOT_DATES]
    assert all(value.year == 2023 and value.hour == 0 for value in dates)
    assert all(value.date() >= datetime(2023, 1, 2).date() for value in dates)
    assert all((value + timedelta(days=10)).date() <= datetime(2023, 12, 31).date() for value in dates)
    assert all(abs((value - old).days) > 10 for value in dates for old in exploratory)
    # The evaluator stratifies uncertainty by DJF/MAM/JJA/SON.
    seasons = [(value.month % 12) // 3 for value in dates]
    assert [seasons.count(index) for index in range(4)] == [16] * 4


def test_all_pilot_and_feedback_configs_parse_and_match_controls(template, tmp_path):
    before = copy.deepcopy(template)
    parsed = {}
    for name, amse_lambda in campaign.PILOT_ARMS:
        config = campaign.make_config(template, tmp_path, name, amse_lambda=amse_lambda)
        path = tmp_path / f"{name}.json"
        campaign.write(path, config)
        parsed[name] = load_training_config(path)
        assert parsed[name].max_steps == 500
        assert parsed[name].checkpoint_every == 100
        assert parsed[name].architecture == load_training_config(TEMPLATE).architecture
        assert parsed[name].distributed.per_device_batch_size == 4
        assert parsed[name].seed == 22
        assert parsed[name].feedback_mode == "closed_loop_sg"
        assert parsed[name].loss_mode == "all_steps"
        assert campaign.evaluation_task({"name": name, "steps": 500, "run_dir": str(tmp_path / name),
                                         "config": {"path": str(path)}})["label"] == campaign.LABELS[name]
    mse = parsed["mse"].to_dict()
    for name, config in parsed.items():
        for key in ("architecture", "data", "optimizer", "sequence", "distributed"):
            assert config.to_dict()[key] == mse[key], (name, key)
    assert parsed["spectral_mse"].loss_kind == "amse"
    assert parsed["spectral_mse"].amse_lambda == 0
    assert parsed["spectral_mse"].amse_lambda_ramp_steps == 0
    for steps in (2, 4, 8):
        config = campaign.make_config(template, tmp_path, f"feedback{steps}", amse_lambda=.5,
                                      steps=8, bptt=steps, feedback="closed_loop", ramp=4)
        path = tmp_path / f"feedback{steps}.json"
        campaign.write(path, config)
        actual = load_training_config(path)
        assert actual.bptt_backend == "explicit_weather_reverse_vjp"
        assert actual.ar_tail_k == steps - 1
        assert actual.distributed.per_device_batch_size == 4
    assert template == before


def test_prepare_freezes_selection_configs_dates_source_and_evaluation_inputs(template, tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    workspace = tmp_path / "workspace"
    experiment = workspace / "artifacts/campaign"
    file(workspace / "src/example.py", "VALUE = 1\n")
    file(workspace / "scripts/example.py", "VALUE = 2\n")
    file(workspace / "third_party/graphcast/graphcast/graphcast.py", "VALUE = 3\n")
    historical = [workspace / "artifacts/checkpoints/v24_Ilya" / name for name in campaign.HISTORICAL]
    campaign.write(historical[1] / "configs/mse_fp32.json", template)
    campaign.write(workspace / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json",
                    {"initialization_times": PILOT_DATES})
    candidates = []
    for ci, directory in enumerate(historical):
        for ti, tag in enumerate(("step2000", "swa_step00500-02000")):
            path = file(directory / "runs/mse_fp32" / f"{tag}.pkl", f"weights-{ci}-{ti}")
            evidence = file(directory / "eval/exact/mse_fp32" / tag / "evaluation.json", "{}")
            spec = file(evidence.parent / "specification.json", "{}")
            run_config = file(directory / "runs/mse_fp32/run_config.json", "{}")
            candidates.append({"experiment": str(directory), "tag": tag,
                "checkpoint": str(path), "checkpoint_sha256": campaign.file_sha256(path),
                "source_run_dir": str(run_config.parent), "exact32_rollout_loss": 4. - ci - .1 * ti,
                "evaluation": campaign.pinned(evidence), "specification": campaign.pinned(spec),
                "run_config": campaign.pinned(run_config)})
    selected = min(candidates, key=lambda row: row["exact32_rollout_loss"])
    def select(experiments, out):
        assert list(experiments) == historical
        selection = {"candidates": candidates, "selected": selected}
        campaign.write(out, selection)
        return selection
    monkeypatch.setattr(evaluator, "select_initialization", select)
    prepared = []
    def call(args):
        values = list(map(str, args))
        assert "prepare" in values
        output = Path(values[values.index("--out-dir") + 1])
        tasks = campaign.read(values[values.index("--tasks-json") + 1])
        campaign.write(output / "manifest.json", {"task_specs": tasks})
        prepared.append((output, tasks))
    monkeypatch.setattr(campaign, "call", call)
    monkeypatch.setattr(evaluator, "load_manifest", lambda path: campaign.read(path))
    campaign.prepare(experiment, workspace)
    manifest = campaign.verify(experiment)
    assert manifest["initialization"]["checkpoint"] == {
        "path": selected["checkpoint"], "sha256": selected["checkpoint_sha256"]}
    assert len(manifest["pilot_entries"]) == 4 and len(manifest["smoke_entries"]) == 7
    assert len(prepared) == 2
    assert len(campaign.read(experiment / "confirmation_dates.json")["initialization_times"]) == 64
    assert all(entry["init_from"] == selected["checkpoint"] for entry in manifest["pilot_entries"] + manifest["smoke_entries"])
    assert manifest["policy"]["confirmation_seeds"] == [22, 23, 24]
    assert manifest["policy"]["confirmation_recipes"] == 1
    with pytest.raises(ValueError, match="snapshot"):
        campaign.verify(experiment, require_snapshot=True)
    with pytest.raises(FileExistsError):
        campaign.prepare(experiment, workspace)
    # Both policy mutation and changes to pinned development evidence stop work.
    unchanged = (experiment / "manifest.json").read_text()
    modified = copy.deepcopy(manifest)
    modified["policy"]["confirmation_updates"] = 100000
    campaign.write(experiment / "manifest.json", modified)
    with pytest.raises(ValueError, match="manifest|Manifest|changed"):
        campaign.verify(experiment)
    (experiment / "manifest.json").write_text(unchanged)
    file(candidates[0]["evaluation"]["path"], "changed evidence")
    with pytest.raises(ValueError, match="Pinned file changed"):
        campaign.verify(experiment)


def test_measured_resource_budget_is_bounded_and_uses_slowest_report():
    manifest = {"resources": {"historical_peak_rss_gib": 180., "historical_seconds_per_update": 20.}}
    reports = [{"peak_rss_gib": 200., "steady_p90_seconds": 25.},
               {"peak_rss_gib": 250., "steady_p90_seconds": 32.}]
    memory, minutes = campaign.resource_budget(manifest, reports, 500)
    assert memory == 352
    assert minutes >= (1200 + 32 * 500) / 60 * 1.35
    assert memory <= 512 and minutes <= 6 * 24 * 60
    with pytest.raises(ValueError, match="bounds|resource|memory"):
        campaign.resource_budget(manifest, [{"peak_rss_gib": 1000., "steady_p90_seconds": 32.}], 500)
    with pytest.raises(ValueError, match="bounds|resource|time"):
        campaign.resource_budget(manifest, [{"peak_rss_gib": 250., "steady_p90_seconds": 5000.}], 500)


def test_sbatch_uses_frozen_entrypoint_explicit_resources_and_dependency(tmp_path):
    manifest = {"workspace": str(tmp_path / "work"), "source_root": str(tmp_path / "source")}
    args = campaign.sbatch(tmp_path, manifest, "train", memory=352, minutes=420, gpu=True,
                           array="0-3", dependency="12345", phase="pilot")
    assert "--mem=352G" in args and "--time=420" in args
    assert "--gres=gpu:1" in args and "--constraint=gpu80" in args
    assert "--dependency=afterok:12345" in args
    assert "--array=0-3" in args and not any("%" in value for value in args if value.startswith("--array="))
    assert str(tmp_path / "source/scripts/experiments/v24_amse_regime.slurm") in args
    assert args[-3:] == ["train", str(tmp_path), "pilot"]


@pytest.mark.parametrize("options", [{"memory": 513}, {"minutes": 8641}, {"memory": 0}, {"minutes": 0}])
def test_sbatch_rejects_unbounded_or_invalid_requests(tmp_path, options):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source")}
    with pytest.raises(ValueError):
        campaign.sbatch(tmp_path, manifest, "train", gpu=True, **options)


def test_prepare_evaluation_enables_spherical_observer_and_confirmation_role(tmp_path, monkeypatch):
    campaign.write(tmp_path / "manifest.json", {"reference_experiment": str(tmp_path / "reference"),
                                                "workspace": str(tmp_path)})
    dates = file(tmp_path / "dates.json", json.dumps({"initialization_times": campaign.confirmation_dates(PILOT_DATES)}))
    commands = []
    monkeypatch.setattr(campaign, "call", lambda args: commands.append(list(map(str, args))))
    result = campaign.prepare_eval(tmp_path, "confirmation", [{"id": "mature_mse"}], dates=dates)
    command = commands[-1]
    assert "--observer-factory" in command
    assert command[command.index("--observer-factory") + 1] == "src.models.mamba.v24_Ilya.spherical_observer:make_observer"
    assert command[command.index("--role") + 1] == "confirmation"
    assert command[command.index("--workspace") + 1] == str(tmp_path)
    assert command[command.index("--spectral-initializations") + 1] == str(dates)
    assert result == tmp_path / "confirmation/evaluation/manifest.json"


def test_submission_is_idempotent_and_unknown_intent_never_retries(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_amse_pair as predecessor
    calls = []
    def submit(args, **kwargs):
        calls.append(list(args))
        return SimpleNamespace(stdout="123456;cluster\n", stderr="")
    monkeypatch.setattr(predecessor.subprocess, "run", submit)
    args = ["sbatch", "--parsable", "frozen.slurm"]
    assert campaign.submit_job(tmp_path, "pilot_train", args) == "123456"
    assert campaign.submit_job(tmp_path, "pilot_train", args) == "123456"
    assert len(calls) == 1
    with pytest.raises(ValueError):
        campaign.submit_job(tmp_path, "pilot_train", args + ["different"])
    campaign.write(tmp_path / "submissions/unknown.json", {"command": args})
    with pytest.raises(RuntimeError, match="intent|unresolved|Unresolved"):
        campaign.submit_job(tmp_path, "unknown", args)
    assert len(calls) == 1


def test_concurrent_submission_has_one_scheduler_call(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_amse_pair as predecessor
    entered, finish = threading.Event(), threading.Event()
    calls = []
    def blocked_submit(args, **kwargs):
        calls.append(list(args))
        entered.set()
        assert finish.wait(timeout=5), "Test failed to release mocked scheduler"
        return SimpleNamespace(stdout="456789\n", stderr="")
    monkeypatch.setattr(predecessor.subprocess, "run", blocked_submit)
    args = ["sbatch", "--parsable", "frozen.slurm"]
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(campaign.submit_job, tmp_path, "pilot_train", args)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises((BlockingIOError, RuntimeError)):
                campaign.submit_job(tmp_path, "pilot_train", args)
        finally:
            finish.set()
        assert running.result(timeout=5) == "456789"
    assert len(calls) == 1
    assert campaign.submit_job(tmp_path, "pilot_train", args) == "456789"


def test_initial_submission_and_release_respect_smoke_and_evaluation_dependencies(template, tmp_path, monkeypatch):
    initial = file(tmp_path / "initial.pkl")
    entries = []
    for name, value in campaign.PILOT_ARMS:
        cfg = campaign.make_config(template, tmp_path, "smoke_" + name, amse_lambda=value, steps=8, ramp=4)
        entry = campaign.config_record(tmp_path, "smoke_" + name, cfg, initial)
        entries.append(entry)
    for steps in (2, 4, 8):
        name = f"smoke_feedback_b{steps}"
        cfg = campaign.make_config(template, tmp_path, name, amse_lambda=.5, steps=8, bptt=steps,
                                    feedback="closed_loop", ramp=4)
        entries.append(campaign.config_record(tmp_path, name, cfg, initial))
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"),
                "source_digest": "frozen", "smoke_entries": entries,
                "initialization": {"checkpoint": campaign.pinned(initial)},
                "resources": {"historical_peak_rss_gib": 180., "historical_seconds_per_update": 20.}}
    required_snapshot = []
    def verify(experiment, require_snapshot=False):
        required_snapshot.append(require_snapshot)
        return manifest
    monkeypatch.setattr(campaign, "verify", verify)
    jobs = []
    def submit(experiment, name, args):
        job_id = str(1000 + len(jobs))
        jobs.append((name, list(args), job_id))
        return job_id
    monkeypatch.setattr(campaign, "submit_job", submit)
    for phase in ("pilot", "trajectory"):
        campaign.write(tmp_path / phase / "evaluation/manifest.json", {"task_specs": [{"id": "mse"}, {"id": "amse"}]})
    campaign.submit(tmp_path)
    by_name = {name: (args, job) for name, args, job in jobs}
    assert "--array=0-3" in by_name["smoke"][0]
    assert "--array=4-6" in by_name["feedback_smoke"][0]
    assert f"--dependency=afterok:{by_name['smoke'][1]}" in by_name["pilot_release"][0]
    assert f"--dependency=afterok:{by_name['trajectory_eval'][1]}" in by_name["trajectory_report"][0]
    assert "pilot_train" not in by_name
    with pytest.raises(FileNotFoundError):
        campaign.release(tmp_path, "pilot")
    assert "pilot_train" not in {row[0] for row in jobs}
    for entry in entries:
        file(entry["checkpoint"])
        campaign.write(tmp_path / "checks" / f"{entry['name']}_complete.json", {
            "passed": True, "source_digest": "frozen", "config": entry["config"],
            "checkpoint": campaign.pinned(entry["checkpoint"]), "initial_checkpoint": campaign.pinned(initial),
            "peak_rss_gib": 200., "steady_p90_seconds": 25.})
    campaign.release(tmp_path, "pilot")
    by_name = {name: (args, job) for name, args, job in jobs}
    assert "--array=0-3" in by_name["pilot_train"][0]
    assert f"--dependency=afterok:{by_name['pilot_train'][1]}" in by_name["pilot_eval"][0]
    assert f"--dependency=afterok:{by_name['pilot_eval'][1]}" in by_name["pilot_report"][0]
    assert required_snapshot == [False, True, True]
    # A successful checkpoint from another smoke task is not interchangeable.
    first = tmp_path / "checks" / f"{entries[0]['name']}_complete.json"
    campaign.write(first, campaign.read(tmp_path / "checks" / f"{entries[1]['name']}_complete.json"))
    before = len(jobs)
    with pytest.raises(ValueError, match="different task|provenance"):
        campaign.release(tmp_path, "pilot")
    assert len(jobs) == before


def selected_recipe(template, tmp_path, *, eligible=True, coherence=0., feedback="closed_loop_sg", bptt=24):
    checkpoint = file(tmp_path / "selected.pkl")
    report = file(tmp_path / "selected_report.json", "{}")
    config = campaign.make_config(template, tmp_path, "selected", amse_lambda=.5, bptt=bptt,
                                   feedback=feedback, ramp=0)
    return {"id": "selected_step00500", "checkpoint": str(checkpoint), "phase": "pilot",
            "config_value": config, "report_evidence": campaign.pinned(report),
            "score": {"eligible": eligible, "mean_coherence_change_vs_mse": coherence}}


def phase_manifest(template, tmp_path):
    initial = file(tmp_path / "mature.pkl")
    dates = file(tmp_path / "dates.json", json.dumps({"initialization_times": campaign.confirmation_dates(PILOT_DATES)}))
    return {"template": template, "source_digest": "frozen", "workspace": str(tmp_path),
            "source_root": str(tmp_path / "source"),
            "initialization": {"checkpoint": campaign.pinned(initial), "source_run_dir": str(tmp_path / "mature"),
                               "reuse": {"exact": "saved_exact", "spectral": "saved_spectral"}},
            "confirmation_dates": campaign.pinned(dates),
            "smoke_entries": [{"name": f"smoke{index}"} for index in range(7)],
            "policy": {"maximum_phases": ["pilot", "feedback", "adaptive", "confirmation"],
                       "confirmation_seeds": [22, 23, 24], "confirmation_updates": 2000}}


def phase_report(experiment, phase, role):
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest
    evaluation = file(experiment / phase / "evaluation/manifest.json", "{}")
    report = experiment / phase / "evaluation/report/joint_selection.json"
    campaign.write(report, {"status": "complete", "evaluation_role": role,
                            "manifest": campaign.pinned(evaluation)})
    plan = {"phase": phase, "entries": [], "evidence": [], "source_digest": "frozen"}
    plan["plan_hash"] = protocol_digest(plan)
    campaign.write(experiment / phase / "plan.json", plan)
    return report


def test_feedback_and_adaptive_plans_are_matched_bounded_and_do_not_average_forecasts(template, tmp_path, monkeypatch):
    manifest = phase_manifest(template, tmp_path)
    selected = selected_recipe(template, tmp_path, eligible=False)
    launched = []
    monkeypatch.setattr(campaign, "launch_phase", lambda *args, **kwargs: launched.append((args, kwargs)))
    campaign.prepare_feedback(tmp_path, manifest, selected)
    args, _ = launched[-1]
    assert args[2] == "feedback"
    entries, tasks = args[3:5]
    assert len(entries) == 2 and len(tasks) == 3
    for index, entry in enumerate(entries):
        previous = selected["checkpoint"]
        for length, stage in zip((2, 4, 8), entry["stages"], strict=True):
            config = load_training_config(Path(stage["config"]["path"]))
            assert config.bptt_steps == length and config.ar_tail_k == length - 1
            assert config.max_steps == 100 and config.amse_lambda == .5
            assert config.distributed.per_device_batch_size == 4
            assert config.feedback_mode == ("closed_loop_sg" if index == 0 else "closed_loop")
            assert stage["init_from"] == previous
            previous = stage["checkpoint"]
    assert entries[0]["stages"][0]["init_from"] == entries[1]["stages"][0]["init_from"]
    for stopped, differentiable in zip(entries[0]["stages"], entries[1]["stages"], strict=True):
        left, right = (campaign.read(entry["config"]["path"]) for entry in (stopped, differentiable))
        for key in ("architecture", "data", "optimizer", "objective", "distributed"):
            assert left[key] == right[key]
    policy = file(tmp_path / "degree_policy_fixture.json", "{}")
    monkeypatch.setattr(campaign, "degree_policy", lambda *args: ([0.] + [.3] * 179, campaign.pinned(policy)))
    campaign.prepare_adaptive(tmp_path, manifest, selected)
    args, _ = launched[-1]
    assert args[2] == "adaptive"
    entries, tasks = args[3:5]
    assert len(entries) == 3 and len(tasks) == 10
    configs = {entry["name"]: load_training_config(Path(entry["config"]["path"])) for entry in entries}
    assert all(config.max_steps == 500 and config.amse_lambda == .5 for config in configs.values())
    assert all(entry["init_from"] == selected["checkpoint"] for entry in entries)
    assert configs["adaptive_precip"].amse_variable_lambda_multipliers["total_precipitation_6hr"] == 0.
    assert len(configs["adaptive_degrees"].amse_degree_lambda_multipliers) == 180


def test_confirmation_is_fixed_recipe_three_matched_seeds_and_final_only(template, tmp_path, monkeypatch):
    manifest = phase_manifest(template, tmp_path)
    selected = selected_recipe(template, tmp_path, feedback="closed_loop", bptt=8)
    selected["config_value"]["objective"]["amse_variable_lambda_multipliers"] = {"total_precipitation_6hr": 0.}
    launched = []
    monkeypatch.setattr(campaign, "launch_phase", lambda *args, **kwargs: launched.append((args, kwargs)))
    campaign.prepare_confirmation(tmp_path, manifest, selected)
    args, kwargs = launched[0]
    assert args[2] == "confirmation" and kwargs["dates"] == manifest["confirmation_dates"]["path"]
    entries, tasks, _, decision = args[3:7]
    assert len(entries) == 6 and len(tasks) == 7
    assert "reuse" not in tasks[0]
    assert decision["selection_locked_before_confirmation"] is True
    assert all(task["step"] == 2000 for task in tasks)
    assert {task["seed"] for task in tasks[1:]} == {22, 23, 24}
    for entry in entries:
        assert [stage["steps"] for stage in entry["stages"]] == [100, 100, 2000]
        assert entry["stages"][0]["init_from"] == manifest["initialization"]["checkpoint"]["path"]
    for seed in (22, 23, 24):
        rows = [task for task in tasks[1:] if task["seed"] == seed]
        assert len(rows) == 2
        mse = next(task for task in rows if task["recipe_id"] == "confirmation_mse")
        amse = next(task for task in rows if task["recipe_id"] == "confirmation_amse")
        assert mse["paired_mse_id"] == amse["paired_mse_id"] == mse["id"]
        left = campaign.read(mse["planned_config"])
        right = campaign.read(amse["planned_config"])
        for key in ("architecture", "data", "optimizer", "sequence", "distributed"):
            assert left[key] == right[key]
        assert left["objective"]["loss_kind"] == "mse_fp32"
        assert right["objective"]["amse_variable_lambda_multipliers"] == {"total_precipitation_6hr": 0.}


@pytest.mark.parametrize("phase,eligible,coherence,expected", [
    ("pilot", False, 0., "feedback"), ("pilot", True, -.002, "feedback"),
    ("pilot", True, -.0005, "adaptive"), ("feedback", False, -.01, "adaptive"),
    ("adaptive", True, 0., "confirmation"),
])
def test_advance_uses_development_only_and_has_bounded_stages(template, tmp_path, monkeypatch,
                                                          phase, eligible, coherence, expected):
    manifest = phase_manifest(template, tmp_path)
    selected = selected_recipe(template, tmp_path, eligible=eligible, coherence=coherence)
    monkeypatch.setattr(campaign, "verify", lambda *args, **kwargs: manifest)
    seen = []
    def candidates(experiment, phases):
        assert "confirmation" not in phases
        return [selected]
    monkeypatch.setattr(campaign, "amse_candidates", candidates)
    for name in ("feedback", "adaptive", "confirmation"):
        monkeypatch.setattr(campaign, "prepare_" + name,
                            lambda *args, target=name: seen.append(target))
    for entry in manifest["smoke_entries"]:
        file(tmp_path / "checks" / f"{entry['name']}_complete.json", "{}")
    report = phase_report(tmp_path, phase, "development")
    campaign.advance(tmp_path, phase)
    assert seen == [expected]
    campaign.write(report, {**campaign.read(report), "evaluation_role": "confirmation"})
    with pytest.raises(ValueError, match="Confirmation results"):
        campaign.advance(tmp_path, phase)
    assert seen == [expected]


def test_confirmation_advance_finishes_without_selecting_or_training(template, tmp_path, monkeypatch):
    manifest = phase_manifest(template, tmp_path)
    monkeypatch.setattr(campaign, "verify", lambda *args, **kwargs: manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("Confirmation must not launch or select new work")
    monkeypatch.setattr(campaign, "amse_candidates", forbidden)
    monkeypatch.setattr(campaign, "submit_job", forbidden)
    path = phase_report(tmp_path, "confirmation", "confirmation")
    campaign.advance(tmp_path, "confirmation")
    status = campaign.read(tmp_path / "status.json")
    assert status["status"] == "complete"
    assert status["confirmation_report"] == campaign.pinned(path)


def test_advance_waits_for_feedback_preflight_only_when_feedback_is_needed(template, tmp_path, monkeypatch):
    manifest = phase_manifest(template, tmp_path)
    selected = selected_recipe(template, tmp_path, eligible=False)
    phase_report(tmp_path, "pilot", "development")
    campaign.write(tmp_path / "submissions/feedback_smoke.json", {"job_id": "78910"})
    monkeypatch.setattr(campaign, "verify", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(campaign, "amse_candidates", lambda *args: [selected])
    submitted, prepared = [], []
    monkeypatch.setattr(campaign, "submit_job", lambda *args: submitted.append(args) or "89012")
    monkeypatch.setattr(campaign, "prepare_feedback", lambda *args: prepared.append("feedback"))
    monkeypatch.setattr(campaign, "prepare_adaptive", lambda *args: prepared.append("adaptive"))
    campaign.advance(tmp_path, "pilot")
    assert not prepared
    assert submitted[0][1] == "feedback_gate"
    assert "--dependency=afterok:78910" in submitted[0][2]
    assert submitted[0][2][-3:] == ["advance", str(tmp_path), "pilot"]
    assert campaign.read(tmp_path / "status.json")["status"] == "waiting_for_feedback_preflight"
    # An eligible result bypasses feedback even while its smoke array is pending.
    selected["score"]["eligible"] = True
    campaign.advance(tmp_path, "pilot")
    assert prepared == ["adaptive"] and len(submitted) == 1
    # Failed qualification requires all three feedback-smoke proofs first.
    selected["score"]["eligible"] = False
    for entry in manifest["smoke_entries"][4:]:
        file(tmp_path / "checks" / f"{entry['name']}_complete.json", "{}")
    campaign.advance(tmp_path, "pilot")
    assert prepared == ["adaptive", "feedback"] and len(submitted) == 1


def test_continuation_plan_pins_evidence_and_requires_matching_preflight(template, tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    manifest = phase_manifest(template, tmp_path)
    manifest["resources"] = {"historical_peak_rss_gib": 180., "historical_seconds_per_update": 20.}
    initial = manifest["initialization"]["checkpoint"]["path"]
    entry = campaign.config_record(tmp_path, "feedback", campaign.make_config(
        template, tmp_path, "feedback", amse_lambda=.5, steps=100, bptt=2, feedback="closed_loop"), initial)
    task = campaign.evaluation_task(entry)
    smoke_entries = []
    for index in range(7):
        config = file(tmp_path / f"smoke{index}.json", "{}")
        checkpoint = file(tmp_path / f"smoke{index}.pkl")
        smoke = {"name": f"smoke{index}", "config": campaign.pinned(config), "checkpoint": str(checkpoint)}
        smoke_entries.append(smoke)
        campaign.write(tmp_path / "checks" / f"smoke{index}_complete.json", {
            "passed": True, "source_digest": "frozen", "config": smoke["config"],
            "checkpoint": campaign.pinned(checkpoint), "peak_rss_gib": 200., "steady_p90_seconds": 25.})
    manifest["smoke_entries"] = smoke_entries
    evidence = file(tmp_path / "decision_evidence.json", "{}")
    def prepare(experiment, phase, tasks, **kwargs):
        campaign.write(experiment / phase / "tasks.json", tasks)
        campaign.write(experiment / phase / "evaluation/manifest.json", {"task_specs": tasks})
    monkeypatch.setattr(campaign, "prepare_eval", prepare)
    monkeypatch.setattr(evaluator, "load_manifest", lambda path: campaign.read(path))
    submitted = []
    monkeypatch.setattr(campaign, "submit_job", lambda experiment, name, args:
                        submitted.append((name, list(args))) or str(100 + len(submitted)))
    arguments = (tmp_path, manifest, "feedback", [entry], [task], [campaign.pinned(evidence)], {"recipe": "fixed"})
    campaign.launch_phase(*arguments)
    assert [name for name, _ in submitted] == ["feedback_train", "feedback_eval", "feedback_report"]
    assert "--dependency=afterok:101" in submitted[1][1]
    assert "--dependency=afterok:102" in submitted[2][1]
    resources = campaign.read(tmp_path / "feedback/resources.json")
    assert resources["max_updates_per_task"] == 100 and len(resources["preflight_evidence"]) == 7
    assert campaign.read_plan(tmp_path, "feedback")["evidence"] == [campaign.pinned(evidence)]
    # A proof from another smoke task must not authorize continuation.
    first = tmp_path / "checks/smoke0_complete.json"
    campaign.write(first, campaign.read(tmp_path / "checks/smoke1_complete.json"))
    with pytest.raises(ValueError, match="preflight provenance"):
        campaign.launch_phase(*arguments)
    assert len(submitted) == 3
    file(evidence, "mutated decision evidence")
    with pytest.raises(ValueError, match="Pinned file changed"):
        campaign.read_plan(tmp_path, "feedback")


def test_smoke_train_completion_proof_and_partial_run_guard(template, tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya import checkpoint
    initial = file(tmp_path / "initial.pkl")
    config = campaign.make_config(template, tmp_path, "smoke", steps=8)
    baseline_path = file(tmp_path / "baseline.npz")
    config["data"]["baseline_checkpoint"] = str(baseline_path)
    entry = campaign.config_record(tmp_path, "smoke", config, initial)
    parsed = load_training_config(Path(entry["config"]["path"]))
    manifest = {"source_digest": "frozen", "initialization": {"checkpoint": campaign.pinned(initial)}}
    calls = []

    def training(args):
        calls.append(list(map(str, args)))
        if "-c" in args:
            return
        directory = Path(entry["run_dir"])
        file(entry["checkpoint"])
        rows = [{"step": index, "loss": 1. / index, "gradient_norm": .2,
                 "step_seconds": 2., "end_to_end_seconds": 3.} for index in range(1, 9)]
        file(directory / "train_metrics.jsonl", "\n".join(json.dumps(row) for row in rows))

    monkeypatch.setattr(campaign, "call", training)
    monkeypatch.setattr(checkpoint, "load_v24_Ilya_training_checkpoint", lambda path: SimpleNamespace(
        completed_step=8, resolved_training_config=parsed.to_dict(), residual_params={"head": {"w": np.ones(2)}},
        baseline_checkpoint_fingerprint=campaign.file_sha256(baseline_path)))
    proof = campaign.train_entry(tmp_path, manifest, entry)
    assert proof["passed"] is True
    assert proof["initial_checkpoint"] == campaign.pinned(initial)
    assert proof["steady_p90_seconds"] == 3.
    assert calls[1][-3:] == ["--init-from", str(initial), "--zero-state-on-init"]
    assert campaign.train_entry(tmp_path, manifest, entry) == proof
    assert len(calls) == 2
    file(entry["checkpoint"], "modified")
    with pytest.raises(ValueError, match="Pinned file changed"):
        campaign.train_entry(tmp_path, manifest, entry)
    second = campaign.config_record(tmp_path, "partial", campaign.make_config(template, tmp_path, "partial", steps=8), initial)
    Path(second["run_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Partial run"):
        campaign.train_entry(tmp_path, manifest, second)


def test_spherical_closure_is_stable_for_large_physical_offset_and_small_error():
    import jax.numpy as jnp
    from src.models.mamba.v24_Ilya.spherical_observer import diagnostic_kernel
    lat, lon = np.linspace(-90, 90, 9), np.arange(16) * 22.5
    _, calculate = diagnostic_kernel(lat, lon, lmax=5)
    rng = np.random.default_rng(3)
    truth = (100000. + rng.normal(size=(2, 9, 16))).astype(np.float32)
    full = (truth + .1 * rng.normal(size=truth.shape)).astype(np.float32)
    result = {key: np.asarray(value) for key, value in calculate(jnp.asarray(np.stack([truth, truth, full]))).items()}
    assert all(np.isfinite(value).all() for value in result.values())
    np.testing.assert_allclose(result["full_grid_mse"], result["full_projected_grid_mse"] + result["full_unresolved_grid_mse"],
                               rtol=3e-4, atol=1e-10)
    assert np.all(result["full_unresolved_grid_mse"] > 0)
    np.testing.assert_array_equal(result["baseline_grid_mse"], 0.)


def test_spherical_observer_publishes_complete_finite_physical_diagnostics(tmp_path):
    import xarray as xr
    from src.models.mamba.v24_Ilya.spectral_observer import FIELD_SPECS, SPECTRAL_VARIABLES
    from src.models.mamba.v24_Ilya.spherical_observer import SphericalObserver
    lat, lon = np.linspace(-90, 90, 9), np.arange(16) * 22.5
    wave = np.cos(np.deg2rad(lat))[:, None] * np.cos(3 * np.deg2rad(lon))[None, :]
    def fields(offset=0.):
        values = {}
        for code, name, level, _ in FIELD_SPECS:
            center = 100000. if code == "MSLP" else 280. if code in ("T2m", "T850") else 1.
            plane = (center + wave + offset).astype(np.float32)
            dims = ("batch", "time", "lat", "lon")
            data = plane[None, None]
            if level is not None:
                dims = ("batch", "time", "level", "lat", "lon")
                data = np.stack((data, data), axis=2)
            values[name] = (dims, data)
        return xr.Dataset(values, coords={"batch": [0], "time": [np.timedelta64(6, "h")],
                                          "lat": lat, "lon": lon, "level": [700, 850]})
    identity = {"model_id": "candidate", "checkpoint_sha256": "checkpoint", "protocol_hash": "protocol"}
    metadata = {"initialization_id": "20230102T000000Z", "initialization_time": "2023-01-02T00:00:00Z",
                "lead_hours": [6, 12], "valid_times": ["2023-01-02T06:00:00Z", "2023-01-02T12:00:00Z"]}
    observer = SphericalObserver(tmp_path, identity, target_steps=2, lmax=5)
    observer.begin_sample(metadata)
    observer.update_step(0, fields(), fields(.2), fields(.1))
    assert not list(tmp_path.glob("*.npz"))
    with pytest.raises(ValueError, match="Incomplete"):
        observer.finish_sample()
    observer.update_step(1, fields(), fields(.4), fields(.3))
    observer.finish_sample()
    with np.load(tmp_path / "20230102T000000Z.npz", allow_pickle=False) as result:
        assert result["full_power"].shape == (2, len(SPECTRAL_VARIABLES), 6)
        assert tuple(result["variables"]) == SPECTRAL_VARIABLES
        np.testing.assert_array_equal(result["lead_hours"], [6, 12])
        for key in ("truth_power", "baseline_power", "full_power", "full_grid_mse"):
            assert np.isfinite(result[key]).all()
        saved = json.loads(str(result["metadata"]))
        assert saved["status"] == "complete" and saved["lmax"] == 5
        for key, value in identity.items():
            assert saved[key] == value
    with pytest.raises(FileExistsError):
        observer.begin_sample(metadata)
