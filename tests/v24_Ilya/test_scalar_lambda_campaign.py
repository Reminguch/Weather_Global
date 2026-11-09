"""Budget retention, development-only selection and immutable resume contracts."""
import copy
import json
from pathlib import Path

import pytest

from scripts.experiments import run_v24_scalar_lambda as campaign


def template():
    return campaign.read(campaign.ROOT / campaign.TEMPLATE)


def manifest(tmp_path):
    return {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"),
            "record_hash": "campaign", "source_digest": "frozen", "template": template(),
            "initial_tasks": campaign.initial_tasks()}


def scores(tasks, lower_wins=True):
    return [{"task": task, "exact_gc_rollout_loss": 20. + (
        index % 2 if lower_wins else 1 - index % 2)} for index, task in enumerate(tasks[:4])]


def test_all_profiles_keep_the_complete_schedule_and_fresh_start(tmp_path):
    original = template()
    before = copy.deepcopy(original)
    tasks = campaign.initial_tasks()
    assert len(tasks) == 6
    assert [x["lambda"] for x in tasks] == [0., 0., 0., 0., .5, .5]
    for task in tasks:
        value = campaign.make_config(original, tmp_path, task, campaign.ROOT)
        assert value["model_family"] == task["model_family"]
        assert value["objective"]["loss_kind"] == "gc_mse_amse_excess"
        assert value["objective"]["spectral_excess_weight"] == task["lambda"]
        assert value["optimizer"]["max_steps"] == 2500
        assert value["optimizer"]["warmup_steps"] == 100
        assert value["optimizer"]["end_learning_rate"] == .1 * task["learning_rate"]
        assert value["optimizer"]["optimizer_kind"] == "adamw"
        assert value["optimizer"]["mamba_lr_multiplier"] == (3 if task["model_family"] == "gc_mamba" else 1)
        assert value["sequence"]["feedback_mode"] == "closed_loop"
        assert tuple(value["sequence"][k] for k in ("bptt_steps", "ar_tail_k", "segment_steps")) == (24, 20, 96)
        assert value["distributed"]["per_device_batch_size"] == 4
        assert value["data"]["train_order_manifest"] == str(tmp_path / "manifests/train_order_seed22.json")
        assert "init_from" not in value and "resume" not in value
    assert original == before


def test_selection_requires_both_rates_but_not_an_improvement_over_gc():
    tasks = campaign.initial_tasks()
    records = scores(tasks)
    # Deliberately poor finite development losses are still valid selection evidence.
    for record in records:
        record["exact_gc_rollout_loss"] += 1e6
    winners = campaign.select_winners(records)
    assert all(winners[f]["task"]["learning_rate"] == campaign.LEARNING_RATES[f][0]
               for f in campaign.FAMILIES)
    for record in records:
        record["exact_gc_rollout_loss"] = 123.
    assert campaign.select_winners(records) == winners
    with pytest.raises(ValueError, match="Both prespecified"):
        campaign.select_winners(records[:3])
    records[0]["exact_gc_rollout_loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        campaign.select_winners(records)


@pytest.mark.parametrize("lower_wins", [True, False])
def test_selection_retains_only_matching_profile_and_calibration_trajectories(tmp_path, monkeypatch, lower_wins):
    from src.models.mamba.v24_Ilya.training import config as config_module
    monkeypatch.setattr(config_module, "load_training_config", lambda path: None)
    data = manifest(tmp_path)
    for task in data["initial_tasks"]:
        path = campaign.config_path(tmp_path, task)
        campaign.write(path, campaign.make_config(data["template"], tmp_path, task, campaign.ROOT))
        task["config"] = campaign.pinned(path)
    result = campaign.choose_arms(tmp_path, data, scores(data["initial_tasks"], lower_wins))
    assert len(result["tasks"]) == 10
    assert len({task["logical_id"] for task in result["tasks"]}) == 10
    for task in result["tasks"]:
        expected = campaign.LEARNING_RATES[task["model_family"]][0 if lower_wins else 1]
        assert task["learning_rate"] == expected
        if task["lambda"] == 0:
            assert task["resume_step"] == 500
        elif task["lambda"] == .5 and lower_wins:
            assert task["resume_step"] == 8
        else:
            assert task["resume_step"] == 0
    assert result["profile_overhead_updates"] == (0 if lower_wins else 16)


def test_profile_jobs_and_release_dependencies_are_unthrottled(tmp_path):
    plan = campaign.initial_submission_plan(tmp_path, manifest(tmp_path))
    assert "--array=0-5" in plan[0]["command"]
    assert "--mem=384G" in plan[0]["command"]
    assert "--time=180" in plan[0]["command"]
    assert "--dependency=afterok:<profile>_0:<profile>_1:<profile>_4" in plan[1]["command"]
    assert "--dependency=afterok:<profile>_2:<profile>_3:<profile>_5" in plan[2]["command"]
    with pytest.raises(ValueError, match="throttles"):
        campaign.sbatch(tmp_path, manifest(tmp_path), "profile", memory=384, minutes=180, array="0-5%2")


def test_gpu_wrapper_removes_inherited_cpu_platform_selection():
    wrapper = (campaign.ROOT / campaign.WRAPPER).read_text()
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME" in wrapper
    assert "WEATHER_PREPARED_BOUNDED_READS=0" in wrapper


def test_measured_budget_uses_latency_and_long_run_host_memory_floor():
    data = [{"steady_p90_seconds": 100., "peak_rss_gib": 10., "startup_seconds": 100.}]
    result = campaign.measured_resources(data, 1250)
    assert result["memory"] >= 234.71 * 1.5
    assert result["minutes"] * 60 >= 1250 * 100 * 1.4 + 1800
    data[0]["steady_p90_seconds"] = 1e6
    with pytest.raises(ValueError, match="resource estimate"):
        campaign.measured_resources(data, 1250)


def test_mamba_calibration_proceeds_without_any_full_gc_profiles(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    for task in data["initial_tasks"]:
        if task["model_family"] == "gc_mamba":
            campaign.seal(campaign.proof_path(tmp_path, task, 8), {
                "model_family": "gc_mamba", "lambda": task["lambda"],
                "initial_numerical_identity": {"params": "same"}, "steady_p90_seconds": 50.,
                "peak_rss_gib": 80., "startup_seconds": 100.})
    captured = []
    monkeypatch.setattr(campaign, "submit_nodes", lambda experiment, nodes: captured.extend(nodes))
    campaign.release_calibration(tmp_path, data, "gc_mamba")
    assert [node["id"] for node in captured] == ["calibrate-gc_mamba", "evaluate-calibration-gc_mamba", "select-gc_mamba"]
    assert "--array=2-3" in captured[0]["command"]
    assert "--dependency=aftercorr:<calibrate-gc_mamba>" in captured[1]["command"]
    assert "--dependency=afterok:<evaluate-calibration-gc_mamba>" in captured[2]["command"]
    assert all(node["command"][-1] == "gc_mamba" for node in captured)


def fake_training(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    task = data["initial_tasks"][0]
    path = campaign.config_path(tmp_path, task)
    campaign.write(path, campaign.make_config(data["template"], tmp_path, task, campaign.ROOT))
    task["config"] = campaign.pinned(path)
    monkeypatch.setattr(campaign, "gpu_required", lambda: None)
    monkeypatch.setattr(campaign, "numeric_initial_identity", lambda path: {"params": "same"})
    calls = []
    def execute(command):
        calls.append(command)
        stop = int(command[command.index("--stop-after-steps") + 1])
        for step in (0, stop):
            path = campaign.checkpoint_path(tmp_path, task, step)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"step{step}".encode())
        rows = [{"step": i, "loss": float(i), "gradient_norm": 1., "end_to_end_seconds": 1.}
                for i in range(1, stop + 1)]
        (tmp_path / "runs" / task["id"] / "train_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(campaign, "call", execute)
    return data, task, calls


def test_retained_profile_resume_never_changes_schedule_or_requires_descent(tmp_path, monkeypatch):
    data, task, calls = fake_training(tmp_path, monkeypatch)
    profile = campaign.run_training_stage(tmp_path, data, task, stop=8)
    assert profile["completed_steps"] == 8 and not profile["scientific_improvement_required"]
    calibration = campaign.run_training_stage(tmp_path, data, task, stop=500, previous=8)
    assert calibration["started_from_step"] == 8
    campaign.check(profile["metrics"])
    assert len(Path(profile["metrics"]["path"]).read_text().splitlines()) == 8
    assert len(Path(calibration["metrics"]["path"]).read_text().splitlines()) == 500
    assert calls[1][-2:] == ["--resume", profile["checkpoint"]["path"]]
    assert "--max-steps" not in calls[0] and "--init-from" not in calls[0]
    assert campaign.read(task["config"]["path"])["optimizer"]["max_steps"] == 2500
    with pytest.raises(FileExistsError, match="must not be repeated"):
        campaign.run_training_stage(tmp_path, data, task, stop=500, previous=8)


def test_resume_rejects_changed_checkpoint_before_executing(tmp_path, monkeypatch):
    data, task, calls = fake_training(tmp_path, monkeypatch)
    profile = campaign.run_training_stage(tmp_path, data, task, stop=8)
    Path(profile["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        campaign.run_training_stage(tmp_path, data, task, stop=500, previous=8)
    assert len(calls) == 1
