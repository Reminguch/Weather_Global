"""Three-source joins retain GC lambda0 and never use rejected positive weights."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments import prepare_v24_mixed_final_comparison as mixed
from scripts.experiments import run_v24_low_lr_downstream as downstream


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def positive_fixture(root, *, constant=False, optimizer_overrides=None):
    tasks = []
    for weight in ((0., .25, .5, .75, 1.) if constant else (.25, .5, .75, 1.)):
        logical = f"gcft_s22_l{round(100 * weight):03d}"
        name = logical + ("_lr3em7_constant_20261112" if constant else "_lr3em7_fresh_20261112")
        config = root / "configs" / f"{name}.json"
        optimizer = {"learning_rate": 3e-7, "max_steps": 2500}
        if constant:
            optimizer.update(learning_rate_schedule="constant", warmup_steps=0, end_learning_rate=None)
        optimizer.update(optimizer_overrides or {})
        write(config, {"model_family": "full_gc", "optimizer": optimizer})
        tasks.append({"id": name, "logical_id": logical, "model_family": "full_gc", "lambda": weight,
                      "learning_rate": 3e-7, "resume_step": 0, "config": mixed.pinned(config)})
    mixed.seal(root / "manifest.json", {"tasks": tasks, "source_digest": "new-frozen-source"})
    mixed.seal(root / "selected_arms_full_gc.json", {"tasks": tasks})
    return tasks


def test_constant_gc_specs_include_fresh_lambda_zero_without_previous_checkpoint(tmp_path):
    positive_fixture(tmp_path, constant=True)
    specs, records = mixed.constant_full_gc_specs(tmp_path)
    assert [spec["lambda"] for spec in specs] == [0, .25, .5, .75, 1]
    assert all(spec["constant_learning_rate"] and spec["learning_rate"] == 3e-7 for spec in specs)
    assert all("run_config" not in spec and not Path(spec["run_config_path"]).exists() for spec in specs)
    assert all(Path(record["path"]).exists() for record in records)


@pytest.mark.parametrize("change", [{"learning_rate_schedule": "cosine"}, {"warmup_steps": 100},
                                    {"end_learning_rate": 3e-8}])
def test_constant_gc_rejects_decay_or_warmup(tmp_path, change):
    positive_fixture(tmp_path, constant=True, optimizer_overrides=change)
    with pytest.raises(ValueError, match="no decay and no warmup"):
        mixed.constant_full_gc_specs(tmp_path)


def test_positive_gc_specs_pin_configs_without_inventing_future_weights(tmp_path):
    positive_fixture(tmp_path)
    specs, records = mixed.positive_full_gc_specs(tmp_path)
    assert len(specs) == 4
    for spec in specs:
        assert spec["learning_rate"] == 3e-7
        assert spec["swa_steps"] == [1000, 1500, 2000, 2500]
        assert spec["final_stored_step"] == 2500
        assert "run_config" not in spec
        assert not Path(spec["run_config_path"]).exists()
        assert spec["final_proof_path"].endswith("_step00002500.json")
    assert all(Path(record["path"]).exists() for record in records)


def test_positive_gc_requires_fresh_low_lr_selection(tmp_path):
    tasks = positive_fixture(tmp_path)
    tasks[0]["resume_step"] = 1250
    (tmp_path / "selected_arms_full_gc.json").unlink()
    mixed.seal(tmp_path / "selected_arms_full_gc.json", {"tasks": tasks})
    with pytest.raises(ValueError, match="cannot resume"):
        mixed.positive_full_gc_specs(tmp_path)


def test_fresh_full_gc_final_config_and_proof_are_resolved_at_completion(tmp_path):
    positive_fixture(tmp_path)
    specs, _ = mixed.positive_full_gc_specs(tmp_path)
    spec = specs[0]
    actual = Path(spec["run_config_path"])
    write(actual, {"config_source": spec["training_config"]["path"]})
    metrics = tmp_path / "metrics.jsonl"
    write(metrics, {"finite_loss_increase": True})
    for path in spec["checkpoint_paths"].values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"native-checkpoint")
    proof_path = Path(spec["final_proof_path"])
    proof_path.parent.mkdir(parents=True)
    mixed.seal(proof_path, {"status": "complete", "completed_steps": 2500,
        "task_id": spec["training_run_id"], "model_family": "full_gc", "lambda": .25,
        "manifest_hash": spec["training_manifest_hash"], "source_digest": spec["training_source_digest"],
        "metrics": mixed.pinned(metrics), "checkpoint": mixed.pinned(spec["checkpoint_paths"]["2500"])})
    resolved, sources, records = mixed.final_sources(spec)
    assert resolved["run_config"] == mixed.pinned(actual)
    assert len(sources) == 4
    assert mixed.pinned(actual) in records
    write(actual, {"config_source": "/rejected/old/config.json"})
    with pytest.raises(ValueError, match="different declared"):
        mixed.final_sources(spec)


def ten_specs():
    specs = []
    for family, prefix in (("gc_mamba", "gcm"), ("full_gc", "gcft")):
        for weight in (0., .25, .5, .75, 1.):
            logical = f"{prefix}_s22_l{round(weight * 100):03d}"
            campaign = "new_positive" if family == "full_gc" and weight else "retained_gc" if family == "full_gc" else "gcm"
            name = logical + "_fresh"
            specs.append({"id": logical, "training_run_id": name, "model_family": family, "lambda": weight,
                "learning_rate": 3e-7 if family == "full_gc" else 1e-4,
                "training_config": {"path": f"/archive/{campaign}/configs/{name}.json"},
                "final_proof_path": f"/archive/{campaign}/checks/{name}.json"})
    return specs


def test_three_source_selection_preserves_all_gcm_and_only_low_lr_gc():
    specs = ten_specs()
    selected = downstream.selected_tasks(specs)
    assert len(selected) == 10
    assert len([task for task in selected if task["learning_rate"] == 3e-7]) == 5
    assert {task["training_elapsed_field"] for task in selected if task["model_family"] == "gc_mamba"} == {"allocation_elapsed_seconds"}
    bad = deepcopy(specs)
    bad[-1]["training_config"] = {"path": "/archive/res1_fresh_gc_mamba_gc_lambda_20261107/configs/gcft_s22_l100.json"}
    with pytest.raises(ValueError, match="Rejected"):
        downstream.selected_tasks(bad)


def test_constant_selection_cannot_retain_old_lambda_zero():
    specs = ten_specs()
    for spec in specs:
        if spec["model_family"] == "full_gc":
            spec["constant_learning_rate"] = True
            spec["final_proof_path"] = f"/archive/new_constant_gc/checks/{spec['training_run_id']}.json"
    assert len(downstream.selected_tasks(specs, constant=True)) == 10
    bad = deepcopy(specs)
    bad[5]["constant_learning_rate"] = False
    with pytest.raises(ValueError, match="retained or cosine"):
        downstream.selected_tasks(bad, constant=True)
    bad = deepcopy(specs)
    bad[5]["final_proof_path"] = "/archive/old_gc_lambda0/checks/endpoint.json"
    with pytest.raises(ValueError, match="same fresh training"):
        downstream.selected_tasks(bad, constant=True)


def test_constant_controller_has_only_new_gc_and_unchanged_gcm_dependencies(tmp_path):
    manifest = {"workspace": "/workspace", "source_root": "/source",
                "resources": {"cpus": 2, "mem_gib": 8, "minutes": 30}}
    args = downstream.command(tmp_path, manifest, "start", ["99999999", "17406862"])
    assert "--dependency=afterok:99999999:17406862" in args
    assert "17049163" not in str(args)


def test_new_controller_depends_on_exact_retained_gc_task_not_cancelled_siblings(tmp_path):
    manifest = {"workspace": "/workspace", "source_root": "/source",
                "resources": {"cpus": 2, "mem_gib": 8, "minutes": 30}}
    args = downstream.command(tmp_path, manifest, "start", ["17049163_0", "99999999", "17406862"])
    assert "--dependency=afterok:17049163_0:99999999:17406862" in args
    assert "17049165" not in str(args) and "17406864" not in str(args)
    with pytest.raises(ValueError, match="Invalid dependency"):
        downstream.command(tmp_path, manifest, "start", ["17049163_0:17049165"])


def test_mixed_final_supports_three_source_dependencies_and_twenty_evaluations(tmp_path):
    plan = {"workspace": "/workspace", "source_root": "/source",
        "training_dependencies": ["17049163_0", "99999999", "17406862"],
        "resources": {"materialize": {"cpus": 2, "mem_gib": 8, "minutes": 30},
                      "evaluate": {"cpus": 4, "mem_gib": 16, "minutes": 62},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 15}}}
    commands = mixed.submission_plan(tmp_path, plan)
    assert "--dependency=afterok:17049163_0:99999999:17406862" in commands["materialize"]
    assert "--array=2-21" in commands["evaluate"]
