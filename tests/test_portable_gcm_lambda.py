"""Check path relocation without changing the already-tested training recipe."""
import copy
import json
from pathlib import Path

import pytest

from scripts.experiments import run_v24_m1_matched_gcm_lambda as original
from scripts.experiments import run_v24_portable_gcm_lambda as portable


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "handoff"
    root.mkdir()
    for relative in ("src/portable_fixture.py", "scripts/fixture.py",
                     "third_party/graphcast/graphcast/graphcast.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    (root / "templates").mkdir()
    for stage, relative in original.TEMPLATES.items():
        source = original.ROOT / "templates" / f"{stage}.json"
        if not source.exists():
            source = original.ROOT / relative
        (root / "templates" / f"{stage}.json").write_text(source.read_text())
    paths = {name: tmp_path / "relocated data" / name for name in portable.INPUT_NAMES}
    entries = []
    for name, relative in sorted(portable.REQUIRED_INPUTS):
        path = paths[name] / relative if relative else paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {name}/{relative}\n")
        entries.append({"input": name, "relative_path": relative,
                        "sha256": portable.pinned(path)["sha256"]})
    portable.write(root / "portable/input_reference.json", {"files": entries, "origin": "unit fixture"})
    monkeypatch.setattr(portable, "ROOT", root)
    return root, paths


def test_local_prepare_changes_only_data_and_output_paths(package, tmp_path):
    root, paths = package
    experiment = tmp_path / "new experiment"
    manifest = portable.prepare(root, experiment, paths)
    assert manifest["format"] == original.FORMAT
    assert manifest["completion_hook"] is None
    assert [task["lambda"] for task in manifest["tasks"]] == list(original.LAMBDAS)
    assert manifest["policy"]["swa_main_steps"] == [500, 1000, 1500, 2000]
    assert manifest["policy"]["swa_total_updates"] == [1000, 1500, 2000, 2500]
    assert "development_data" not in manifest and "reference_campaign" not in manifest
    for task in manifest["tasks"]:
        for stage in ("initial", "main"):
            saved = portable.read(task[f"{stage}_config"]["path"])
            template = portable.read(root / "templates" / f"{stage}.json")
            expected = original.make_config(template, experiment, task, root, stage)
            for name in portable.INPUT_NAMES:
                assert saved["data"][name] == str(paths[name].resolve())
                expected["data"][name] = saved["data"][name]
            assert saved == expected
            assert saved["output"]["output_root"].startswith(str(experiment))
    assert portable.verify(experiment)["record_hash"] == manifest["record_hash"]
    plan = portable.execution_plan(experiment, manifest)
    assert plan["part1"][4] == "1" and plan["part2"][4] == "2"
    assert "dabanin" not in json.dumps(plan)
    assert "sbatch" not in json.dumps(plan)
    with pytest.raises(FileExistsError, match="new campaign"):
        portable.prepare(root, experiment, paths)


def test_altered_input_fingerprint_fails_before_creating_campaign(package, tmp_path):
    root, paths = package
    paths["baseline_checkpoint"].write_text("different initializer")
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="Input fingerprint differs"):
        portable.prepare(root, destination, paths)
    assert not destination.exists()


def test_required_input_list_cannot_be_silently_weakened(package):
    root, paths = package
    reference = portable.read(root / "portable/input_reference.json")
    reference["files"].pop()
    portable.write(root / "portable/input_reference.json", reference)
    with pytest.raises(ValueError, match="Required input fingerprints"):
        portable.validate_inputs(root, paths)


def test_reference_disallows_escape_and_duplicate_files(package):
    root, paths = package
    path = root / "portable/input_reference.json"
    reference = portable.read(path)
    altered = copy.deepcopy(reference)
    altered["files"].append(copy.deepcopy(altered["files"][0]))
    portable.write(path, altered)
    with pytest.raises(ValueError, match="Duplicate"):
        portable.validate_inputs(root, paths)
    reference["files"].append({"input": "stats_dir", "relative_path": "../secret", "sha256": "bad"})
    portable.write(path, reference)
    with pytest.raises(ValueError, match="relative path"):
        portable.validate_inputs(root, paths)


def test_local_manifest_preserves_inputs_and_rejects_source_mutation(package, tmp_path):
    root, paths = package
    destination = tmp_path / "campaign"
    manifest = portable.prepare(root, destination, paths)
    (Path(manifest["source_root"]) / "src/portable_fixture.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        portable.verify(destination)


def test_execution_dispatch_uses_original_stage_functions(monkeypatch, tmp_path, capsys):
    manifest = {"tasks": [{"id": "arm"}]}
    monkeypatch.setattr(portable, "verify", lambda root, frozen: manifest)
    calls = []
    monkeypatch.setattr(original, "train", lambda *args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(portable.sys, "argv", ["driver", "train", "--experiment", str(tmp_path),
                                              "--task", "3", "--part", "2"])
    portable.main()
    assert calls == [(tmp_path, manifest, 3, 2)]
    calls.clear()
    monkeypatch.setattr(original, "finish", lambda *args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(portable.sys, "argv", ["driver", "finish", "--experiment", str(tmp_path)])
    portable.main()
    assert calls == [(tmp_path, manifest)]


def test_wrapper_uses_activated_environment_and_no_cluster_defaults():
    script = (original.ROOT / portable.WRAPPER).read_text()
    assert "WEATHER_ACTIVATE_SCRIPT" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "unset JAX_PLATFORMS JAX_PLATFORM_NAME" in script
    assert "--task" in script and "--part" in script
    for site_detail in ("dabanin", "gpu80", "gpu-medium", "graphcast_env.sh", "sbatch", "scancel"):
        assert site_detail not in script
