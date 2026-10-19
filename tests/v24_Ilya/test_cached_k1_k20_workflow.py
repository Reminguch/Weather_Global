"""Focused curriculum workflow guards without submitting Slurm jobs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v24_cached_k1_k20 as workflow
from src.models.mamba.v24_Ilya.training.config import load_training_config


@pytest.fixture
def experiment(tmp_path):
    producer = workflow.read(workflow.ROOT / "artifacts/cache_builds/v24_res1_gcsmall_bptt24_k0_20260919/config.json")
    original = copy.deepcopy(producer)
    configs, entries = workflow.make_configs(producer, tmp_path)
    assert producer == original
    for name, value in configs.items():
        workflow.write(tmp_path / "configs" / f"{name}.json", value)
    eval_data = tmp_path / "evaluation.zarr"
    workflow.write(eval_data / ".zmetadata", {
        "zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}})
    workflow.write(tmp_path / "manifest.json", {
        "workspace": str(workflow.ROOT), "entries": entries,
        "eval_data": str(eval_data), "eval_metadata": workflow.frozen_zarr_metadata(eval_data), "epochs": 25})
    workflow.write(tmp_path / "checks/cache_release.json", {"prepared_source_gib": 238.36})
    (tmp_path / "logs").mkdir()
    return tmp_path, configs, entries


def test_configs_preserve_full_preflight_schedule_and_use_live_k20(experiment):
    root, configs, entries = experiment
    for width, batch, steps, live_steps in ((128, 8, 1400, 250), (512, 4, 2700, 500)):
        name = f"w{width}"
        config = load_training_config(root / "configs" / f"{name}_k1.json")
        smoke = load_training_config(root / "configs" / f"{name}_k1_preflight.json")
        live = load_training_config(root / "configs" / f"{name}_k20.json")
        assert config.architecture.residual_width == width
        assert config.architecture.width == 512
        assert config.architecture.residual_initialization == "fresh"
        assert config.distributed.global_batch_size == batch
        assert config.max_steps == smoke.max_steps == steps
        assert config.learning_rate_schedule == smoke.learning_rate_schedule == "cosine"
        assert config.end_learning_rate == smoke.end_learning_rate == 1e-5
        assert config.warmup_steps == smoke.warmup_steps == steps // 25
        assert (config.ar_tail_k, config.feedback_mode) == (0, "baseline")
        assert (live.ar_tail_k, live.feedback_mode) == (20, "closed_loop_sg")
        assert live.max_steps == live_steps
        assert live.learning_rate == 1e-5 and live.warmup_steps == 0
        assert live.learning_rate_schedule == "constant"
        assert config.bptt_steps == live.bptt_steps == 24
        assert config.segment_steps == live.segment_steps == 96


def test_width512_only_keeps_curriculum_and_excludes_width128(experiment):
    root, configs, _ = experiment
    selected, entries = workflow.make_configs(configs["w512_k1"], root, widths=(512,))
    assert {entry["name"] for entry in entries} == {"w512"}
    assert all(name.startswith("w512_") for name in selected)
    assert entries[0]["k1_steps"] == 2700
    assert entries[0]["k20_steps"] == 500
    for invalid in ((), (512, 512), (256,)):
        with pytest.raises(ValueError, match="unique supported"):
            workflow.make_configs(configs["w512_k1"], root, widths=invalid)


def test_sbatch_records_idempotent_afterok_dependencies_and_snapshot_launcher(experiment, monkeypatch):
    root, _, _ = experiment
    calls = []
    def run(args, **options):
        calls.append(args)
        return SimpleNamespace(stdout="12345;cluster\n")
    monkeypatch.setattr(workflow.subprocess, "run", run)
    job = workflow.submit_job(root, "test", "k1-preflight", dependency=[111, 222],
                              array="0-1", gpu=True, memory=96, minutes=120)
    assert job == "12345"
    assert "--dependency=afterok:111:222" in calls[0]
    assert "--array=0-1" in calls[0]
    assert "--constraint=gpu40&nomig" in calls[0]
    assert "--mem=96G" in calls[0] and "--time=120" in calls[0]
    assert str(root / "source" / workflow.LAUNCHER) in calls[0]
    assert workflow.submit_job(root, "test", "k1-preflight") == job
    assert len(calls) == 1


def test_verify_rejects_mutated_frozen_config(experiment, monkeypatch):
    root, _, _ = experiment
    reference = root / "reference_evaluation.json"
    workflow.write(reference, {"chosen_idx": [1]})
    manifest = workflow.read(root / "manifest.json")
    manifest["reference_evaluation_sha256"] = workflow.file_sha256(reference)
    workflow.write(root / "manifest.json", manifest)
    hashes = {str(path.relative_to(root)): workflow.file_sha256(path)
              for path in (root / "configs").glob("*.json")}
    workflow.write(root / "config_hashes.json", hashes)
    workflow.write(root / "identity.json", {
        "source_digest": "fixed", "manifest_sha256": workflow.file_sha256(root / "manifest.json"),
        "config_hashes_sha256": workflow.file_sha256(root / "config_hashes.json")})
    monkeypatch.setattr(workflow, "verify_source", lambda path: {"source_digest": "fixed"})
    workflow.verify(root, frozen=False)
    with (root / "configs/w128_k1.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Changed configuration"):
        workflow.verify(root, frozen=False)


def test_k1_release_scales_validation_duration_not_entire_preflight(experiment, monkeypatch):
    root, _, _ = experiment
    for name in ("w128", "w512"):
        workflow.write(root / "checks" / f"{name}_k1_preflight.json", {
            "passed": True, "elapsed_seconds": 999,
            "validation": {"duration_seconds": 2}})
    budgets, jobs = [], []
    def budget(report, updates, *, extra_seconds=0):
        budgets.append((updates, extra_seconds))
        return 32, 120
    monkeypatch.setattr(workflow, "measured_budget", budget)
    def submit(*args, **kwargs):
        jobs.append((args, kwargs))
        return str(len(jobs))
    monkeypatch.setattr(workflow, "submit_job", submit)
    workflow.k1_release(root)
    assert budgets == [(280, 5475), (540, 5475)]
    # Every subsequent block depends on its own prior checkpoint/evaluation job.
    for offset in (0, 6):
        assert jobs[offset][1]["dependency"] is None
        for i in range(1, 5):
            assert jobs[offset + i][1]["dependency"] == [str(offset + i)]
        assert jobs[offset + 5][1]["dependency"] == [str(offset + 5)]


def _saved_eval(root, entry, *, wrong_checkpoint=False):
    requested = root / "requested.pkl"
    requested.write_bytes(b"native residual checkpoint")
    config = workflow.read(entry["k1"])
    baseline = root / "baseline.npz"
    baseline.write_bytes(b"fixed baseline")
    config["data"]["baseline_checkpoint"] = str(baseline)
    workflow.write(entry["k1"], config)
    workflow.write(root / "identity.json", {"source_digest": "frozen-source"})
    result = {
        "evaluation_status": "complete", "evaluated_samples": 8, "target_steps": 40,
        "chosen_idx": list(range(8)), "ckpt": str(root / "different.pkl") if wrong_checkpoint else str(requested),
        "ckpt_in": config["data"]["baseline_checkpoint"],
        "eval_mode": "cold_full", "residual_state_init_resolved": "zero",
        "residual_state_init": "zero", "eval_feedback": "full",
        "original_graphcast_loss": {"full_rollout": 1.25, "baseline_rollout": 1.5,
                                   "improvement_pct_rollout": 100 * (1 - 1.25 / 1.5)},
    }
    workflow.write(root / "eval" / entry["name"] / "existing.json", result)
    workflow.write(root / "eval" / entry["name"] / "existing.provenance.json", {
        "checkpoint": str(requested.resolve()), "checkpoint_sha256": workflow.file_sha256(requested),
        "baseline_checkpoint": str(baseline.resolve()), "baseline_sha256": workflow.file_sha256(baseline),
        "source_digest": "frozen-source", "samples": 8, "leads": 40,
        "mode": "cold_full", "initial_state": "zero", "seed": 0})
    return requested


def test_reused_evaluation_cannot_be_assigned_to_another_checkpoint(experiment):
    root, _, entries = experiment
    requested = _saved_eval(root, entries[0], wrong_checkpoint=True)
    with pytest.raises(ValueError):
        workflow.eval_model(root, entries[0], requested, "existing")


def test_valid_native_scalar_metric_evaluation_is_reused_with_hash_checks(experiment):
    root, _, entries = experiment
    requested = _saved_eval(root, entries[0])
    actual = workflow.eval_model(root, entries[0], requested, "existing")
    assert actual["loss"] == 1.25
    requested.write_bytes(b"different weights in same path")
    with pytest.raises(ValueError, match="provenance"):
        workflow.eval_model(root, entries[0], requested, "existing")


def test_failed_or_nonfinite_resource_evidence_cannot_release_jobs():
    valid = {"passed": True, "peak_host_gib": 20, "steady_seconds": 10, "elapsed_seconds": 100}
    assert workflow.measured_budget(valid, 100) == (32, 62)
    for changed in ({"passed": False}, {"peak_host_gib": 0}, {"steady_seconds": float("nan")},
                    {"steady_seconds": float("inf")}):
        with pytest.raises(ValueError):
            workflow.measured_budget({**valid, **changed}, 100)


def test_production_budget_covers_full_source_after_short_preflight(experiment):
    root, _, _ = experiment
    assert workflow.production_memory(root, 32) == 312
    assert workflow.production_memory(root, 384) == 384


def test_cache_stages_keep_pilot_source_after_workspace_edits(tmp_path):
    from scripts.experiments.frozen_runtime import freeze_source
    workspace, root = tmp_path / "workspace", tmp_path / "experiment"
    for directory in ("src", "scripts/preprocessing", "third_party/graphcast/graphcast"):
        (workspace / directory).mkdir(parents=True)
    (workspace / "src/__init__.py").write_text("")
    module = workspace / "src/producer_version.py"
    module.write_text('VERSION = "pilot-code"\n')
    (workspace / "third_party/graphcast/graphcast/graphcast.py").write_text("# vendor\n")
    script = workspace / "scripts/preprocessing/precompute_v24_baseline.py"
    script.write_text('''
import json, sys
from pathlib import Path
from src.producer_version import VERSION
args = sys.argv[1:]
out = Path(args[args.index('--output-root') + 1])
out.mkdir(parents=True, exist_ok=True)
phase = 'verify' if '--verify' in args else 'pilot' if '--pilot-parity' in args else 'generate'
(out / (phase + '.json')).write_text(json.dumps({'version': VERSION, 'script': __file__}))
''')
    freeze_source(workspace, root / "source")
    workflow.write(root / "manifest.json", {
        "workspace": str(workspace), "producer_config": str(root / "producer.json"),
        "pilot_root": str(root / "pilot"), "cache_root": str(root / "cache"), "shards": 8})
    workflow.cache_command(root, "pilot", 0)
    module.write_text('VERSION = "changed-code"\n')
    script.write_text('raise RuntimeError("working tree producer ran")\n')
    workflow.cache_command(root, "cache-shard", 0)
    workflow.cache_command(root, "cache-verify", 0)
    for directory, phase in (("pilot", "pilot"), ("cache", "generate"), ("cache", "verify")):
        actual = workflow.read(root / directory / f"{phase}.json")
        assert actual["version"] == "pilot-code"
        assert actual["script"] == str(root / "source/scripts/preprocessing/precompute_v24_baseline.py")


def test_live_stage_transfers_weights_with_zero_state_and_new_optimizer(experiment, monkeypatch):
    root, _, entries = experiment
    parent = root / "selected.pkl"
    monkeypatch.setattr(workflow, "select_parent", lambda *args: parent)
    seen = []
    monkeypatch.setattr(workflow, "command", lambda *args: seen.append(args))
    output = workflow.live_train(root, entries[0], preflight=False)
    args = seen[0][-1]
    assert args == ["--config", entries[0]["k20"], "--init-from", parent, "--zero-state-on-init"]
    assert output.name == "checkpoint_step00000250.pkl"
    assert "--resume" not in args


def _tree_hashes(root):
    return {str(path.relative_to(root)): workflow.file_sha256(path)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    """A historical producer and cache, with unrelated current model edits."""
    producer = workflow.read(workflow.ROOT / "artifacts/cache_builds/v24_res1_gcsmall_bptt24_k0_20260919/config.json")
    workspace, previous = tmp_path / "workspace", tmp_path / "previous"
    root, cache = tmp_path / "recovery", tmp_path / "existing-cache"
    historical_model = "src/models/mamba/v24_Ilya/model.py"
    for relative in (*workflow.RECOVERY_FILES, historical_model,
                     "third_party/graphcast/graphcast/graphcast.py"):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# historical code\n")
    snapshot = workflow.freeze_source(workspace, previous / "source")
    configs, entries = workflow.make_configs(producer, previous, widths=(512,))
    configs["producer"] = producer
    for name, config in configs.items():
        workflow.write(previous / "configs" / f"{name}.json", config)
    eval_data = tmp_path / "evaluation.zarr"
    workflow.write(eval_data / ".zmetadata", {
        "zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}})
    workflow.write(previous / "reference_evaluation.json", {"chosen_idx": list(range(32))})
    cached_manifest = {"manifest_sha256": "a" * 64, "num_shards": 8, "partial": False,
        "chunks": [{"id": shard, "shard": shard, "raw_anchor_indices": [shard]}
                   for shard in range(8)]}
    workflow.write(cache / "manifest.json", cached_manifest)
    workflow.write(cache / "READY.json", {
        "manifest_sha256": "a" * 64, "chunks": 8, "predictions": 8, "shards": 8, "partial": False})
    for shard in range(8):
        workflow.write(cache / f"shard_{shard:03d}" / "COMPLETE.json", {
            "manifest_sha256": "a" * 64, "chunk_ids": [shard], "files": {}})
    workflow.write(previous / "checks/cache_release.json", {
        "passed": True, "manifest_sha256": "a" * 64, "prepared_source_gib": 238.36})
    workflow.write(previous / "manifest.json", {
        "workspace": str(workspace), "experiment": str(previous), "entries": entries,
        "producer_config": str(previous / "configs/producer.json"), "cache_root": str(cache),
        "pilot_root": str(tmp_path / "existing-pilot"), "shards": 8,
        "eval_data": str(eval_data), "eval_metadata": workflow.frozen_zarr_metadata(eval_data),
        "reference_evaluation_sha256": workflow.file_sha256(previous / "reference_evaluation.json")})
    workflow.write(previous / "config_hashes.json", {
        str(path.relative_to(previous)): workflow.file_sha256(path)
        for path in (previous / "configs").glob("*.json")})
    workflow.write(previous / "identity.json", {
        "source_digest": snapshot["source_digest"],
        "manifest_sha256": workflow.file_sha256(previous / "manifest.json"),
        "config_hashes_sha256": workflow.file_sha256(previous / "config_hashes.json")})
    before_previous, before_cache = _tree_hashes(previous), _tree_hashes(cache)
    for relative in workflow.RECOVERY_FILES:
        (workspace / relative).write_text("# repaired consumer code\n")
    (workspace / historical_model).write_text("# unrelated current model change\n")
    (workspace / "src/new_unrelated_module.py").write_text("# unrelated new source\n")
    monkeypatch.setattr(workflow, "ROOT", workspace)
    workflow.prepare_recovery(root, previous, deterministic=True)
    return SimpleNamespace(root=root, previous=previous, workspace=workspace, cache=cache,
        historical_model=historical_model, cached_manifest=cached_manifest,
        before_previous=before_previous, before_cache=before_cache)


def test_recovery_overlays_only_consumer_fixes_and_preserves_history(recovery):
    case = recovery
    manifest = workflow.verify(case.root, frozen=False)
    old_source = workflow.verify_source(case.previous / "source")
    source = workflow.verify_source(case.root / "source")
    assert source["workspace"] == str(case.workspace)
    assert set(source["files"]) == set(old_source["files"])
    changed = {name for name in source["files"]
               if source["files"][name] != old_source["files"][name]}
    assert changed == set(workflow.RECOVERY_FILES)
    assert (case.root / "source" / case.historical_model).read_text() == "# historical code\n"
    assert _tree_hashes(case.previous) == case.before_previous
    assert _tree_hashes(case.cache) == case.before_cache
    assert manifest["cache_root"] == str(case.cache)
    assert manifest["cache_reuse"]["previous_source_digest"] == old_source["source_digest"]
    assert manifest["cache_reuse"]["previous_identity_sha256"] == workflow.file_sha256(case.previous / "identity.json")
    assert manifest["k1_deterministic"] is True
    assert workflow.read(manifest["producer_config"]) == workflow.read(case.previous / "configs/producer.json")
    assert [(entry["width"], entry["k1_steps"], entry["k20_steps"])
            for entry in manifest["entries"]] == [(512, 2700, 500)]


def _verify_recovery_with_mock_data(case, monkeypatch):
    from src.models.mamba.v24_Ilya import baseline_cache
    from src.models.mamba.v24_Ilya.training import cached_data
    seen, shards = [], []
    def open_data(config):
        seen.append(config)
        return SimpleNamespace(manifest=case.cached_manifest,
            identity={"manifest_sha256": "a" * 64, "chunks": 8, "num_shards": 8})
    def verify_shard(root, manifest, shard, *, checksums):
        assert root == case.cache and manifest == case.cached_manifest
        assert checksums is False
        shards.append(shard)
        return workflow.read(case.cache / f"shard_{shard:03d}" / "COMPLETE.json")
    monkeypatch.setattr(cached_data, "open_cached_training_data", open_data)
    monkeypatch.setattr(baseline_cache, "verify_shard", verify_shard)
    monkeypatch.setattr(workflow, "ROOT", case.root / "source")
    workflow.verify_reused_cache(case.root)
    assert len(seen) == 1
    assert seen[0].config_path == case.root / "configs/producer.json"
    assert seen[0].execution.cache_root == case.cache
    assert seen[0].execution.expected_manifest_sha256 == "a" * 64
    assert shards == list(range(8))


def test_recovery_revalidates_all_shards_and_submits_only_training_gate(recovery, monkeypatch):
    case = recovery
    _verify_recovery_with_mock_data(case, monkeypatch)
    evidence = workflow.read(case.root / "checks/cache_reuse.json")
    assert evidence["passed"] is True
    assert evidence["source_digest"] == workflow.read(case.root / "identity.json")["source_digest"]
    assert evidence["manifest_sha256"] == workflow.file_sha256(case.root / "manifest.json")
    calls = []
    def run(args, **options):
        calls.append(args)
        return SimpleNamespace(stdout=f"{1000 + len(calls)};cluster\n")
    monkeypatch.setattr(workflow.subprocess, "run", run)
    workflow.submit(case.root)
    workflow.submit(case.root)
    assert len(calls) == 2
    assert "--job-name=k1k20-k1-preflight" in calls[0]
    assert "--array=0-0" in calls[0]
    assert "--mem=96G" in calls[0] and "--time=120" in calls[0]
    assert not any(arg.startswith("--dependency=") for arg in calls[0])
    assert "--job-name=k1k20-k1-release" in calls[1]
    assert "--dependency=afterok:1001" in calls[1]
    assert set(workflow.read(case.root / "submissions.json")) == {"k1-preflight", "k1-release"}
    assert _tree_hashes(case.previous) == case.before_previous
    assert _tree_hashes(case.cache) == case.before_cache


@pytest.mark.parametrize("changed", ["source", "READY.json", "shard_007/COMPLETE.json",
                                     "release", "verification", "missing_verification"])
def test_recovery_identity_failure_prevents_any_submission(recovery, monkeypatch, changed):
    case = recovery
    _verify_recovery_with_mock_data(case, monkeypatch)
    if changed == "source":
        path = case.root / "source" / case.historical_model
    elif changed == "release":
        path = case.root / "checks/cache_release.json"
    elif changed in ("verification", "missing_verification"):
        path = case.root / "checks/cache_reuse.json"
    else:
        path = case.cache / changed
    if changed == "missing_verification":
        path.unlink()
    elif changed == "verification":
        value = workflow.read(path)
        value["source_digest"] = "stale"
        workflow.write(path, value)
    else:
        with path.open("a") as handle:
            handle.write("\n")
    calls = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.submit(case.root)
    assert calls == []
    assert not (case.root / "submissions.json").exists()


@pytest.mark.parametrize("stage", ["k1", "k20"])
@pytest.mark.parametrize("learned", [True, False])
def test_preflight_requires_parameter_learning_before_release_evidence(experiment, monkeypatch, stage, learned):
    from src.models.mamba.v24_Ilya.training import cached_k1_runner
    root, _, entries = experiment
    entry = entries[1]
    checked = []
    def parameter_report(before, after):
        checked.append((before, after))
        return {"passed": learned}
    # Keep the default raising=True: both preflights must import a real helper.
    monkeypatch.setattr(cached_k1_runner, "parameter_update_report", parameter_report)
    commands = []
    monkeypatch.setattr(workflow, "command", lambda *args: commands.append(args))
    monkeypatch.setattr(workflow, "eval_model", lambda *args, **kwargs: {"loss": 1.0})
    monkeypatch.setattr(workflow, "training_measurement", lambda config: {
        "updates": 20 if stage == "k1" else 8, "steady_seconds": 1.0,
        "max_first_update_seconds": 2.0})
    monkeypatch.setattr(workflow, "gpu_peak", lambda *args: {"peak_mib": 100, "total_mib": 1000})
    parent = root / "selected.pkl"
    monkeypatch.setattr(workflow, "select_parent", lambda *args: parent)
    if stage == "k1":
        run = workflow.run_dir(entry["k1_preflight"])
        workflow.write(run / "checkpoint_roundtrip.json", {"passed": True})
        workflow.write(run / "cached_k1_execution.json", {"peak_host_gib": 8})
        (run / "validation_metrics.jsonl").write_text(json.dumps({"duration_seconds": 3}) + "\n")
        expected_pair = (workflow.checkpoint(entry["k1_preflight"], 0),
                         workflow.checkpoint(entry["k1_preflight"], 20))
        preflight = workflow.k1_preflight
    else:
        expected_pair = (parent, workflow.checkpoint(entry["k20_preflight"], 8))
        preflight = workflow.k20_preflight
    evidence_path = root / "checks" / f"w512_{stage}_preflight.json"
    if learned:
        preflight(root, entry, 0)
        evidence = workflow.read(evidence_path)
        assert evidence["passed"] is True
        assert evidence["parameter_updates"] == {"passed": True}
    else:
        with pytest.raises(ValueError, match="did not update"):
            preflight(root, entry, 0)
        assert not evidence_path.exists()
    assert checked == [expected_pair]
    if stage == "k1":
        assert len(commands) == 2
        assert "--resume" not in commands[0][-1]
        resumed = commands[1][-1]
        assert resumed[resumed.index("--resume") + 1] == workflow.checkpoint(entry["k1_preflight"], 2)
        assert "--validate" in resumed
