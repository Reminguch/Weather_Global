"""Portable full-GC handoff preserves four fresh trajectories and exact continuation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments import run_v24_portable_full_gc_ar as campaign


def template():
    return {"model_family": "full_gc", "architecture": {"resolution": 1.},
            "data": {"prepared_root": "/old/prepared", "anchor_manifest_root": "/old/anchors",
                     "baseline_checkpoint": "/old/gc.npz", "stats_dir": "/old/stats",
                     "train_order_manifest": "/old/order.json"},
            "sequence": {"ar_tail_k": 20, "bptt_steps": 24, "segment_steps": 96,
                         "feedback_mode": "closed_loop"},
            "distributed": {"per_device_batch_size": 4},
            "objective": {"loss_kind": "gc_mse_amse_excess", "spectral_excess_weight": 0},
            "optimizer": {"max_steps": 2500, "learning_rate": 3e-7,
                          "end_learning_rate": None, "warmup_steps": 0,
                          "learning_rate_schedule": "constant", "seed": 22},
            "output": {"output_root": "/old/runs", "run_name": "old"}}


def test_all_four_horizons_are_fresh_and_differ_only_in_k_and_output(tmp_path):
    original = template()
    paths = {key: tmp_path / key for key in campaign.INPUT_KEYS}
    configs = [campaign.make_config(original, tmp_path, horizon, paths, tmp_path / "order.json")
               for horizon in campaign.HORIZONS]
    assert original == template()
    assert len({cfg["output"]["run_name"] for cfg in configs}) == 4
    assert configs[-1]["output"]["run_name"] == "gcft_s22_k20_lr3em7_constant"
    normalized = []
    for horizon, cfg in zip(campaign.HORIZONS, configs, strict=True):
        assert cfg["sequence"]["ar_tail_k"] == horizon
        assert cfg["optimizer"] == original["optimizer"]
        assert cfg["objective"] == original["objective"]
        assert cfg["data"]["baseline_checkpoint"] == str(paths["baseline_checkpoint"])
        cfg = copy.deepcopy(cfg)
        cfg.pop("output")
        cfg["sequence"].pop("ar_tail_k")
        normalized.append(cfg)
    assert all(cfg == normalized[0] for cfg in normalized)


@pytest.mark.parametrize("field,value", [("learning_rate", 1e-5),
                                        ("learning_rate_schedule", "cosine"),
                                        ("warmup_steps", 100),
                                        ("end_learning_rate", 3e-8)])
def test_nonstandard_template_cannot_silently_change_the_recipe(tmp_path, field, value):
    cfg = template()
    cfg["optimizer"][field] = value
    with pytest.raises(ValueError, match="template differs"):
        campaign.make_config(cfg, tmp_path, 20, {}, tmp_path / "order")


def test_order_relocation_preserves_exact_mapping_and_checks_source_identity(tmp_path):
    roots = {"anchor_manifest_root": tmp_path / "anchors", "prepared_root": tmp_path / "prepared"}
    local = {"anchor_indices": roots["anchor_manifest_root"] / "anchors/anchor_indices.npy",
             "split_train": roots["anchor_manifest_root"] / "anchors/split_train.npy",
             "time": roots["prepared_root"] / "coords/time.npy"}
    for index, path in enumerate(local.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.arange(4) + index)
    original = {"sources": {key: {**campaign.pinned(path), "path": "/old/" + path.name}
                            for key, path in local.items()},
                "segments": [[0, 1]], "epoch_permutations": [[1, 0]],
                "updates": [{"epoch": 0, "segment_ids": [1, 0]}]}
    rebased = campaign.relocate_order(original, roots)
    assert {k: v for k, v in original.items() if k != "sources"} == {
        k: v for k, v in rebased.items() if k != "sources"}
    assert all(rebased["sources"][key] == campaign.pinned(path) for key, path in local.items())
    assert all(value["path"].startswith("/old/") for value in original["sources"].values())
    np.save(local["time"], np.zeros(4))
    with pytest.raises(ValueError, match="input differs: time"):
        campaign.relocate_order(original, roots)


def fake_campaign(tmp_path, monkeypatch):
    task = {"id": "gcft_s22_k20_lr3em7", "horizon": 20}
    config = tmp_path / "config.json"
    config.write_text("{}")
    task["config"] = campaign.pinned(config)
    manifest = {"source_root": str(tmp_path / "source"), "source_digest": "source",
                "record_hash": "manifest", "tasks": [task] * 4,
                "baseline": {"sha256": "baseline"}}
    (tmp_path / "checks").mkdir()
    calls = []
    def execute(command):
        calls.append(command)
        stop = int(command[command.index("--stop-after-steps") + 1])
        run = tmp_path / "runs" / task["id"]
        (run / "checkpoints").mkdir(parents=True, exist_ok=True)
        for step in (0, stop):
            campaign.checkpoint_path(tmp_path, task, step).write_text(str(step))
        (run / "run_config.json").write_text("{}")
        rows = [{"step": step, "loss": step * 1., "gradient_norm": step * 1.}
                for step in range(1, stop + 1)]
        (run / "train_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(campaign, "call", execute)
    monkeypatch.setattr(campaign, "gpu_required", lambda: None)
    monkeypatch.setattr(campaign, "verify_training_checkpoint", lambda *args: None)
    monkeypatch.setattr(campaign, "numeric_initial_identity", lambda path: {"residual_params": "identical"})
    return manifest, task, calls


def test_finite_rising_loss_finishes_both_parts_without_optimizer_reset(tmp_path, monkeypatch):
    manifest, task, calls = fake_campaign(tmp_path, monkeypatch)
    first = campaign.train(tmp_path, manifest, 0, 1)
    assert first["completed_steps"] == first["total_training_updates"] == 1250
    assert "--resume" not in calls[0] and "--init-from" not in calls[0]
    second = campaign.train(tmp_path, manifest, 0, 2)
    assert second["completed_steps"] == second["total_training_updates"] == 2500
    assert second["started_from_step"] == 1250
    assert second["scientific_improvement_required"] is False
    assert calls[1][calls[1].index("--resume") + 1] == str(campaign.checkpoint_path(tmp_path, task, 1250))
    assert "--init-from" not in calls[1] and "--zero-state-on-init" not in calls[1]


def test_modified_midpoint_history_cannot_resume(tmp_path, monkeypatch):
    manifest, task, calls = fake_campaign(tmp_path, monkeypatch)
    campaign.train(tmp_path, manifest, 0, 1)
    with (tmp_path / "runs" / task["id"] / "train_metrics.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Live training history"):
        campaign.train(tmp_path, manifest, 0, 2)
    assert len(calls) == 1


def test_saved_nonfinite_or_wrong_family_is_an_error(tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya import checkpoint as checkpoint_module
    from src.models.mamba.v24_Ilya.training import config as config_module
    config = {"model_family": "full_gc"}
    saved = SimpleNamespace(completed_step=1250, model_family="full_gc",
        baseline_checkpoint_fingerprint="baseline", resolved_training_config=config,
        residual_params={"p": np.array([1.])}, residual_state={}, optimizer_state={})
    monkeypatch.setattr(checkpoint_module, "load_v24_Ilya_training_checkpoint", lambda path: saved)
    monkeypatch.setattr(config_module, "load_training_config", lambda path: SimpleNamespace(to_dict=lambda: config))
    campaign.verify_training_checkpoint(tmp_path / "pkl", tmp_path / "config", "baseline", 1250)
    saved.model_family = "gc_mamba"
    with pytest.raises(ValueError, match="progress/family"):
        campaign.verify_training_checkpoint(tmp_path / "pkl", tmp_path / "config", "baseline", 1250)
    saved.model_family = "full_gc"
    saved.residual_params["p"][0] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        campaign.verify_training_checkpoint(tmp_path / "pkl", tmp_path / "config", "baseline", 1250)


def test_execution_plan_has_four_fresh_horizons_and_no_cluster_submission(tmp_path):
    manifest = {"source_root": str(tmp_path / "source"),
                "tasks": [{"horizon": k} for k in campaign.HORIZONS]}
    plan = campaign.plan(tmp_path, manifest)
    assert [task["horizon"] for task in plan["training"]] == [8, 12, 16, 20]
    assert all(task["part1"][-2] == "1" and task["part2"][-2] == "2" for task in plan["training"])
    assert "sbatch" not in json.dumps(plan)
    wrapper = (campaign.ROOT / campaign.WRAPPER).read_text()
    assert "dabanin" not in wrapper and "gpu80" not in wrapper and "module load" not in wrapper
