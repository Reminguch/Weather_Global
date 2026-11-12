"""Protect total-update accounting and inherited evaluation provenance."""
from pathlib import Path

import pytest

from scripts.experiments import prepare_v24_matched_budget_comparison as matched
from scripts.experiments import run_v24_raw_checkpoint_pilot as pilot
from scripts.experiments.run_v24_mgda import pinned, seal


def toy_parent(tmp_path):
    models = []
    for identity, family in (("M1_raw", "gc_mamba"), ("gcm_s22_l000", "gc_mamba"),
                             ("gcft_s22_l000", "full_gc")):
        run = tmp_path / identity
        (run / "checkpoints").mkdir(parents=True)
        (run / "run_config.json").write_text("{}")
        for step in (500, 750, 1000, 1250, 2000):
            (run / "checkpoints" / f"checkpoint_step{step:08d}.pkl").write_text(f"{identity}:{step}")
        models.append({"id": identity, "label": identity, "model_family": family,
                       "step": 2000 if identity == "M1_raw" else 1250,
                       "lambda": None if identity == "M1_raw" else 0,
                       "run_config": pinned(run / "run_config.json"),
                       "checkpoint": pinned(run / "checkpoints" / "checkpoint_step00001250.pkl")})
    return {"models": models, "record_hash": "parent", "source_digest": "source"}


def test_m1_budget_accounts_for_initial_stage_and_new_runs_do_not(tmp_path):
    parent = toy_parent(tmp_path)
    m1 = matched.earlier_model(parent, "M1_raw", "m1_total1000_raw", 1000)
    assert m1["step"] == 500
    assert m1["total_training_updates"] == m1["training_updates"] == 1000
    assert m1["preceding_training_updates"] == 500
    assert Path(m1["checkpoint"]["path"]).name == "checkpoint_step00000500.pkl"
    assert not m1["id"].startswith("M1_")
    m1_later = matched.earlier_model(parent, "M1_raw", "m1_total1250_raw", 1250)
    assert m1_later["step"] == 750
    for identity in ("gcm_s22_l000", "gcft_s22_l000"):
        new = matched.earlier_model(parent, identity, identity + "_total1000", 1000)
        assert new["step"] == new["total_training_updates"] == 1000
        assert new["preceding_training_updates"] == 0
    assert parent["models"][0]["step"] == 2000


def test_unavailable_budget_is_not_silently_substituted(tmp_path):
    with pytest.raises(ValueError, match="Unsupported matched budget"):
        matched.earlier_model(toy_parent(tmp_path), "M1_raw", "m1_1500", 1500)


def test_reused_model_requires_matching_completion_and_preserves_identity(tmp_path):
    parent = toy_parent(tmp_path)
    identity = "gcft_s22_l000"
    model = matched.parent_model(parent, identity)
    reference_root = tmp_path / "reference"
    directory = reference_root / "evaluation" / identity
    directory.mkdir(parents=True)
    for name in ("evaluation.json", "pooled_spectra.npz"):
        (directory / name).write_text(name)
    record = {"status": "complete", "manifest_hash": "parent", "count": 4,
              "checkpoint": model["checkpoint"],
              "evaluation": pinned(directory / "evaluation.json"),
              "pooled_spectra": pinned(directory / "pooled_spectra.npz")}
    seal(directory / "complete.json", record)
    reused, _ = matched.reused_model(parent, {"path": "parent.json", "sha256": "hash"},
                                    reference_root, identity)
    assert reused["id"] == identity
    assert reused["checkpoint"] == model["checkpoint"]
    assert reused["total_training_updates"] == reused["step"] == 1250
    assert reused["reuse"]["manifest_hash"] == parent["record_hash"]
    parent["record_hash"] = "different-campaign"
    with pytest.raises(ValueError, match="Reused completion does not match"):
        matched.reused_model(parent, {}, reference_root, identity)


def test_only_four_missing_forecasts_are_dispatched(tmp_path):
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"),
                "models": [{"reuse": {}}, {"reuse": {}}, {}, {}, {}, {},
                           {"reuse": {}}, {"reuse": {}}]}
    command = pilot.submission_command(tmp_path, manifest, "evaluate")
    assert "--array=2,3,4,5" in command
    assert not any("%" in arg for arg in command if arg.startswith("--array="))
