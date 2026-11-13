from pathlib import Path

import numpy as np

from scripts.analyze_models.comparison_policy import POLICY, exclusion, partition_models, presentation_directory
from scripts.analyze_models.report_v24_scalar_lambda import paired_contrasts


def model(run, *, family="full_gc", rate=1e-5, campaign=POLICY["campaign"]):
    return {"id": run, "model_family": family, "learning_rate": rate,
            "checkpoint": {"path": f"/archive/{campaign}/runs/{run}/checkpoints/step.pkl"}}


def test_only_rejected_full_gc_training_identities_are_excluded():
    excluded = [model(run) for run in POLICY["run_ids"]] + [model("gcft_s22_l025", rate=3e-7)]
    retained = [model("gcft_s22_l000_lr3em7_fresh_20261107", rate=3e-7),
                model("gcft_s22_l025_lr3em7", rate=3e-7),
                model("gcft_s22_l025", campaign="future_low_lr_sweep"),
                model("gcft_s22_l025", family="gc_mamba"),
                {"id": "gcft_s22_l025", "model_family": "full_gc", "learning_rate": 1e-5}]
    included, rejected = partition_models(excluded + retained)
    assert included == retained
    assert len(rejected) == 6
    assert all(row["policy_id"] == POLICY["id"] for row in rejected)


def test_config_identity_and_campaign_fallback_cover_selected_tasks():
    task = {"id": "gcft_s22_l050_profile", "logical_id": "gcft_s22_l050", "model_family": "full_gc",
            "learning_rate": 1e-5, "config": {"path": f"/archive/{POLICY['campaign']}/configs/gcft_s22_l050_profile.json"}}
    assert exclusion(task)["training_run_id"] == "gcft_s22_l050_profile"
    del task["config"]
    assert exclusion(task, Path("/archive") / POLICY["campaign"])["model_id"] == "gcft_s22_l050"


def test_filtered_outputs_have_a_distinct_idempotent_destination():
    original = Path("/archive/report")
    filtered = presentation_directory(original, [{"model_id": "rejected"}])
    assert filtered != original
    assert presentation_directory(filtered, [{}]) == filtered
    assert presentation_directory(original, []) == original


def test_partial_scalar_comparison_only_pairs_available_lambda_arms():
    records = [
        {"model_family": "gc_mamba", "lambda": 0., "variant": "raw"},
        {"model_family": "full_gc", "lambda": 0., "variant": "raw"},
        {"model_family": "gc_mamba", "lambda": .25, "variant": "raw"},
    ]
    point = np.array([[3.], [4.], [2.]])
    draws = np.repeat(point[None], 5, axis=0)
    contrasts = paired_contrasts(records, point, draws)
    assert len(contrasts) == 1
    assert contrasts[0]["lambda"] == 0.
    assert contrasts[0]["mamba_advantage_percentage_points"] == -1.
