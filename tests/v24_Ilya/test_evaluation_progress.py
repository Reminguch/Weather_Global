from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.analyze_models.merge_v24_Ilya_eval_shards import (
    merge_v24_Ilya_eval_shards,
)
from src.models.mamba.v24_Ilya.evaluation import (
    build_evaluation_result,
    partial_result_path,
    write_evaluation_snapshot,
)
from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
from src.models.mamba.v24_Ilya import data as evaluation_data
from src.models.mamba.v24_Ilya.data import EvalDataset, valid_scored_eval_indices
from src.models.mamba.v24_Ilya.metrics import V24IlyaMetricAccumulator


def test_partial_result_path_is_a_distinct_json_sibling(tmp_path) -> None:
    output = tmp_path / "step200.json"
    assert partial_result_path(output) == tmp_path / "step200.partial.json"
    assert partial_result_path(tmp_path / "result") == tmp_path / "result.partial.json"


def test_atomic_partial_snapshot_exposes_progress_and_metrics(tmp_path) -> None:
    path = tmp_path / "evaluation.partial.json"
    base_output = {
        "architecture_id": "v24_Ilya",
        "n_samples": 3,
    }
    metrics = {
        "original_graphcast_loss": {"improvement_pct_rollout": 1.25},
        "per_variable_per_step": {
            "2m_temperature": {"rmse_full": [1.0]},
        },
    }
    written = write_evaluation_snapshot(
        path,
        base_output=base_output,
        metric_output=metrics,
        chosen_indices=[2, 5, 8],
        completed_samples=2,
        status="partial",
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["evaluation_status"] == "partial"
    assert loaded["evaluated_samples"] == 2
    assert loaded["remaining_samples"] == 1
    assert loaded["chosen_idx"] == [2, 5, 8]
    assert loaded["completed_chosen_idx"] == [2, 5]
    assert loaded["metrics_available"] is True
    assert loaded["metrics_complete"] is False
    assert loaded["intermediate_metrics_omitted"] == [
        "rmsb_baseline",
        "rmsb_full",
        "improvement_pct_rmsb",
    ]
    assert loaded["per_variable_per_step"]["2m_temperature"]["rmse_full"] == [1.0]
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_complete_result_requires_all_samples_and_marks_full_metrics() -> None:
    with pytest.raises(ValueError, match="must include every selected sample"):
        build_evaluation_result(
            base_output={},
            metric_output={"metric": 1.0},
            chosen_indices=[1, 2],
            completed_samples=1,
            status="complete",
        )

    output = build_evaluation_result(
        base_output={},
        metric_output={"metric": 1.0},
        chosen_indices=[1, 2],
        completed_samples=2,
        status="complete",
    )
    assert output["evaluation_status"] == "complete"
    assert output["metrics_complete"] is True
    assert output["intermediate_metrics_omitted"] == []
    assert output["completed_chosen_idx"] == [1, 2]


def test_eval_config_validates_anchor_shards(tmp_path) -> None:
    common = {
        "ckpt": tmp_path / "checkpoint.pkl",
        "out_json": tmp_path / "output.json",
        "eval_mode": "warm_full",
        "anchor_shard_count": 4,
    }
    config = V24IlyaEvalConfig(**common, anchor_shard_index=3)
    assert config.anchor_shard_count == 4
    assert config.anchor_shard_index == 3

    with pytest.raises(ValueError, match="anchor_shard_index"):
        V24IlyaEvalConfig(**common, anchor_shard_index=4)
    with pytest.raises(ValueError, match="force_idx cannot"):
        V24IlyaEvalConfig(**common, anchor_shard_index=0, force_idx=2)


def test_matched_warmup_history_selects_common_scored_dates(tmp_path) -> None:
    common = {
        "ckpt": tmp_path / "checkpoint.pkl",
        "out_json": tmp_path / "output.json",
        "eval_mode": "warm_full",
        "anchor_history_steps": 60,
    }
    for warmup_steps in (20, 40, 60):
        config = V24IlyaEvalConfig(**common, warmup_steps=warmup_steps)
        assert config.effective_anchor_history_steps == 60
    with pytest.raises(ValueError, match="at least the effective warmup"):
        V24IlyaEvalConfig(**common, warmup_steps=61)

    dataset = xr.Dataset(coords={"time": np.arange(164), "lat": [0.0]})
    eval_data = EvalDataset(
        dataset=dataset,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    indices = valid_scored_eval_indices(
        eval_data,
        history_steps=60,
        target_steps=40,
    )
    np.testing.assert_array_equal(indices, np.arange(61, 124))


def test_eval_step_iterator_loads_bounded_blocks(monkeypatch) -> None:
    calls = []

    def fake_build(eval_data, *, indices, target_steps, task_config):
        del eval_data, task_config
        calls.append((indices[0], target_steps))
        times = np.arange(1, target_steps + 1) * np.timedelta64(6, "h")
        values = np.arange(target_steps, dtype=np.float32)[None, :, None, None]
        targets = xr.Dataset(
            {"x": (("batch", "time", "lat", "lon"), values)},
            coords={"batch": [0], "time": times, "lat": [0.0], "lon": [0.0]},
        )
        forcings = xr.Dataset(
            {"f": (("batch", "time"), np.zeros((1, target_steps), np.float32))},
            coords={"batch": [0], "time": times},
        )
        return xr.Dataset(), targets, forcings

    monkeypatch.setattr(evaluation_data, "build_eval_batch", fake_build)
    eval_data = EvalDataset(
        dataset=xr.Dataset(coords={"time": np.arange(30), "lat": [0.0]}),
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    steps = list(
        evaluation_data.iter_eval_steps(
            eval_data,
            final_input_idx=10,
            total_steps=10,
            block_steps=4,
            task_config=object(),
        )
    )
    assert calls == [(10, 4), (14, 4), (18, 2)]
    assert len(steps) == 10
    assert [target.time.values[0] for target, _forcing in steps] == [
        np.timedelta64(6 * step, "h") for step in range(1, 11)
    ]


def test_merge_script_reconstructs_exact_metrics_and_removes_states(tmp_path) -> None:
    global_indices = [2, 5, 8, 11]
    shard_json_paths = []
    merge_state_paths = []
    monolithic = V24IlyaMetricAccumulator(
        target_steps=1,
        latitudes=np.asarray([-45.0, 45.0]),
    )

    for shard_index, anchor_index in enumerate(global_indices):
        truth = xr.Dataset(
            {
                "2m_temperature": (
                    ("batch", "time", "lat", "lon"),
                    np.zeros((1, 1, 2, 1), dtype=np.float32),
                ),
            },
            coords={
                "batch": [0],
                "time": [0],
                "lat": [-45.0, 45.0],
                "lon": [0.0],
            },
        )
        baseline = truth + np.asarray(
            [1.0 + shard_index, -0.5 * shard_index],
            dtype=np.float32,
        ).reshape(1, 1, 2, 1)
        full = truth + (baseline - truth) * 0.5
        accumulator = V24IlyaMetricAccumulator(
            target_steps=1,
            latitudes=np.asarray([-45.0, 45.0]),
        )
        accumulator.update(truth, baseline, full)
        monolithic.update(truth, baseline, full)

        state_path = tmp_path / f"shard{shard_index}.merge_state.pkl"
        with state_path.open("wb") as handle:
            pickle.dump(accumulator.export_merge_state(), handle)
        shard_path = tmp_path / f"shard{shard_index}.json"
        base_output = {
            "architecture_id": "v24_Ilya",
            "schema_version": 1,
            "checkpoint_format": "test",
            "resolution": 0.25,
            "seed": 0,
            "eval_mode": "warm_full",
            "warmup_steps": 4,
            "warmup_feedback": "truth",
            "eval_feedback": "full",
            "rs_reset_after_warmup": False,
            "rs_reset_every_step": False,
            "baseline_branch": "pure_baseline_self_rollout",
            "residual_state_init": "zero",
            "residual_state_init_requested": "zero",
            "residual_state_init_resolved": "zero",
            "target_steps": 1,
            "sample_total_steps": 5,
            "n_samples": 4,
            "anchor_shard_count": 4,
            "anchor_shard_index": shard_index,
            "global_chosen_idx": global_indices,
            "merge_state_path": str(state_path),
            "ckpt": "checkpoint.pkl",
            "ckpt_in": "baseline.npz",
        }
        write_evaluation_snapshot(
            shard_path,
            base_output=base_output,
            metric_output=accumulator.finalize(),
            chosen_indices=[anchor_index],
            completed_samples=1,
            status="complete",
        )
        shard_json_paths.append(shard_path)
        merge_state_paths.append(state_path)

    output_path = tmp_path / "merged.json"
    output = merge_v24_Ilya_eval_shards(
        shard_json_paths=shard_json_paths,
        merge_state_paths=merge_state_paths,
        out_json=output_path,
        delete_merge_states=True,
    )
    expected = monolithic.finalize()
    np.testing.assert_allclose(
        output["per_variable_per_step"]["2m_temperature"]["rmsb_full"],
        expected["per_variable_per_step"]["2m_temperature"]["rmsb_full"],
    )
    np.testing.assert_allclose(
        output["per_variable_per_step"]["2m_temperature"]["rmse_full"],
        expected["per_variable_per_step"]["2m_temperature"]["rmse_full"],
    )
    assert output["chosen_idx"] == global_indices
    assert output["merged_anchor_shards"] == [0, 1, 2, 3]
    assert output["evaluation_status"] == "complete"
    assert output_path.exists()
    assert not any(path.exists() for path in merge_state_paths)
