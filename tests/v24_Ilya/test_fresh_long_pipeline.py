"""Recovery and matched-evaluation guards for the long fresh run."""
import json
from pathlib import Path

import pytest

from scripts.experiments import run_v24_res1_fresh_long as pipeline


def test_resume_archives_unsaved_metrics_and_partial_last_line(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "train_metrics.jsonl"
    original = ''.join(json.dumps({"step": s, "loss": 1.0}) + '\n' for s in (1, 2, 3)) + '{"step":4'
    path.write_text(original)
    pipeline.trim_training_records(run, 2, tmp_path)
    assert [json.loads(s)["step"] for s in path.read_text().splitlines()] == [1, 2]
    backups = list((tmp_path / "checks/recovery_history").glob("*/train_metrics.jsonl"))
    assert len(backups) == 1 and backups[0].read_text() == original


def test_resume_refuses_missing_committed_history_without_mutation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "train_metrics.jsonl"
    original = '{"step": 1}\n{"step": 3}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.trim_training_records(run, 3, tmp_path)
    assert path.read_text() == original


def test_evaluation_rejects_different_forecast_starts():
    with pytest.raises(ValueError, match="chosen_idx"):
        pipeline.validate_result({"chosen_idx": [2]}, {"chosen_idx": [1]})
