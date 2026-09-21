"""Exact restart comparisons and realistic torn-log recovery."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('resume_smoke', Path(__file__).resolve().parents[2] /
                                            'scripts/training/smoke_neuralgcm_resume.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def rows():
    return [{'update': i, 'loss': 1/i, 'origin': f'frame-{i}', 'warm': bool(i%2),
             'chunk': i%4, 'seconds': 0.2} for i in range(1, 5)]


def test_wall_clock_differences_are_not_numerical_differences():
    a, b = rows(), rows()
    for row in b: row['seconds'] = 99
    smoke.compare_metrics(a, b, 4)


@pytest.mark.parametrize('fault', ['loss', 'origin', 'warm', 'chunk', 'missing', 'duplicate'])
def test_reject_numerical_sampling_and_replay_mismatches(fault):
    a, b = rows(), rows()
    if fault == 'loss': b[2]['loss'] += 1e-10
    elif fault == 'origin': b[2]['origin'] = 'different-frame'
    elif fault == 'warm': b[2]['warm'] = not b[2]['warm']
    elif fault == 'chunk': b[2]['chunk'] = 0
    elif fault == 'missing': b.pop()
    elif fault == 'duplicate': b[2]['update'] = 2
    with pytest.raises(AssertionError): smoke.compare_metrics(a, b, 4)


def test_real_checkpoint_log_reconciliation(tmp_path):
    from src.models.neuralgcm_residual.checkpoint import reconcile_metrics
    reference, resumed = tmp_path/'reference', tmp_path/'resumed'
    reference.mkdir()
    (reference/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows()))
    tail = smoke.seed_interrupted_log(reference, resumed, 2)
    reconcile_metrics(resumed/'metrics.jsonl', 2)
    assert smoke.metrics(resumed/'metrics.jsonl') == rows()[:2]
    assert (resumed/'metrics.jsonl.after_2.archive').read_text() == tail
