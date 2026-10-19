"""Submission retries must never silently duplicate allocations."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.experiments import submit_v24_recovery_plan as submit


def setup_plan(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'nodes': [
        {'id': 'diagnostic', 'depends_on': [], 'command': ['sbatch', '--parsable', 'probe.slurm'],
         'workflow': '/frozen/probe', 'source_digest': 'abc'},
        {'id': 'summary', 'depends_on': ['diagnostic'], 'dependency_condition': 'afterany',
         'command': ['sbatch', '--parsable', 'summary.slurm'],
         'workflow': '/frozen/probe', 'source_digest': 'abc'}]}))
    monkeypatch.setattr(sys, 'argv', ['submit', '--plan', str(path), '--ledger', str(tmp_path / 'ledger'), '--submit'])
    return tmp_path / 'ledger/jobs.json'


def test_dependency_ids_and_sanitized_gpu_environment_survive_retry(tmp_path, monkeypatch):
    ledger = setup_plan(tmp_path, monkeypatch)
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    calls = []
    def run(command, **kwargs):
        assert 'JAX_PLATFORMS' not in kwargs['env']
        calls.append(command)
        return SimpleNamespace(stdout=str(100 + len(calls)) + '\n', stderr='')
    monkeypatch.setattr(submit.subprocess, 'run', run)
    submit.main()
    submit.main()
    assert len(calls) == 2
    assert '--dependency=afterany:101' in calls[1]
    assert json.loads(ledger.read_text())['jobs']['summary']['job_id'] == '102'


def test_ambiguous_submission_is_not_retried(tmp_path, monkeypatch):
    ledger = setup_plan(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(submit.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        submit.main()
    with pytest.raises(ValueError, match='unresolved'):
        submit.main()
    assert len(calls) == 1
    assert json.loads(ledger.read_text())['jobs']['diagnostic']['status'] == 'submission_intent'
