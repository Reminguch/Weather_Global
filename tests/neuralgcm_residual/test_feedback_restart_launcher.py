import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest


spec = importlib.util.spec_from_file_location('feedback_restart', Path(__file__).resolve().parents[2] /
    'scripts/experiments/restart_neuralgcm_feedback.py')
restart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restart)


@pytest.mark.parametrize('action,qos,hours', [('smoke', 'gpu-test', '01:00:00'),
                                             ('slice', 'gpu-short', '24:00:00')])
def test_scheduler_contract_and_afterok_dependencies(tmp_path, action, qos, hours):
    manifest = {'source_root': '/frozen/source', 'python': '/env/python', 'source_id': 'source'}
    key, plan = restart.job_plan(tmp_path, manifest, action, 'r2p8_w128_di16',
                                 after=['101', '102'], stage='pretrain')
    batch = plan['sbatch']
    assert batch[batch.index('--qos') + 1] == qos
    assert batch[batch.index('--time') + 1] == hours
    assert '--partition' not in batch
    assert batch[batch.index('--dependency') + 1] == 'afterok:101:102'
    assert plan['afterok'] == ['101', '102']
    assert '/frozen/source/scripts/experiments/restart_neuralgcm_feedback.py' in plan['script']
    assert 'JAX_PLATFORMS=cuda' in plan['script']


def test_all_formal_jobs_depend_on_every_smoke_and_independent_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(restart, 'verify_reuse', lambda *a: None)
    calls = []
    def submit(root, manifest, action, run, **kw):
        calls.append((action, run, kw))
        return str(100 + len(calls))
    monkeypatch.setattr(restart, 'submit_one', submit)
    manifest = {'configs': dict.fromkeys(['a', 'b', 'c', 'd']), 'source_id': 'source',
                'pilot_root': str(tmp_path / 'pilot')}
    restart.submit_all(SimpleNamespace(experiment_root=tmp_path, pilot_job='99'), manifest)
    assert [x[0] for x in calls] == ['smoke'] * 4 + ['slice'] * 4
    assert all(x[2]['after'] == ['99', '101', '102', '103', '104'] for x in calls[4:])


def test_completed_pilot_uses_report_instead_of_expired_slurm_dependency(tmp_path, monkeypatch):
    from src.models.neuralgcm_residual.io import write_json, read_json, sha256
    monkeypatch.setattr(restart, 'verify_reuse', lambda *a: None)
    report = tmp_path / 'pilot/outputs/report.json'
    write_json(report, {'passed': True})
    calls = []
    def submit(root, manifest, action, run, **kw):
        calls.append((action, run, kw))
        return str(100 + len(calls))
    monkeypatch.setattr(restart, 'submit_one', submit)
    manifest = {'configs': dict.fromkeys(['a', 'b', 'c', 'd']), 'source_id': 'source',
                'pilot_root': str(tmp_path / 'pilot')}
    restart.submit_all(SimpleNamespace(experiment_root=tmp_path, pilot_job='99'), manifest)
    assert all(x[2]['after'] == ['101', '102', '103', '104'] for x in calls[4:])
    chain = read_json(tmp_path / 'pipeline/restart_chain.json')
    assert chain['independent_smoke_report_sha256'] == sha256(report)


def test_missing_smoke_evidence_cannot_release_production(tmp_path):
    with pytest.raises(FileNotFoundError):
        restart.validate_smoke_reports(tmp_path, {'configs': {'run': {'sha256': 'config'}}})


def test_shortened_execution_cannot_target_production_root(tmp_path):
    args = SimpleNamespace(experiment_root=tmp_path, smoke_mode=True)
    with pytest.raises(ValueError, match='isolated smoke root'):
        restart.execute(args, {'smoke_only': False})


def test_startup_audit_is_outside_training_output(tmp_path, monkeypatch):
    from src.models.neuralgcm_residual import worker
    manifest = {'source_id': 'source'}
    args = SimpleNamespace(experiment_root=tmp_path, smoke_mode=False,
                           run_id='run', stage='pretrain', resume=None)
    monkeypatch.setattr(restart, 'verify_reuse', lambda *a: None)
    monkeypatch.setattr(restart, 'validate_smoke_reports', lambda *a: {'passed': True})
    def execute(root, m, stage, **kw):
        worker.require_production_gates(root, m, 'res2p8', None)
        assert not (root / 'runs/run/seed22/pretrain').exists()
        assert list((root / 'pipeline/startup/run/pretrain').glob('*.json'))
    monkeypatch.setattr(worker, 'execute', execute)
    restart.execute(args, manifest)


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_completed_slice_requires_receipt_and_chains_only_after_pretrain(tmp_path, monkeypatch, stage):
    from src.models.neuralgcm_residual import worker
    (tmp_path / 'pipeline').mkdir()
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setattr(restart, 'verify_reuse', lambda *a: None)
    monkeypatch.setattr(restart, 'validate_smoke_reports', lambda *a: {})
    monkeypatch.setattr(restart, 'load_module', lambda *a: SimpleNamespace(
        latest_checkpoint=lambda *a: (0, None)))
    child = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw: child.append(command))
    receipts = []
    monkeypatch.setattr(worker, 'require_receipt', lambda *a: receipts.append(a))
    submitted = []
    monkeypatch.setattr(restart, 'submit_one', lambda *a, **kw: submitted.append(kw))
    args = SimpleNamespace(experiment_root=tmp_path, run_id='run', stage=stage, attempt=0)
    restart.train_slice(args, {'source_root': '/frozen', 'python': '/env/python', 'source_id': 'source'})
    assert receipts[0][1] == stage
    assert '--resume' not in child[0]
    assert submitted == ([{'stage': 'finetune', 'after': ['999']}] if stage == 'pretrain' else [])


def test_timed_out_slice_resubmits_only_after_checkpointed_progress(tmp_path, monkeypatch):
    (tmp_path / 'pipeline').mkdir()
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setattr(restart, 'verify_reuse', lambda *a: None)
    monkeypatch.setattr(restart, 'validate_smoke_reports', lambda *a: {})
    checkpoints = iter([(100, Path('/saved/100.pkl')), (200, Path('/saved/200.pkl'))])
    monkeypatch.setattr(restart, 'load_module', lambda *a: SimpleNamespace(
        latest_checkpoint=lambda *a: next(checkpoints)))
    def timeout(command, **kwargs):
        assert command[-2:] == ['--resume', '/saved/100.pkl']
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    submitted = []
    monkeypatch.setattr(restart, 'submit_one', lambda *a, **kw: submitted.append(kw))
    args = SimpleNamespace(experiment_root=tmp_path, run_id='run', stage='pretrain', attempt=1)
    restart.train_slice(args, {'source_root': '/frozen', 'python': '/env/python', 'source_id': 'source'})
    assert submitted == [{'stage': 'pretrain', 'attempt': 2, 'after': ['999']}]
