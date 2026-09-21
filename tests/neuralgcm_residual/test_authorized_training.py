import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('authorized_training',
    Path(__file__).resolve().parents[2] / 'scripts/experiments/run_neuralgcm_authorized_training.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def policy_fixture(root):
    manifest = {'source_id': 'source', 'configs': {'run': {'sha256': 'config'}}}
    write(root / 'manifest.json', manifest)
    report = dict(passed=True, source_id='source', config_sha256='config', run_id='run',
        checks={str(i): {'passed': True} for i in range(15)}, exact_checkpoint_equality=True,
        exact_metrics_equality=True, independent_processes=True, interrupted_log_recovery=True)
    evidence = {}
    for name in ('detailed', 'resume'):
        path = root / f'{name}.json'
        write(path, report)
        evidence[name] = dict(path=str(path), sha256=runner.file_hash(path))
    policy = dict(schema='neuralgcm_user_authorized_start_v1', experiment_root=str(root),
        source_id='source', authorization_text='Start training after data preparation',
        runner=dict(path=str(Path(runner.__file__).resolve()), sha256=runner.file_hash(runner.__file__)),
        evidence={'run': evidence}, constraint='a100&gpu80&pcie',
        resources=dict(memory_gb=64, pretrain_hours=24, finetune_hours=24))
    path = root / 'pipeline/authorized_training_policy.json'
    write(path, policy)
    return policy, path


def test_policy_requires_exact_completed_evidence(tmp_path):
    policy, path = policy_fixture(tmp_path)
    runner.validate_policy(tmp_path, runner.file_hash(path))
    evidence_path = Path(policy['evidence']['run']['resume']['path'])
    evidence_path.write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        runner.validate_policy(tmp_path, runner.file_hash(path))


@pytest.mark.parametrize('field', ['passed', 'exact_checkpoint_equality', 'independent_processes'])
def test_newly_pinned_failed_report_still_rejected(tmp_path, field):
    policy, path = policy_fixture(tmp_path)
    evidence = policy['evidence']['run']['resume']
    report = runner.read(evidence['path'])
    report[field] = False
    write(Path(evidence['path']), report)
    evidence['sha256'] = runner.file_hash(evidence['path'])
    write(path, policy)
    with pytest.raises(ValueError):
        runner.validate_policy(tmp_path, runner.file_hash(path))


def test_authorization_hash_cannot_be_silently_changed(tmp_path):
    policy, path = policy_fixture(tmp_path)
    original = runner.file_hash(path)
    policy['authorization_text'] = 'changed'
    write(path, policy)
    with pytest.raises(ValueError, match='policy changed'):
        runner.validate_policy(tmp_path, original)


def test_full_statistics_gate_rejects_pilot_and_changed_ready(tmp_path):
    origins = ['2015-01-02T00', '2015-01-02T06']
    cache = dict(producer_id='producer', shards=[dict(split='train', origins=origins)])
    stats = dict(origin_count=2, origins_sha256=digest(origins), split='train',
                 dataset_id='dataset', native_schema_id='native')
    stats_path, cache_root = tmp_path / 'statistics.json', tmp_path / 'cache'
    write(stats_path, stats)
    write(cache_root / 'manifest.json', cache)
    write(cache_root / 'READY.json', {'passed': True})
    receipt = {'result': dict(statistics_sha256=runner.file_hash(stats_path),
        ready_sha256=runner.file_hash(cache_root / 'READY.json'), producer_id='producer')}
    runtime = SimpleNamespace(store=SimpleNamespace(identity='dataset'),
        adapter=SimpleNamespace(identity='native'), normalization=SimpleNamespace(identity=digest(stats)))
    worker = SimpleNamespace(require_receipt=lambda *a: receipt)
    backend = SimpleNamespace(digest=digest, resolve_resources=lambda *a:
        dict(statistics=str(stats_path), cache_root=str(cache_root)))
    args = (tmp_path, {'source_id': 'source'}, 'res2p8', runtime, worker, backend)
    assert runner.data_gate(*args) == receipt
    stats['origin_count'] = 1
    write(stats_path, stats)
    receipt['result']['statistics_sha256'] = runner.file_hash(stats_path)
    runtime.normalization.identity = digest(stats)
    with pytest.raises(ValueError, match='Full production'):
        runner.data_gate(*args)
    stats['origin_count'] = 2
    write(stats_path, stats)
    receipt['result']['statistics_sha256'] = runner.file_hash(stats_path)
    runtime.normalization.identity = digest(stats)
    write(cache_root / 'READY.json', {'changed': True})
    with pytest.raises(ValueError, match='Full production'):
        runner.data_gate(*args)


@pytest.mark.parametrize('progress', [False, True])
def test_timeout_continues_only_with_committed_progress(tmp_path, monkeypatch, progress):
    checkpoints = iter([(50, tmp_path / 'midpoint.pkl'),
                        (100, tmp_path / 'last.pkl') if progress else (50, tmp_path / 'midpoint.pkl')])
    backend = SimpleNamespace(latest_checkpoint=lambda *a: next(checkpoints))
    def timeout(command, **kwargs):
        assert command[-2:] == ['--resume', str(tmp_path / 'midpoint.pkl')]
        assert kwargs['timeout'] == 85500
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    submitted = []
    monkeypatch.setattr(runner, 'submit_training', lambda *a: submitted.append(a[-4:]))
    args = (tmp_path, {'python': 'python'},
            {'runner': {'path': 'runner.py'}, 'resources': {'pretrain_hours': 24}},
            'policy-hash', backend, 'run', 'pretrain', 0)
    if progress:
        runner.train_slice(*args)
        assert submitted == [('run', 'pretrain', 1, ['123'])]
    else:
        with pytest.raises(RuntimeError, match='no checkpointed progress'):
            runner.train_slice(*args)
        assert submitted == []


def test_successful_pretrain_submits_finetune_after_receipt(tmp_path, monkeypatch):
    backend = SimpleNamespace(latest_checkpoint=lambda *a: (0, None))
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setitem(sys.modules, 'src.models.neuralgcm_residual.worker',
                        SimpleNamespace(require_receipt=lambda *a: calls.append(('receipt', a[1]))))
    monkeypatch.setattr(runner, 'submit_training', lambda *a: calls.append(('submit', a[-4:])))
    runner.train_slice(tmp_path, {'python': 'python', 'source_id': 'source'},
        {'runner': {'path': 'runner.py'}, 'resources': {'pretrain_hours': 24}},
        'policy-hash', backend, 'run', 'pretrain', 0)
    assert calls == [('receipt', 'pretrain'), ('submit', ('run', 'finetune', 0, ['123']))]


def test_jobs_keep_production_resources_and_content_bound_policy(tmp_path):
    policy, _ = policy_fixture(tmp_path)
    key, plan = runner.job_plan(tmp_path, {'python': '/python', 'source_id': 'source',
        'source_root': '/frozen'}, policy, 'policy-hash', 'run', 'pretrain', 0, ['456'])
    assert key == 'authorized-pretrain-run-000'
    assert plan['sbatch'][plan['sbatch'].index('--qos') + 1] == 'gpu-short'
    assert '--partition' not in plan['sbatch']
    assert plan['sbatch'][-1] == 'afterok:456'
    assert '--policy-sha256 policy-hash' in plan['script']
