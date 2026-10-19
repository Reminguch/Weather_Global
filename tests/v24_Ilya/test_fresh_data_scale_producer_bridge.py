"""Scheduler compatibility must never become a scientific retry operation."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import fresh_data_scale_producer_bridge as bridge
from scripts.experiments import run_v24_cached_40yr as original


@pytest.mark.parametrize('minutes,qos', [(1440, None), (1441, 'gpu-medium'), (4320, 'gpu-medium'),
                                        (4321, 'gpu-long'), (8640, 'gpu-long')])
def test_only_gpu_partition_and_required_qos_change(minutes, qos):
    args = ['sbatch', '--partition=gpu', '--gres=gpu:1', '--constraint=gpu40&nomig', '--time=' + str(minutes),
            '/immutable/source/wrapper.slurm', '/campaign', 'cache_full', '/workspace']
    corrected = bridge.corrected_sbatch_args(args, {'constraint': 'gpu40&nomig', 'time_minutes': minutes})
    expected = [a for a in args if a != '--partition=gpu']
    if qos:
        expected.insert(1, '--qos=' + qos)
    assert corrected == expected
    assert '--partition=gpu' in args


def test_cpu_submission_and_resource_settings_are_unchanged():
    args = ['sbatch', '--partition=cpu', '--mem=64G', '--time=240', 'script']
    assert bridge.corrected_sbatch_args(args, {'constraint': None, 'time_minutes': 240}) == args


@pytest.fixture
def producer(tmp_path, monkeypatch):
    root = tmp_path / 'producer'
    for name in ('intents', 'reports', 'source'):
        (root / name).mkdir(parents=True)
    campaign = dict(root=str(root), source=str(root / 'source'), workspace=str(tmp_path),
                    workflow_id='immutable-workflow', source_digest='immutable-source')
    entry = dict(status='release_failed', job_id='123', attempt=1, ended_at='2026-10-15T10:00:00+00:00',
                 spec=original.spec('prepare_verify'))
    original.write(root / 'state.json', dict(workflow_id=campaign['workflow_id'], automation_enabled=True,
                                            jobs={'prepare_verify': entry}))
    job = original.spec('cache_pilot', constraint='gpu40&nomig', memory=64, minutes=240)
    command = original.sbatch_args(campaign, 'cache_full', job, ['123'])
    record = dict(command=command, error='sbatch: error: This is not allowed. Please remove --partition=gpu',
                  created_at='2026-10-15T10:00:01+00:00')
    original.write(root / 'intents/cache_full_1.rejected.json', record)
    result = {'passed': True}
    calls = []
    def mark(c, key, **fields):
        assert fields.pop('expected_job_id') == '123'
        state = original.read(root / 'state.json')
        state['jobs'][key].update(fields)
        original.write(root / 'state.json', state)
    driver = SimpleNamespace(read=original.read, write=original.write, sha=original.sha,
        digest=original.digest, now=original.now, locked=lambda *a: nullcontext(),
        load_campaign=lambda root: campaign, require_report=lambda *a: result,
        require_cache=lambda *a: {'manifest_sha256': 'fullcache'}, mark_job=mark,
        sbatch_args=original.sbatch_args)
    def advance(c, key, spec, scientific):
        calls.append(driver.sbatch_args(c, 'cache_full', job, ['123']))
    driver.advance = advance
    monkeypatch.setattr(bridge, 'inactive_completed', lambda _: True)
    return driver, campaign, entry, record, result, calls


def test_repair_uses_old_advance_and_restores_old_function(producer):
    driver, campaign, entry, record, result, calls = producer
    original_args = driver.sbatch_args
    proof = bridge.repair_release(driver, campaign, 'prepare_verify')
    assert proof['passed']
    assert driver.sbatch_args is original_args
    assert len(calls) == 1 and '--partition=gpu' not in calls[0]
    assert calls[0][-4:] == record['command'][-4:]
    assert driver.read(Path(campaign['root']) / 'state.json')['jobs']['prepare_verify']['status'] == 'complete'
    assert (Path(campaign['root']) / 'reports/scheduler_compat_prepare_verify_1.json').exists()
    # Completed release is idempotent, with no second successor submission.
    assert bridge.repair_release(driver, campaign, 'prepare_verify') is None
    assert len(calls) == 1


@pytest.mark.parametrize('mutation', ['stale', 'different_parent', 'different_error'])
def test_rejects_unrelated_or_unproven_submission_failures(producer, mutation):
    driver, campaign, entry, record, result, calls = producer
    if mutation == 'stale':
        record['created_at'] = '2026-10-15T09:00:00+00:00'
    elif mutation == 'different_parent':
        record['command'] = [a.replace('afterok:123', 'afterok:999') for a in record['command']]
    else:
        record['error'] = 'Connection dropped; submission result unknown'
    driver.write(Path(campaign['root']) / 'intents/cache_full_1.rejected.json', record)
    with pytest.raises(RuntimeError, match='not a definite current'):
        bridge.repair_release(driver, campaign, 'prepare_verify')
    assert not calls


def test_does_not_advance_queued_or_running_parent(producer, monkeypatch):
    driver, campaign, _, _, _, calls = producer
    monkeypatch.setattr(bridge, 'inactive_completed', lambda _: False)
    assert bridge.repair_release(driver, campaign, 'prepare_verify') is None
    assert not calls


def test_does_not_retry_failed_scientific_report(producer):
    driver, campaign, _, _, result, calls = producer
    result['passed'] = False
    with pytest.raises(ValueError, match='passing scientific report'):
        bridge.repair_release(driver, campaign, 'prepare_verify')
    assert not calls


def test_ambiguous_intent_propagates_without_marking_complete(producer):
    driver, campaign, _, _, _, calls = producer
    old_args = driver.sbatch_args
    def advance(*args):
        raise RuntimeError('Unresolved submission intent')
    driver.advance = advance
    with pytest.raises(RuntimeError, match='Unresolved submission intent'):
        bridge.repair_release(driver, campaign, 'prepare_verify')
    assert driver.sbatch_args is old_args
    assert driver.read(Path(campaign['root']) / 'state.json')['jobs']['prepare_verify']['status'] == 'release_failed'
    assert not calls


def test_accounting_must_show_completed(monkeypatch):
    outputs = iter(['', 'FAILED|\n'])
    monkeypatch.setattr(bridge.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(outputs), stderr='', returncode=0))
    with pytest.raises(RuntimeError, match='did not finish successfully'):
        bridge.inactive_completed('123')


def test_completed_job_aged_out_of_squeue_still_requires_sacct(monkeypatch):
    outputs = iter([SimpleNamespace(stdout='', stderr='slurm_load_jobs error: Invalid job id specified\n', returncode=1),
                    SimpleNamespace(stdout='COMPLETED|\n', stderr='', returncode=0)])
    monkeypatch.setattr(bridge.subprocess, 'run', lambda *a, **k: next(outputs))
    assert bridge.inactive_completed('123')


def test_other_squeue_errors_are_not_treated_as_completed(monkeypatch):
    monkeypatch.setattr(bridge.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='', stderr='Unable to contact slurm controller', returncode=1))
    with pytest.raises(bridge.subprocess.CalledProcessError):
        bridge.inactive_completed('123')


def test_bridge_stops_when_final_report_has_been_submitted(producer, monkeypatch):
    driver, campaign, _, _, _, calls = producer
    root = Path(campaign['root'])
    driver.write(root / 'campaign.json', campaign)
    state = driver.read(root / 'state.json')
    state['jobs']['report'] = {'job_id': '999', 'status': 'submitted', 'spec': {'stage': 'report'}}
    driver.write(root / 'state.json', state)
    monkeypatch.setattr(bridge, 'load_driver', lambda root: (driver, campaign))
    proof = bridge.run(root)
    assert proof['report_job_id'] == '999'
    assert 'scientific completion remains separate' in proof['meaning']
    assert not calls
