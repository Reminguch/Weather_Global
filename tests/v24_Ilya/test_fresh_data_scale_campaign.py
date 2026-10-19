import json
from pathlib import Path

import pytest

from scripts.experiments import run_v24_fresh_data_scale as campaign


def manifest(tmp_path):
    for name in ('intents', 'reports', 'logs'):
        (tmp_path / name).mkdir()
    value = dict(root=str(tmp_path), workspace=str(tmp_path), source=str(tmp_path / 'source'),
                 workflow_id='test-identity', wait_hours=143,
                 arms={name: {'width': 128 if name.startswith('w128') else 512}
                       for name in campaign.NEW_ARMS})
    campaign.write(tmp_path / 'state.json', dict(automation_enabled=True, jobs={}))
    return value


def test_three_arms_wait_for_real_cache_and_shared_initialization(tmp_path):
    m = manifest(tmp_path)
    jobs = {key: (spec, deps) for key, spec, deps in campaign.initial_jobs(m)}
    assert jobs['qualify_w512_7y'][1] == []
    assert jobs['qualify_w128_40y'][1] == ['cache_verify']
    assert jobs['qualify_w512_40y'][1] == ['cache_verify', 'qualify_w512_7y']
    assert jobs['cache_generate'][0]['gpu'] is None  # import existing predictions
    assert jobs['cache_generate'][0]['array'] == 40
    assert not any('w128_7y' in key for key in jobs)


def test_submit_is_idempotent_and_conflicting_identity_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    m = manifest(tmp_path)
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'run', lambda args, **kw:
                        (calls.append(args) or SimpleNamespace(returncode=0, stdout='12345\n', stderr='')))
    spec = campaign.job('qualify', arm='w512_7y', gpu='gpu80&nomig')
    assert campaign.submit_job(m, 'qualify', spec) == '12345'
    assert campaign.submit_job(m, 'qualify', spec) == '12345'
    assert len(calls) == 1
    with pytest.raises(ValueError, match='Conflicting'):
        campaign.submit_job(m, 'qualify', campaign.job('different'))
    assert not (tmp_path / 'intents/qualify.json').exists()


def test_ambiguous_submission_is_not_repeated(tmp_path, monkeypatch):
    from types import SimpleNamespace
    m = manifest(tmp_path)
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'run', lambda args, **kw:
                        (calls.append(args) or SimpleNamespace(returncode=0, stdout='unknown', stderr='')))
    with pytest.raises(RuntimeError, match='Unrecognized'):
        campaign.submit_job(m, 'stage', campaign.job('stage'))
    with pytest.raises(RuntimeError, match='Unresolved'):
        campaign.submit_job(m, 'stage', campaign.job('stage'))
    assert len(calls) == 1


def test_failure_proof_cannot_release_downstream(tmp_path):
    m = manifest(tmp_path)
    record = dict(workflow_id=m['workflow_id'], result={'passed': False})
    record['sha256'] = campaign.digest(record)
    campaign.write(tmp_path / 'reports/qualify.json', record)
    with pytest.raises(ValueError, match='Failed prerequisite'):
        campaign.require_result(m, 'qualify')


def test_all_final_models_gate_2023(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    jobs = {f'eval2022_{arm}': {'job_id': str(i + 100)}
            for i, arm in enumerate(campaign.NEW_ARMS)}
    partial = dict(jobs)
    partial.pop('eval2022_w512_40y')
    campaign.write(tmp_path / 'state.json', {'jobs': partial})
    calls = []
    monkeypatch.setattr(campaign, 'submit_job', lambda *args: calls.append(args))
    campaign.maybe_freeze_finals(m)
    assert not calls
    campaign.write(tmp_path / 'state.json', {'jobs': jobs})
    campaign.maybe_freeze_finals(m)
    assert calls[0][1] == 'freeze_finals'
    assert calls[0][3] == ['100', '101', '102']


def test_qualification_releases_exact_resume_chain(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    campaign.write(tmp_path / 'state.json', {'automation_enabled': True,
        'jobs': {'qualify_w512_7y': {'job_id': '50'}}})
    calls = []
    def submit(m, key, spec, dependencies=()):
        calls.append((key, spec, list(dependencies)))
        return str(100 + len(calls))
    monkeypatch.setattr(campaign, 'submit_job', submit)
    campaign.advance(m, 'qualify_w512_7y', campaign.job('qualify', arm='w512_7y'),
        {'benchmark': {'peak_host_gib': 60, 'p90_end_to_end_seconds': 10, 'validation_seconds': 500}})
    assert [row[1]['stop'] for row in calls[:5]] == [4000, 8000, 12000, 16000, 20000]
    assert calls[0][2] == ['50']
    assert all(calls[i][2] == [str(100 + i)] for i in range(1, len(calls)))
    assert calls[-1][1]['stage'] == 'eval2022'


def test_sbatch_dependencies_and_no_array_throttle(tmp_path):
    m = manifest(tmp_path)
    args = campaign.sbatch_args(m, 'cache', campaign.job('cache', array=40), ['1', '2'])
    assert '--array=0-39' in args and '--dependency=afterok:1:2' in args
    assert '--partition=cpu' in args
    with pytest.raises(ValueError, match='dependency'):
        campaign.sbatch_args(m, 'bad', campaign.job('bad'), ['1:afterany:2'])


def test_resource_budget_rejects_nonfinite_measurements():
    with pytest.raises(ValueError, match='resource'):
        campaign.training_budget({'benchmark': {'peak_host_gib': float('nan')}}, 512)


@pytest.mark.parametrize('minutes,qos', [(240, 'gpu-short'), (1440, 'gpu-short'),
                                      (1441, 'gpu-medium'), (4321, 'gpu-long')])
def test_della_assigns_gpu_partition_with_valid_qos(tmp_path, minutes, qos):
    args = campaign.sbatch_args(manifest(tmp_path), 'gpu',
                               campaign.job('gpu', gpu='gpu80&nomig', minutes=minutes))
    assert not any(v.startswith('--partition=') for v in args)
    assert f'--qos={qos}' in args


def test_merge_rejects_changed_completed_shard(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    m['paths'] = {}
    m['arms']['w512_7y'].update(name='w512_7y', root=str(tmp_path/'arm'))
    f = tmp_path/'arm/evaluation/2023_shard_00.json'
    campaign.write(f, {'changed': True})
    monkeypatch.setattr(campaign, 'require_result', lambda *args:
                        {'output': {'path': str(f), 'sha256': 'original'}})
    with pytest.raises(ValueError, match='shard changed'):
        campaign.execute_stage(m, 'merge', campaign.job('merge', arm='w512_7y'), None)
