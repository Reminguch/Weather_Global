"""Continuation requires complete immutable evidence and safe resource rounding."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from scripts.experiments.drive_common_grid_2023 import (
    bounded_submit, digest, find_production_records, immutable_file, merge_submission, object_hash,
    parse_accounting, production_submission, resource_budget, rss_gib, utc_text,
    verify_pilot, verify_staging, wait_staging, write_state,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('rss,elapsed,memory,seconds,walltime', [
    (7.1, 120, 16, 4500, '01:15:00'),
    (10.7, 400, 24, 4500, '01:15:00'),
    (60.6, 600, 96, 7200, '02:00:00'),
    (80, 9000, 120, 100800, '1-04:00:00'),
])
def test_gc025_resources_round_up_with_cpu_and_walltime_headroom(rss, elapsed, memory, seconds, walltime):
    result = resource_budget('gc025', {'peak_rss_gib': rss, 'elapsed_seconds': elapsed})
    assert result == {'shards': 16, 'mem_gib': memory, 'walltime_seconds': seconds, 'walltime': walltime}
    assert result['mem_gib'] >= 1.5 * rss
    assert result['walltime_seconds'] >= 1.4 * elapsed * 8


def test_m1_known_budget_requires_observed_memory_headroom():
    result = resource_budget('m1', {'peak_rss_gib': 8, 'elapsed_seconds': 500})
    assert result['shards'] == 8 and result['mem_gib'] == 16 and result['walltime'] == '01:02:00'
    with pytest.raises(ValueError, match='headroom'):
        resource_budget('m1', {'peak_rss_gib': 13, 'elapsed_seconds': 500})
    with pytest.raises(ValueError, match='Finite positive'):
        resource_budget('gc025', {'peak_rss_gib': float('nan'), 'elapsed_seconds': 500})


def test_accounting_uses_complete_array_allocation_and_batch_memory():
    output = '123_0|456|COMPLETED|0:0|500|\n123_0.batch|456.batch|COMPLETED|0:0|500|8388608K\n123_0.extern|456.extern|COMPLETED|0:0|500|\n'
    result = parse_accounting(output, '123')
    assert result['complete'] is True and result['max_rss_gib'] == 8
    assert result['elapsed_seconds'] == 500
    assert result['accepted_job_ids'] == ['123_0', '456']
    assert rss_gib('8192M') == 8 and rss_gib('8G') == 8
    assert parse_accounting('123_0|456|RUNNING|0:0|50|\n', '123')['complete'] is False
    with pytest.raises(RuntimeError, match='failed'):
        parse_accounting('123_0|456|TIMEOUT|0:15|3720|8G\n', '123')
    with pytest.raises(ValueError, match='unexpected array element'):
        parse_accounting('123_1|456|COMPLETED|0:0|50|8G\n', '123')


@pytest.fixture
def staged(tmp_path):
    source, data = tmp_path / 'source', tmp_path / 'data'
    stager = source / 'scripts/preprocessing/stage_common_grid_2023.py'
    helper = source / 'scripts/analyze_models/common_grid_metrics.py'
    for path in (stager, helper):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen fixture source\n')
    starts = [datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=index * 3) for index in range(122)]
    times = sorted({start + timedelta(hours=6 * lead) for start in starts for lead in range(1, 41)})
    names = ['Z500','Z850','T500','T850','T2m','MSL','Q700','Q850','U850','V850','U10','V10']
    protocol = {'source_root': str(source), 'data_root': str(data),
                'initialization_times': list(map(utc_text, starts)), 'lead_hours': list(range(6, 241, 6)),
                'verification': {'diagnostics': names}}
    config = {'initialization_times': protocol['initialization_times'], 'lead_hours': protocol['lead_hours'],
              'diagnostics': dict.fromkeys(names, {}), 'source_uri': 'gs://fixture-source',
              'source_structure_sha256': 'a' * 64, 'static_source_time': '2022-12-31T18:00:00Z',
              'stager_sha256': digest(stager), 'regrid_code_sha256': digest(helper)}
    config_hash = object_hash(config, compact=True)
    manifest = {'configuration': config, 'configuration_sha256': config_hash,
                'expected_inputs025_count': 122, 'expected_common_truth_count': 1492}
    _write(data / 'staging_manifest.json', manifest)
    entries = []
    for kind, stamps in [('inputs025', starts), ('common_truth', times)]:
        for stamp in stamps:
            path = data / kind / (stamp.strftime('%Y%m%dT%H%M%SZ') + '.nc')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{kind}:{stamp}'.encode())
            receipt = {'schema_version': 1, 'kind': kind, 'timestamp': utc_text(stamp), 'path': str(path),
                       'configuration_sha256': config_hash, 'source_uri': config['source_uri'],
                       'source_structure_sha256': config['source_structure_sha256'],
                       'static_source_time': config['static_source_time'], 'size_bytes': path.stat().st_size,
                       'sha256': digest(path)}
            if kind == 'inputs025':
                receipt['input_times'] = [utc_text(stamp - timedelta(hours=6)), utc_text(stamp)]
            _write(Path(str(path) + '.READY.json'), receipt)
            entries.append({key: receipt[key] for key in ('path', 'sha256', 'size_bytes', 'kind', 'timestamp')})
    _write(data / '.READY.json', {'schema_version': 1, 'configuration_sha256': config_hash,
                                'input_count': 122, 'truth_count': 1492,
                                'manifest_sha256': digest(data / 'staging_manifest.json'), 'files': entries})
    _write(data / 'staging_report.json', {'schema_version': 1, 'configuration_sha256': config_hash,
                                        'status': 'complete', 'mode': 'both', 'limit_initializations': None,
                                        'limit_truth_times': None, 'requested_count': 1614,
                                        'verified_existing_count': 41, 'newly_completed_count': 1573})
    return protocol, data


def test_full_staging_checks_all_receipts_and_frozen_code(staged):
    protocol, data = staged
    result = verify_staging(protocol, progress=lambda *args, **kwargs: None)
    assert result['input_count'] == 122 and result['truth_count'] == 1492
    assert len(result['truth_hashes']) == 1492
    first = next((data / 'common_truth').glob('*.nc'))
    first.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='file content differs'):
        verify_staging(protocol, progress=lambda *args, **kwargs: None)


@pytest.mark.parametrize('fault,pattern', [('subset','not completed'), ('missing','missing, duplicate'),
                                         ('code','frozen source'), ('receipt','provenance differs')])
def test_staging_cannot_release_subset_missing_files_or_wrong_provenance(staged, fault, pattern):
    protocol, data = staged
    if fault == 'subset':
        path = data / 'staging_report.json'
        value = json.loads(path.read_text()); value['status'] = 'requested_subset_complete'
        _write(path, value)
    elif fault == 'missing':
        path = data / '.READY.json'; value = json.loads(path.read_text()); value['files'].pop(); _write(path, value)
    elif fault == 'code':
        Path(protocol['source_root'], 'scripts/analyze_models/common_grid_metrics.py').write_text('changed')
    elif fault == 'receipt':
        path = next((data / 'inputs025').glob('*.READY.json'))
        value = json.loads(path.read_text()); value['configuration_sha256'] = 'f' * 64; _write(path, value)
    with pytest.raises(ValueError, match=pattern):
        verify_staging(protocol, progress=lambda *args, **kwargs: None)


def test_pilot_checks_one_initialization_matching_truth_job_and_finite_scores(tmp_path):
    names = ['Z500','Z850','T500','T850','T2m','MSL','Q700','Q850','U850','V850','U10','V10']
    start = '2023-01-01T00:00:00Z'
    protocol = {'initialization_times': [start], 'lead_hours': list(range(6, 241, 6)),
                'verification': {'diagnostics': names}, 'protocol_hash': 'p', 'original_protocol_hash': 'o',
                'gc025_checkpoint_sha256': 'c'}
    _write(tmp_path / 'protocol.json', protocol)
    _write(tmp_path / 'initializations_smoke.json', {'initializations': [start]})
    valid = [utc_text(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(hours=lead)) for lead in protocol['lead_hours']]
    truth = dict.fromkeys(valid, 'a' * 64)
    result = {'schema_version': 1, 'status': 'complete', 'model': 'gc025', 'protocol_hash': 'p',
              'protocol_sha256': digest(tmp_path / 'protocol.json'), 'original_protocol_hash': 'o',
              'initialization_manifest_sha256': digest(tmp_path / 'initializations_smoke.json'),
              'shard_index': 0, 'shard_count': 1, 'slurm_job_id': '456', 'elapsed_seconds': 420,
              'peak_rss_gib': 7, 'evidence': {'checkpoint_sha256': 'c'},
              'per_initialization': [{'initialization_time': start, 'lead_hours': protocol['lead_hours'],
                                      'valid_times': valid, 'truth_sha256': ['a' * 64] * 40,
                                      'mse': {'gc025': {name: [1.] * 40 for name in names}}}]}
    pilot = tmp_path / 'pilot/gc025_00.json'; _write(pilot, result)
    account = {'accepted_job_ids': ['123_0', '456'], 'max_rss_gib': 8, 'elapsed_seconds': 500}
    evidence = verify_pilot(tmp_path, protocol, 'gc025', '123', account, truth)
    assert evidence['peak_rss_gib'] == 8 and evidence['elapsed_seconds'] == 500
    result['per_initialization'][0]['mse']['gc025']['Z500'][0] = float('nan'); _write(pilot, result)
    with pytest.raises(ValueError, match='invalid scores'):
        verify_pilot(tmp_path, protocol, 'gc025', '123', account, truth)


def test_immutable_control_files_refuse_replacement(tmp_path):
    path = tmp_path / 'controller.py'
    immutable_file(path, b'fixed')
    immutable_file(path, b'fixed')
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match='differs'):
        immutable_file(path, b'changed')


def test_existing_production_ledger_is_adopted_without_resubmitting(tmp_path, monkeypatch):
    record = {'phase': 'production', 'model': 'm1', 'job_id': '123', 'shards': 8, 'mem': '16G', 'time': '01:02:00'}
    _write(tmp_path / 'submissions/123.json', record)
    state, path = {'operations': {}}, tmp_path / 'state.json'
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.bounded_submit', lambda *a, **k: pytest.fail('duplicate submission'))
    budget = {'shards': 8, 'mem_gib': 16, 'walltime': '01:02:00'}
    assert production_submission(tmp_path, {}, 'm1', budget, state, path) == '123'
    assert state['operations']['m1']['status'] == 'submitted'
    record['job_id'] = '124'; _write(tmp_path / 'submissions/124.json', record)
    with pytest.raises(RuntimeError, match='Multiple existing'):
        production_submission(tmp_path, {}, 'm1', budget, state, path)


def test_ambiguous_submission_intent_stops_without_resubmission(tmp_path, monkeypatch):
    state = {'operations': {'gc025': {'status': 'submitting'}}}
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.bounded_submit', lambda *a, **k: pytest.fail('duplicate submission'))
    with pytest.raises(RuntimeError, match='Ambiguous earlier'):
        production_submission(tmp_path, {}, 'gc025', {}, state, tmp_path / 'state.json')


def test_merge_depends_on_both_arrays_and_reuses_its_saved_job(tmp_path, monkeypatch):
    seen = []
    def submit(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout='789\n', stderr='')
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.bounded_submit', submit)
    state, path = {'operations': {}}, tmp_path / 'state.json'
    jobs, protocol = {'m1': '123', 'gc025': '456'}, {'workspace': str(tmp_path)}
    assert merge_submission(tmp_path, protocol, jobs, state, path) == '789'
    assert '--dependency=afterok:123:456' in seen[0] and '--kill-on-invalid-dep=yes' in seen[0]
    assert merge_submission(tmp_path, protocol, jobs, state, path) == '789'
    assert len(seen) == 1


def test_scheduler_timeout_terminates_local_process_group_and_preserves_ambiguity(monkeypatch):
    signals, calls = [], []
    class Process:
        pid = 987
        returncode = -15
        def communicate(self, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired('sbatch', timeout)
            return 'partial scheduler output', 'network stalled'
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.subprocess.Popen', lambda *a, **k: Process())
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.os.killpg', lambda pid, sig: signals.append((pid, sig)))
    result = bounded_submit(['sbatch', 'script'], cwd='.', timeout=120)
    assert result.returncode == 124 and 'ambiguous' in result.stderr
    assert result.stdout == 'partial scheduler output'
    assert signals == [(987, signal.SIGTERM)]
    assert calls == [120, 10]


def test_staging_wait_accepts_complete_report_after_process_exits(tmp_path, monkeypatch):
    protocol = {'data_root': str(tmp_path), 'source_root': str(tmp_path / 'source')}
    _write(tmp_path / 'staging_report.json', {'status': 'complete'})
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.staging_process_command', lambda pid: None)
    wait_staging(protocol, 123)


@pytest.mark.parametrize('status,pattern', [('failed', 'Staging failed'), ('running', 'exited before')])
def test_staging_wait_rejects_failed_or_incomplete_dead_process(tmp_path, monkeypatch, status, pattern):
    protocol = {'data_root': str(tmp_path), 'source_root': str(tmp_path / 'source')}
    _write(tmp_path / 'staging_report.json', {'status': status, 'error': 'network error'})
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.staging_process_command', lambda pid: None)
    with pytest.raises(RuntimeError, match=pattern):
        wait_staging(protocol, 123)


def test_staging_wait_validates_process_identity_and_polls_every_30_seconds(tmp_path, monkeypatch):
    protocol = {'data_root': str(tmp_path), 'source_root': str(tmp_path / 'source')}
    report = tmp_path / 'staging_report.json'
    _write(report, {'status': 'running', 'newly_completed_count': 5})
    script = str(tmp_path / 'source/scripts/preprocessing/stage_common_grid_2023.py')
    command = ['python', '-u', script, '--output-root', str(tmp_path)]
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.staging_process_command', lambda pid: command)
    sleeps, progress = [], []
    def finish(seconds):
        sleeps.append(seconds)
        _write(report, {'status': 'complete'})
        _write(tmp_path / '.READY.json', {})
    monkeypatch.setattr('scripts.experiments.drive_common_grid_2023.time.sleep', finish)
    wait_staging(protocol, 123, on_wait=progress.append)
    assert sleeps == [30] and progress[0]['newly_completed_count'] == 5
    command[-1] = str(tmp_path / 'other_campaign')
    with pytest.raises(ValueError, match='not this campaign'):
        wait_staging(protocol, 123)
