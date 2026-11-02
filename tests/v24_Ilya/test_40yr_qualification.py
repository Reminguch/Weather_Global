"""Qualification must fail closed and select the actual physical objective."""
import json
from pathlib import Path

import pytest

from scripts.experiments import cached_40yr_qualification as q


def test_exact_selection_uses_aggregate_physical_loss(tmp_path):
    records = [
        {'role': 'fixed_checkpoint', 'step': step, 'loss': legacy,
         'selected_segments': 15, 'available_segments': 15,
         'original_graphcast_loss': {'full_rollout': exact}}
        for step, legacy, exact in [(500, .1, 3.), (1000, .2, 2.), (1500, .01, 2.5)]
    ]
    records += [{'role': 'fixed_checkpoint', 'step': 2000, 'loss': .001,
                 'selected_segments': 1, 'available_segments': 15,
                 'original_graphcast_loss': {'full_rollout': .1}}]
    (tmp_path / 'validation_metrics.jsonl').write_text('\n'.join(map(json.dumps, records)))
    checkpoint = tmp_path / 'checkpoints/checkpoint_step00001000.pkl'
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert q.select_exact_checkpoint(tmp_path)['checkpoint'] == str(checkpoint)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        q.select_exact_checkpoint(tmp_path)


def test_memory_probe_rejects_accumulating_year_mappings():
    stable = [{'VmRSS_gib': value} for value in [10, 12, 12, 12, 13, 13]]
    assert q.bounded_memory_report(stable)['passed']
    growth = [{'VmRSS_gib': value} for value in [10, 12, 15, 19, 23, 26]]
    assert not q.bounded_memory_report(growth)['passed']
    with pytest.raises(ValueError):
        q.bounded_memory_report(stable[:3])


def test_gpu_peak_uses_this_process_only_and_rejects_missing_measurements(monkeypatch):
    assert q.parse_process_gpu_memory('123, 1024\n456, 35000\n123, 2048', 123) == 2
    assert q.parse_process_gpu_memory('123, [N/A]\n456, 35000', 123) is None
    monkeypatch.setattr(q, '_PROCESS_GPU_PEAK', None)
    with pytest.raises(RuntimeError, match='measurements unavailable'):
        q._gpu_peak({})
    assert q._gpu_peak({'gpu_memory': {'peak_bytes_in_use': 8 * 2**30}}) == 8


def test_exact32_dates_are_stable_and_fit_2022():
    value = q.exact_initializations()['initializations']
    assert len(value) == len(set(value)) == 32
    assert value[0] == '2022-01-02T00:00:00Z'
    assert value[-1] == '2022-12-01T12:00:00Z'
    assert value == sorted(value)


def test_eval_cli_carries_fresh_architecture_and_correct_feedback(tmp_path):
    from src.models.mamba.v24_Ilya.config import parse_args
    reference = q.ROOT / 'configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json'
    payload = json.loads(reference.read_text())
    payload['architecture'].update(residual_width=128, residual_initialization='fresh')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(payload))
    campaign = {'configs': {'128': str(path)}, 'paths': {
        'source_zarr': '/data/new.zarr', 'baseline_checkpoint': '/data/baseline.npz', 'stats_dir': '/data/stats'}}
    for mode, feedback in [('cold_bp', False), ('cold_full', True)]:
        result = parse_args(q.evaluation_arguments(campaign, 128, {'checkpoint': '/model/best.pkl'},
                             mode, tmp_path / 'starts.json', tmp_path / 'eval.json'))
        assert result.is_full_feedback == feedback
        assert result.residual_initialization == 'fresh'
        assert result.residual_width == 128
        assert result.target_steps == 40 and result.n_samples == 32
        assert result.stream_block_steps == 4 and result.omit_rms_bias


def test_failed_batch_integrity_cannot_release_production(tmp_path, monkeypatch):
    campaign = {'root': str(tmp_path), 'numerical_policy': 'advisory'}
    def failing(*args):
        raise q.WorkerFailure(1, '/log', 'integrity', 'independent batch lanes mismatch')
    monkeypatch.setattr(q, '_run', failing)
    result = q.execute_training_pilot(campaign, 128, 'full')
    assert result['passed'] is False and result['failure_kind'] == 'integrity'
    assert not (tmp_path / 'qualification/w128_full/production_integrity.json').exists()
    assert json.loads((tmp_path / 'reports/qualification/training_w128_full.json').read_text())['passed'] is False


def test_oom_disqualifies_gpu_without_faking_integrity(tmp_path, monkeypatch):
    campaign = {'root': str(tmp_path), 'numerical_policy': 'advisory'}
    monkeypatch.setattr(q, '_run', lambda *a: (_ for _ in ()).throw(q.WorkerFailure(1, '/log', 'resource', 'out of memory')))
    result = q.execute_cache_pilot(campaign, 'mig')
    assert result['passed'] is False
    assert result['integrity_passed'] is False
    assert result['failure_kind'] == 'resource'


@pytest.mark.parametrize('parity,peak,passed,kind', [
    (True, 20, True, None), (True, 32, False, 'resource'),
    (True, None, False, 'resource'), (False, 20, False, 'integrity'),
])
def test_cache_pilot_requires_parity_and_measured_gpu_headroom(tmp_path, monkeypatch, parity, peak, passed, kind):
    campaign = {'root': str(tmp_path)}
    def run(*args):
        report = {'passed': parity, 'chunks': 4}
        if peak is not None:
            report['peak_gpu_gib'] = peak
        path = tmp_path / 'qualification/cache_full/pilot_report.json'
        path.write_text(json.dumps(report))
    monkeypatch.setattr(q, '_run', run)
    result = q.execute_cache_pilot(campaign, 'full')
    assert result['passed'] is passed
    assert result['failure_kind'] == kind
    assert result['integrity_passed'] is parity


@pytest.mark.parametrize('numerical_policy', ['strict', 'advisory'])
def test_cache_worker_records_runtime_and_preserves_measured_peak_on_resume(tmp_path, monkeypatch, numerical_policy):
    from src.models.mamba.v24_Ilya import baseline_cache
    from src.models.mamba.v24_Ilya.training import cached_diagnostics
    campaign = {'root': str(tmp_path), 'paths': {'cache_config': '/config'},
                'numerical_policy': numerical_policy}
    pilot = tmp_path / 'qualification/cache_full'
    pilot.mkdir(parents=True)
    monkeypatch.setattr(cached_diagnostics, 'require_matched_gpu_environment',
                        lambda policy: {'device': 'NVIDIA A100-SXM4-40GB', 'numerical_policy': policy})
    monkeypatch.setattr(cached_diagnostics, 'memory_snapshot',
                        lambda: {'gpu_memory': {'peak_bytes_in_use': 5 * 2**30}})
    monkeypatch.setattr(q, '_PROCESS_GPU_PEAK', None)
    monkeypatch.setattr(q, 'validate_allocated_gpu', lambda kind: {'is_mig': False})
    monkeypatch.setattr(baseline_cache, 'load_context', lambda path: 'context')
    monkeypatch.setattr(baseline_cache, 'build_manifest', lambda context, shards, chunks: {'chunks': chunks})
    def generate(context, manifest, root, index, **kwargs):
        assert context == 'context' and manifest['chunks'] == 4
        assert kwargs['numerical_policy'] == numerical_policy
        if not (root / 'pilot_report.json').exists():
            (root / 'pilot_report.json').write_text(json.dumps({'passed': True}))
    monkeypatch.setattr(baseline_cache, 'generate_shard', generate)
    q._cache_worker(campaign, 'full')
    report = json.loads((pilot / 'pilot_report.json').read_text())
    assert report['runtime']['numerical_policy'] == 'strict'
    assert report['peak_gpu_gib'] == 5
    report['peak_gpu_gib'] = 25
    (pilot / 'pilot_report.json').write_text(json.dumps(report))
    q._cache_worker(campaign, 'full')
    assert json.loads((pilot / 'pilot_report.json').read_text())['peak_gpu_gib'] == 25


@pytest.mark.parametrize('expected,is_mig,total,name,passed', [
    ('full', False, 39.5, 'NVIDIA A100-SXM4-40GB', True),
    ('mig', True, 39.3, 'NVIDIA A100-SXM4-80GB MIG 3g.40gb', True),
    ('full', True, 39.3, 'NVIDIA A100-SXM4-80GB MIG 3g.40gb', False),
    ('mig', False, 39.5, 'NVIDIA A100-SXM4-40GB', False),
    ('full', False, 79.0, 'NVIDIA A100-SXM4-80GB', False),
    ('mig', True, 19.5, 'NVIDIA A100-SXM4-80GB MIG 2g.20gb', False),
])
def test_qualification_checks_actual_allocated_device(monkeypatch, expected, is_mig, total, name, passed):
    identity = {'is_mig': is_mig, 'total_memory_gib': total, 'name': name}
    monkeypatch.setattr(q, '_cuda_nvml_identity', lambda: (identity, lambda: 2.0))
    monkeypatch.setattr(q, '_ALLOCATED_GPU_MEMORY', None)
    if passed:
        assert q.validate_allocated_gpu(expected) == identity
        assert q._ALLOCATED_GPU_MEMORY() == 2
    else:
        with pytest.raises(RuntimeError, match='Allocated GPU does not match'):
            q.validate_allocated_gpu(expected)
        assert q._ALLOCATED_GPU_MEMORY is None


def test_helper_reports_do_not_overwrite_campaign_envelopes(tmp_path):
    campaign = {'root': str(tmp_path)}
    envelope = tmp_path / 'reports/cache_full.json'
    envelope.parent.mkdir()
    envelope.write_text('{"workflow_id": "pinned"}')
    result = q._report(campaign, 'cache_full', {'passed': True})
    assert json.loads(envelope.read_text()) == {'workflow_id': 'pinned'}
    assert Path(result['report_path']).parent.name == 'qualification'
