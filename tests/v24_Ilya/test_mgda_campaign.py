"""Bounded MGDA workflow tests; no production models or Slurm submissions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v24_mgda as campaign


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def calibration():
    return {'version': 1, 'fields': ['T2m', 'U10', 'V10', 'MSLP', 'TP6', 'T850', 'Q700', 'U850', 'V850'],
            'wavenumbers': list(range(1, 181)), 'floors': [[.001] * 180 for _ in range(9)],
            'gc_scale': 2., 'power_scale': .3}


def test_production_and_smoke_configs_preserve_matched_contract(tmp_path):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    template = campaign.read(campaign.ROOT / campaign.PRIOR / 'configs/mse.json')
    before = copy.deepcopy(template)
    configs = {}
    for arm in campaign.ARMS:
        for smoke in (False, True):
            config = campaign.make_config(template, tmp_path, arm, calibration(), smoke=smoke)
            path = put(tmp_path / f'{arm}_{smoke}.json', config)
            parsed = load_training_config(path)
            assert parsed.max_steps == (100 if smoke else 2000)
            assert parsed.checkpoint_every == (50 if smoke else 100)
            assert parsed.decay_end_step == 1000 and parsed.warmup_steps == 100
            assert parsed.weight_decay == 0 and parsed.seed == 22
            assert parsed.optimizer_kind == ('mgda_rms' if arm == 'mgda' else 'rms_mse')
            assert parsed.power_calibration['floors'] == calibration()['floors']
            configs[arm, smoke] = config
    for key in ('architecture', 'data', 'distributed', 'sequence', 'objective'):
        assert configs['mgda', False][key] == configs['mse', False][key]
    assert template == before


def test_eleven_tasks_pair_same_steps_and_swa(tmp_path):
    manifest = {'initialization': {'path': '/weights/init.pkl'}, 'initial_source_run_dir': '/weights'}
    tasks = campaign.tasks_for(manifest, tmp_path)
    assert len(tasks) == 11 and len({x['id'] for x in tasks}) == 11
    assert tasks[0]['id'] == 'mature_mse'
    by_id = {x['id']: x for x in tasks}
    for task in tasks[1:]:
        assert task['seed'] == 22
        assert task['endpoint'] == ('primary' if task['id'].endswith('_swa') else
                                    'secondary' if task['step'] == 2000 else 'diagnostic')
        if task['family'] == 'mgda':
            paired = by_id[task['paired_mse_id']]
            assert paired['family'] == 'mse' and paired['step'] == task['step']
    assert '01100-02000' in by_id['mgda_swa']['checkpoint']


def test_budget_is_arm_specific_measured_and_bounded():
    proof = {'passed': True, 'completed_steps': 100, 'peak_rss_gib': 240., 'steady_p90_seconds': 50.}
    budget = campaign.resource_budget(proof, 234.3)
    assert budget == {'memory_gib': 336, 'minutes': 2280}
    assert campaign.resource_budget({**proof, 'steady_p90_seconds': 30.}, 234.3)['minutes'] < budget['minutes']
    for changes in ({'passed': False}, {'completed_steps': 99}, {'peak_rss_gib': float('nan')},
                    {'peak_rss_gib': 400.}, {'steady_p90_seconds': 1000.}):
        with pytest.raises(ValueError):
            campaign.resource_budget({**proof, **changes}, 234.3)


def test_sealed_records_and_pins_reject_changes(tmp_path):
    path = tmp_path / 'manifest.json'
    campaign.seal(path, {'seed': 22})
    assert campaign.unseal(path)['seed'] == 22
    campaign.seal(path, {'seed': 22})
    with pytest.raises(ValueError):
        campaign.seal(path, {'seed': 23})
    record = campaign.pinned(path)
    value = campaign.read(path)
    value['seed'] = 23
    campaign.write(path, value)
    with pytest.raises(ValueError):
        campaign.unseal(path)
    with pytest.raises(ValueError):
        campaign.check(record)


def test_dry_run_is_read_only_and_graph_ends_at_report(tmp_path, monkeypatch):
    m = {'workspace': str(tmp_path), 'source_root': str(tmp_path / 'source'), 'source_digest': 'hash'}
    monkeypatch.setattr(campaign, 'verify', lambda _: m)
    put(tmp_path / 'manifest.json', m)
    monkeypatch.setattr(campaign, 'submit_once', lambda *a, **k: pytest.fail('dry run submitted'))
    before = list(tmp_path.rglob('*'))
    plan = campaign.dry_run(tmp_path)
    assert list(tmp_path.rglob('*')) == before
    assert plan['submitted'] is False and plan['terminal_stage'] == 'report'
    nodes = {x['id']: x for x in plan['nodes']}
    assert nodes['smoke']['depends_on'] == ['calibrate']
    assert nodes['release']['depends_on'] == ['smoke']
    assert nodes['swa']['depends_on'] == ['train_mgda', 'train_mse']
    assert nodes['report']['depends_on'] == ['evaluate']
    assert '--array=0-10' in nodes['evaluate']['command']
    assert not any('confirm' in name or 'seed23' in name for name in nodes)


def test_submission_idempotence_and_unresolved_intent(tmp_path, monkeypatch):
    from scripts.experiments import run_v24_amse_pair as predecessor
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='12345;cluster\n', stderr='')
    monkeypatch.setattr(predecessor.subprocess, 'run', run)
    args = ['sbatch', '--parsable', 'frozen.slurm']
    assert campaign.submit_once(tmp_path, 'calibrate', args) == '12345'
    assert campaign.submit_once(tmp_path, 'calibrate', args) == '12345'
    assert len(calls) == 1
    put(tmp_path / 'submissions/unknown.json', {'command': args})
    with pytest.raises(RuntimeError, match='Unresolved'):
        campaign.submit_once(tmp_path, 'unknown', args)
    assert len(calls) == 1


def test_sbatch_uses_frozen_entrypoint_and_unthrottled_dependencies(tmp_path):
    m = {'workspace': str(tmp_path), 'source_root': str(tmp_path / 'source')}
    args = campaign.sbatch(tmp_path, m, 'swa', memory=16, minutes=20, array='0-1', dependency='10:11')
    assert '--dependency=afterok:10:11' in args and '--array=0-1' in args
    assert args[-3:] == [str(tmp_path / 'source/scripts/experiments/v24_mgda.slurm'), 'swa', str(tmp_path)]
    with pytest.raises(ValueError):
        campaign.sbatch(tmp_path, m, 'smoke', memory=384, minutes=180, array='0-1%1')
    with pytest.raises(ValueError):
        campaign.sbatch(tmp_path, m, 'train', memory=513, minutes=100)


def test_evaluation_attempt_isolated_until_validated(tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    monkeypatch.setattr(campaign, 'verify', lambda *a, **kw: {'source_root': str(tmp_path)})
    monkeypatch.setattr(campaign, 'gpu_required', lambda: None)
    manifest = {'task_specs': [{'id': 'mgda_swa'}], 'output_dir': str(tmp_path / 'evaluation')}
    monkeypatch.setattr(evaluator, 'load_manifest', lambda _: manifest)
    original = evaluator.evaluation_directory
    seen = []
    def worker(path, task, stage):
        directory = evaluator.evaluation_directory(manifest, manifest['task_specs'][task], stage)
        seen.append(directory)
        put(directory / 'evaluation.json', {'status': 'complete'})
    monkeypatch.setattr(evaluator, 'evaluate_worker', worker)
    monkeypatch.setattr(campaign, 'call', lambda args: worker(None, 0, args[-1]))
    monkeypatch.setattr(evaluator, 'resolve_result', lambda p, i, s: (seen[-1], {}, {}, False))
    monkeypatch.setattr(evaluator, 'load_spectral_result', lambda *args: {'global_mse': [1]})
    campaign.evaluate(tmp_path, 0)
    assert all('attempts' in str(path) for path in seen)
    assert evaluator.evaluation_directory is original
    assert (tmp_path / 'evaluation/results/mgda_swa/exact/evaluation.json').exists()
    assert (tmp_path / 'evaluation/results/mgda_swa/spectral/evaluation.json').exists()


def test_failed_evaluation_retains_attempt_and_never_promotes(tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    monkeypatch.setattr(campaign, 'verify', lambda *a, **kw: {'source_root': str(tmp_path)})
    monkeypatch.setattr(campaign, 'gpu_required', lambda: None)
    manifest = {'task_specs': [{'id': 'mgda_swa'}]}
    monkeypatch.setattr(evaluator, 'load_manifest', lambda _: manifest)
    def worker(path, index, stage):
        directory = evaluator.evaluation_directory(manifest, manifest['task_specs'][index], stage)
        directory.mkdir(parents=True)
        raise RuntimeError('GPU failed')
    monkeypatch.setattr(evaluator, 'evaluate_worker', worker)
    monkeypatch.setattr(campaign, 'call', lambda args: worker(None, 0, args[-1]))
    with pytest.raises(RuntimeError, match='GPU failed'):
        campaign.evaluate(tmp_path, 0)
    assert not (tmp_path / 'evaluation/results/mgda_swa').exists()
    with pytest.raises(FileExistsError):
        campaign.evaluate(tmp_path, 0)


def test_preparation_freezes_prescribed_init_data_and_source_without_selection(tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    workspace = tmp_path / 'workspace'
    template = campaign.read(campaign.ROOT / campaign.PRIOR / 'configs/mse.json')
    for directory in ('src', 'scripts', 'third_party/graphcast/graphcast'):
        put(workspace / directory / 'example.py', {'example': 1})
    put(workspace / 'third_party/graphcast/graphcast/graphcast.py', {'example': 1})
    initialization = put(workspace / campaign.INITIALIZATION, {'weights': 1})
    monkeypatch.setattr(campaign, 'INITIALIZATION_SHA256', campaign.file_sha256(initialization))
    put(workspace / campaign.PRIOR / 'initialization_selection.json',
        {'selected': {'checkpoint_sha256': campaign.file_sha256(initialization)}})
    template['data'].update(prepared_root='data/prepared', anchor_manifest_root='data/anchors',
                             baseline_checkpoint='data/weights.npz', stats_dir='data/stats')
    put(workspace / campaign.PRIOR / 'configs/mse.json', template)
    for relative in ('data/weights.npz', 'data/prepared/metadata.json', 'data/anchors/metadata.json',
                     'data/anchors/anchors/anchor_indices.npy', 'data/anchors/anchors/anchor_times.npy',
                     'data/anchors/anchors/split_train.npy', 'data/anchors/anchors/split_val.npy',
                     'data/stats/stddev_by_level.nc', 'data/stats/mean_by_level.nc', 'data/stats/diffs_stddev_by_level.nc'):
        put(workspace / relative, {'fixture': 1})
    for stage in ('exact', 'spectral'):
        store = workspace / 'data' / stage
        metadata = put(store / '.zmetadata', {'zarr_consolidated_format': 1, 'metadata': {'x': {}}})
        put(workspace / campaign.REFERENCE / 'eval' / stage / 'mse_fp32/swa_step00500-02000/specification.json',
            {'data_path': str(store), 'data_metadata': campaign.pinned(metadata),
             'initialization_times': [f'2023-{month:02d}-15T00:00:00Z' for month in (1, 2, 4, 5, 7, 8, 10, 11)]})
    monkeypatch.setattr(evaluator, 'select_initialization', lambda *a, **kw: pytest.fail('selection repeated'))
    monkeypatch.setattr(campaign, 'submit_once', lambda *a, **kw: pytest.fail('preparation submitted'))
    experiment = workspace / 'artifacts/mgda_test'
    manifest = campaign.prepare(experiment, workspace)
    assert manifest['policy']['seed'] == 22 and manifest['policy']['updates'] == 2000
    assert manifest['policy']['repeat_seeds'] == [] and manifest['policy']['confirmation'] is False
    assert not (experiment / 'submissions').exists()
    assert campaign.dry_run(experiment)['calibration_pending']
    # Later workspace source changes do not change the frozen execution tree.
    put(workspace / 'src/example.py', {'changed': 2})
    campaign.verify(experiment)
    put(experiment / 'source/src/example.py', {'changed': 3})
    with pytest.raises(ValueError, match='source changed'):
        campaign.verify(experiment)
