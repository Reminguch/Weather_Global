"""Baseline stages must keep the pilot's code and configuration after queueing."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.preprocessing import submit_v24_baseline as workflow


@pytest.fixture
def prepared(tmp_path):
    workspace = tmp_path / 'workspace'
    for directory in ('src', 'scripts/preprocessing', 'scripts/experiments', 'third_party/graphcast/graphcast'):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    (workspace / 'src/__init__.py').write_text('')
    (workspace / 'src/producer_version.py').write_text('VERSION = "pilot-code"\n')
    (workspace / 'third_party/graphcast/graphcast/graphcast.py').write_text('# required vendor\n')
    for relative in ('scripts/experiments/frozen_runtime.py', 'scripts/preprocessing/submit_v24_baseline.py'):
        shutil.copyfile(workflow.ROOT / relative, workspace / relative)
    for name in ('precompute_v24_baseline', 'submit_v24_baseline_after_pilot', 'verify_v24_baseline'):
        shutil.copyfile(workflow.ROOT / f'scripts/experiments/{name}.slurm',
                        workspace / f'scripts/experiments/{name}.slurm')
    (workspace / 'scripts/preprocessing/precompute_v24_baseline.py').write_text('''
import json
from pathlib import Path
import sys
from src.producer_version import VERSION
args = sys.argv[1:]
root = Path(args[args.index('--output-root') + 1])
root.mkdir(parents=True, exist_ok=True)
phase = 'verify' if '--verify' in args else 'pilot' if '--pilot-parity' in args else 'generate'
payload = {'version': VERSION, 'config': json.loads(Path(args[args.index('--config') + 1]).read_text()),
           'cwd': str(Path.cwd()), 'script': __file__, 'args': args}
(root / (phase + '.json')).write_text(json.dumps(payload))
''')
    config = workspace / 'config.json'
    config.write_text(json.dumps({'data': {'prepared_root': 'data/fields', 'anchor_manifest_root': 'data/anchors',
                                        'baseline_checkpoint': 'data/model.npz', 'stats_dir': 'data/stats'},
                                  'output': {'output_root': 'artifacts/train', 'run_name': 'test'}}))
    record = workflow.prepare(workspace, config, Path('artifacts/pilot'), Path('artifacts/production'))
    return workspace, record


def test_pilot_generation_and_verification_keep_same_source_after_live_edits(prepared, monkeypatch):
    workspace, record = prepared
    source, pilot = Path(record['source']), Path(record['pilot_root'])
    monkeypatch.setattr(workflow, 'ROOT', source)
    # Both an imported model module and the subprocess entry point drift while queued.
    (workspace / 'src/producer_version.py').write_text('VERSION = "changed-code"\n')
    (workspace / 'scripts/preprocessing/precompute_v24_baseline.py').write_text('raise RuntimeError("live producer ran")\n')
    (workspace / 'config.json').write_text('{}')
    for phase in ('pilot', 'generate', 'verify'):
        output = pilot if phase == 'pilot' else Path(record['output_root'])
        workflow.run_stage(pilot, phase, Path(record['config']), output, 0)
        result = json.loads((output / (phase + '.json')).read_text())
        assert result['version'] == 'pilot-code'
        assert result['cwd'] == str(workspace)
        assert result['script'] == str(source / 'scripts/preprocessing/precompute_v24_baseline.py')
        assert result['config']['data']['prepared_root'] == str(workspace / 'data/fields')
        assert result['config']['output']['output_root'] == str(workspace / 'artifacts/train')


def test_reprepare_reuses_original_snapshot_and_rejects_different_config(prepared):
    workspace, record = prepared
    (workspace / 'src/producer_version.py').write_text('VERSION = "later-edit"\n')
    assert workflow.prepare(workspace, workspace / 'config.json', Path(record['pilot_root']),
                            Path(record['output_root'])) == record
    assert 'pilot-code' in (Path(record['source']) / 'src/producer_version.py').read_text()
    (workspace / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='another workspace/configuration'):
        workflow.prepare(workspace, workspace / 'config.json', Path(record['pilot_root']), Path(record['output_root']))


@pytest.mark.parametrize('changed', ['config', 'source'])
def test_changed_snapshot_or_pinned_configuration_stops_before_execution(prepared, monkeypatch, changed):
    _, record = prepared
    source, pilot = Path(record['source']), Path(record['pilot_root'])
    monkeypatch.setattr(workflow, 'ROOT', source)
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **k: pytest.fail('launched invalid workflow'))
    path = Path(record['config']) if changed == 'config' else source / 'src/producer_version.py'
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        workflow.run_stage(pilot, 'pilot', Path(record['config']), pilot, 0)


def test_unsnapshotted_pilot_cannot_be_rebound(tmp_path):
    pilot = tmp_path / 'old-pilot'
    pilot.mkdir()
    manifest = pilot / 'manifest.json'
    manifest.write_text('{"compatibility_sha256": "historical"}')
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='old pilot cannot be rebound'):
        workflow.verify_workflow(pilot)
    with pytest.raises(ValueError, match='already contains artifacts'):
        workflow.prepare(tmp_path, tmp_path / 'config.json', pilot, tmp_path / 'production')
    assert manifest.read_bytes() == before
    assert not (pilot / 'source').exists()


def test_submission_pins_both_stages_and_is_idempotent(prepared, monkeypatch):
    workspace, record = prepared
    calls = []
    monkeypatch.setattr(workflow, 'submit', lambda args: calls.append(args) or str(900 + len(calls)))
    first = workflow.submit_workflow(record)
    assert workflow.submit_workflow(record) == first
    assert len(calls) == 2
    assert all(args[0] == f'--chdir={workspace}' for args in calls)
    assert '--dependency=afterok:901' in calls[1]
    for args in calls:
        assert record['config'] in args
        assert record['pilot_root'] in args
        assert any(str(Path(record['source']) / 'scripts/experiments') in value for value in args)
    assert calls[0][-3:] == [str(workspace), record['source'], record['pilot_root']]
    assert calls[1][-2:] == [str(workspace), record['source']]


def test_run_requires_original_source_and_output(prepared, monkeypatch):
    workspace, record = prepared
    pilot = Path(record['pilot_root'])
    with pytest.raises(ValueError, match='source differs'):
        workflow.run_stage(pilot, 'pilot', Path(record['config']), pilot, 0)
    monkeypatch.setattr(workflow, 'ROOT', Path(record['source']))
    with pytest.raises(ValueError, match='Stage output differs'):
        workflow.run_stage(pilot, 'generate', Path(record['config']), workspace / 'other-output', 0)


def test_slurm_workflow_scripts_parse():
    for name in ('precompute_v24_baseline', 'submit_v24_baseline_after_pilot', 'verify_v24_baseline'):
        subprocess.run(['bash', '-n', str(workflow.ROOT / f'scripts/experiments/{name}.slurm')], check=True)


def test_new_workflow_pins_shard_count_and_initial_dependency(prepared, monkeypatch):
    workspace, old = prepared
    pilot = workspace / 'artifacts/forty_shards'
    record = workflow.prepare(workspace, workspace / 'config.json', pilot,
                              workspace / 'artifacts/forty_output', num_shards=40, initial_dependency='123:124')
    assert record['num_shards'] == 40
    calls = []
    monkeypatch.setattr(workflow, 'submit', lambda args: calls.append(args) or str(500 + len(calls)))
    workflow.submit_workflow(record)
    assert '--dependency=afterok:123:124' in calls[0]
    assert '--dependency=afterok:501' in calls[1]
    with pytest.raises(ValueError, match='another workspace/configuration'):
        workflow.prepare(workspace, workspace / 'config.json', pilot,
                         workspace / 'artifacts/forty_output', num_shards=8, initial_dependency='123:124')


@pytest.mark.parametrize('kwargs', [{'num_shards': 0}, {'initial_dependency': '123; sbatch anything'}])
def test_workflow_rejects_invalid_layout_and_dependency(tmp_path, kwargs):
    with pytest.raises(ValueError):
        workflow.prepare(tmp_path, tmp_path / 'config', tmp_path / 'pilot', tmp_path / 'cache', **kwargs)


def test_generation_and_verification_propagate_pinned_num_shards(prepared, monkeypatch):
    workspace, _ = prepared
    record = workflow.prepare(workspace, workspace / 'config.json',
                              workspace / 'large_pilot', workspace / 'large_cache', num_shards=40)
    monkeypatch.setattr(workflow, 'ROOT', Path(record['source']))
    for phase in ('generate', 'verify'):
        workflow.run_stage(Path(record['pilot_root']), phase, Path(record['config']),
                           Path(record['output_root']), 39)
        payload = json.loads((Path(record['output_root']) / f'{phase}.json').read_text())
        arguments = payload['args']
        assert arguments[arguments.index('--num-shards') + 1] == '40'
        if phase == 'generate':
            assert arguments[arguments.index('--shard-index') + 1] == '39'
