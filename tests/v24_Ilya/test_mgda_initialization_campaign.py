"""Initialization, exact continuation and bounded submission contracts."""
import copy
from pathlib import Path

import pytest

from scripts.experiments import run_v24_mgda_initialization as campaign
from scripts.experiments.frozen_runtime import freeze_source
from src.models.mamba.v24_Ilya.training.config import load_training_config, validate_resume_config


def prepared_manifest(root, kind):
    return {'kind': kind, 'workspace': str(root), 'source_root': str(root / 'source'),
        'source_digest': 'source', 'record_hash': 'manifest',
        'initial_step': 200 if kind == 'continuation' else 0,
        'stage_endpoints': campaign.endpoints(kind),
        'resources': {'memory': 384, 'steady_p90_seconds': 140., 'startup_allowance_seconds': 1800.}}


@pytest.mark.parametrize('kind', ['continuation', 'fresh'])
def test_configs_keep_shared_physics_and_matched_controls(tmp_path, kind):
    comparable = []
    for arm in campaign.ARMS:
        source = campaign.ROOT / campaign.PARENT / 'configs' / f'{arm}.json'
        template = campaign.read(source)
        original = copy.deepcopy(template)
        output = campaign.make_config(template, tmp_path, arm, kind)
        path = tmp_path / f'{arm}.json'
        campaign.write(path, output)
        config = load_training_config(path)
        assert config.feedback_mode == 'closed_loop'
        assert (config.bptt_steps, config.ar_tail_k, config.segment_steps) == (24, 20, 96)
        assert config.distributed.global_batch_size == 4
        assert config.objective_pair == 'mse_amse_excess' and config.optimizer_kind == 'mgda_rms'
        assert config.mgda_backtracking['decrease_policy'] == config.mgda_backtracking['repeatability_policy'] == 'warn'
        assert config.seed == 22
        if kind == 'continuation':
            validate_resume_config(config, load_training_config(source).to_dict(), completed_step=200)
            assert config.max_steps == 2000 and config.learning_rate == config.end_learning_rate == 1e-6
        else:
            assert config.max_steps == 2500
            assert config.learning_rate == 1e-5 and config.end_learning_rate == 3e-6
            assert config.warmup_steps == 100 and config.decay_end_step == 2000
            assert config.architecture.residual_initialization == 'baseline_overlay'
            assert config.architecture.temporal_init_scheme == 'mamba1'
            assert config.architecture.temporal_zero_init_out
        output.pop('output')
        output['optimizer'].pop('mgda_direction')
        comparable.append(output)
        assert template == original
    assert comparable[0] == comparable[1]


@pytest.mark.parametrize('kind', ['continuation', 'fresh'])
def test_stage_dependency_graph_counts_every_update_once(tmp_path, kind):
    manifest = prepared_manifest(tmp_path, kind)
    plan = campaign.submission_plan(tmp_path, manifest)
    training = [n for n in plan if n['id'].startswith('train_')]
    stops = [n['stop_after_steps'] for n in training]
    assert stops == ([1100, 2000] if kind == 'continuation' else [1250, 2500])
    assert stops[-1] == (2000 if kind == 'continuation' else 2500)
    starts = [manifest['initial_step'], *stops[:-1]]
    assert [end - start for start, end in zip(starts, stops)] == (
        [900, 900] if kind == 'continuation' else [1250, 1250])
    assert sum(end - start for start, end in zip(starts, stops)) == (1800 if kind == 'continuation' else 2500)
    for index, node in enumerate(training):
        assert '--array=0-1' in node['command']
        assert not any('%' in arg for arg in node['command'] if arg.startswith('--array'))
        assert node['depends_on'] == (f'<train_{index - 1:02d}>' if index else None)
        if index:
            assert f'--dependency=aftercorr:<train_{index - 1:02d}>' in node['command']
    assert plan[-1]['depends_on'] == '<evaluate_exact>:<evaluate_spectral>'
    assert all(n['depends_on'] == f'<train_{len(training)-1:02d}>' for n in plan[-3:-1])


def test_snapshot_preserves_every_old_runtime_except_declared_changes(tmp_path):
    workspace = tmp_path / 'workspace'
    old = tmp_path / 'old'
    files = {'src/unchanged.py': 'preserved', campaign.CONFIG_MODULE: 'old config',
             'third_party/graphcast/graphcast/graphcast.py': 'vendor',
             'scripts/existing.py': 'existing'}
    for relative, value in files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    freeze_source(workspace, old)
    for relative in (campaign.CONFIG_MODULE, campaign.DRIVER, campaign.WRAPPER):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('new ' + relative)
    (workspace / 'src/unchanged.py').write_text('unrelated concurrent edit')
    frozen, provenance = campaign.freeze(old, tmp_path / 'snapshot', workspace)
    assert (tmp_path / 'snapshot/src/unchanged.py').read_text() == 'preserved'
    assert set(provenance['changed_files']) == {campaign.CONFIG_MODULE, campaign.DRIVER, campaign.WRAPPER}


def setup_train(tmp_path, monkeypatch, kind):
    manifest = prepared_manifest(tmp_path, kind)
    manifest['initial'] = {}
    arm = 'mgda'
    if kind == 'continuation':
        target = tmp_path / 'initial' / arm
        target.mkdir(parents=True)
        (target / 'checkpoint.pkl').write_bytes(b'checkpoint')
        files = {}
        for name in ('train_metrics.jsonl', 'backtracking_trials.jsonl', 'checkpoint_metrics.jsonl'):
            path = target / name
            path.write_text('history')
            live = tmp_path / 'runs' / arm / name
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_text('history')
            files[name] = campaign.pinned(path)
        manifest['initial'][arm] = {'checkpoint': campaign.pinned(target / 'checkpoint.pkl'), 'history': files}
    monkeypatch.setattr(campaign, 'verify', lambda *a, **k: manifest)
    monkeypatch.setattr(campaign, 'gpu_required', lambda: None)
    calls = []
    def execute(command):
        calls.append(command)
        run = tmp_path / 'runs' / arm
        run.mkdir(parents=True, exist_ok=True)
        for name in ('train_metrics.jsonl', 'backtracking_trials.jsonl'):
            (run / name).write_text('new completed history')
        (run / 'final.pkl').write_bytes(b'final')
    monkeypatch.setattr(campaign, 'call', execute)
    monkeypatch.setattr(campaign.pilot, 'validate_run', lambda *a, **k:
                        ([{'decrease_warning': True}], tmp_path / 'runs' / arm / 'final.pkl'))
    return manifest, calls


@pytest.mark.parametrize('kind', ['continuation', 'fresh'])
def test_start_uses_exact_resume_or_fresh_initialization_without_warmstart(tmp_path, monkeypatch, kind):
    manifest, calls = setup_train(tmp_path, monkeypatch, kind)
    result = campaign.train(tmp_path, 0, 0)
    command = list(map(str, calls[0]))
    assert '--init-from' not in command and '--zero-state-on-init' not in command
    assert ('--resume' in command) == (kind == 'continuation')
    assert command[command.index('--stop-after-steps') + 1] == str(manifest['stage_endpoints'][0])
    assert result['warning_counts']['decrease_warning'] == 1
    with pytest.raises(FileExistsError, match='must not be repeated'):
        campaign.train(tmp_path, 0, 0)


def test_continuation_rejects_changed_history_before_execution(tmp_path, monkeypatch):
    _, calls = setup_train(tmp_path, monkeypatch, 'continuation')
    (tmp_path / 'runs/mgda/train_metrics.jsonl').write_text('uncommitted modification')
    with pytest.raises(ValueError, match='history differs'):
        campaign.train(tmp_path, 0, 0)
    assert not calls


def test_following_stage_requires_pinned_previous_checkpoint_and_histories(tmp_path, monkeypatch):
    manifest, calls = setup_train(tmp_path, monkeypatch, 'fresh')
    first = campaign.train(tmp_path, 0, 0)
    campaign.train(tmp_path, 0, 1)
    assert str(first['checkpoint']['path']) in list(map(str, calls[1]))
    assert calls[1][-2] == '--resume'
    (tmp_path / 'runs/mgda/train_metrics.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Pinned input changed'):
        campaign.train_proof(tmp_path, manifest, 'mgda')
    assert len(calls) == 2
