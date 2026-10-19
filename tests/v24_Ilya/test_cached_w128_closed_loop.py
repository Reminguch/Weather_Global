"""Protocol and failure isolation for the cached-to-closed-loop continuation."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_cached_w128_closed_loop as campaign
from src.models.mamba.v24_Ilya.training.config import load_training_config
from src.models.mamba.v24_Ilya.config import parse_args


@pytest.fixture
def setup(tmp_path):
    entries = []
    for arm in campaign.ARMS:
        source = campaign.ROOT / campaign.PARENTS / f'{arm}_carry/configs/online.json'
        template = json.loads(source.read_text())
        entry = {'arm': arm, 'parent': {'path': str(tmp_path / f'{arm}_parent.pkl')}}
        for smoke in (False, True):
            cfg = campaign.make_config(template, tmp_path, arm, smoke=smoke)
            path = tmp_path / f'{arm}_{smoke}.json'
            path.write_text(json.dumps(cfg))
            entry['smoke_config' if smoke else 'config'] = str(path)
        entries.append(entry)
    m = {'entries': entries, 'eval_data': '/evaluation.zarr', 'chosen_idx': list(range(32)),
         'workspace': str(campaign.ROOT), 'source_root': str(tmp_path / 'source'),
         'resources': {'di32': {'memory': 352, 'minutes': 480},
                       'di64': {'memory': 384, 'minutes': 480},
                       'eval': {'memory': 16, 'minutes': 90}}}
    return tmp_path, m


def test_continuation_preserves_parent_architecture_and_changes_only_declared_fields(setup):
    root, m = setup
    for entry in m['entries']:
        arm = entry['arm']
        source = campaign.read(campaign.ROOT / campaign.PARENTS / f'{arm}_carry/configs/online.json')
        cfg = campaign.read(entry['config'])
        resolved = load_training_config(Path(entry['config']))
        assert cfg['architecture'] == source['architecture']
        assert cfg['data'] == source['data']
        assert resolved.feedback_mode == 'closed_loop_sg'
        assert resolved.distributed.global_batch_size == 1
        assert (resolved.bptt_steps, resolved.ar_tail_k, resolved.segment_steps) == (24, 20, 96)
        assert resolved.temporal_state_policy == 'carry'
        assert resolved.loss_mode == 'all_steps'
        assert (resolved.max_steps, resolved.learning_rate, resolved.warmup_steps) == (2000, 1e-5, 200)
        assert resolved.validation.every_steps == 500
        args = campaign.train_command(entry)
        assert args[-3:] == ['--init-from', entry['parent']['path'], '--zero-state-on-init']


def test_evaluation_cli_matches_each_native_architecture_and_cold_protocol(setup):
    root, m = setup
    for entry in m['entries']:
        args = campaign.eval_command(m, entry, root / 'checkpoint.pkl', root / 'out.json')
        parsed = parse_args(list(map(str, args[3:])))
        assert parsed.architecture.temporal_d_inner == int(entry['arm'][2:])
        assert parsed.architecture.residual_width == 128
        assert parsed.target_steps == 40 and parsed.n_samples == 32
        assert parsed.effective_warmup_steps == 0
        assert parsed.sample_total_steps == 64  # Historical matched32 reservation.
        assert parsed.residual_state_init == 'zero'


def test_failed_smoke_blocks_continuation(monkeypatch, setup):
    root, m = setup
    calls = []
    monkeypatch.setattr(campaign, 'call', lambda args: calls.append(args))
    def fail(*args, **kwargs):
        raise ValueError('Smoke failed')
    monkeypatch.setattr(campaign, 'check_training', fail)
    with pytest.raises(ValueError, match='Smoke failed'):
        campaign.train(root, m, m['entries'][0])
    assert len(calls) == 1
    assert m['entries'][0]['smoke_config'] in calls[0]


def test_submission_graph_is_bounded_and_uses_per_arm_dependencies(monkeypatch, setup):
    root, m = setup
    monkeypatch.setattr(campaign, 'submit_once', lambda *args: pytest.fail('Dry run submitted'))
    campaign.submit(root, m, dry_run=True)
    commands = campaign.read(root / 'submission_dry_run.json')['commands']
    assert len(commands) == 6
    assert '--array=0,5' in commands['parents']
    for arm, selection in [('di32', '1-4'), ('di64', '6-9')]:
        assert f'--dependency=afterok:<train_{arm}>' in commands[f'eval_{arm}']
        assert f'--array={selection}' in commands[f'eval_{arm}']
    assert '--dependency=afterok:<parents>:<eval_di32>:<eval_di64>' in commands['report']
    assert all('%' not in arg for cmd in commands.values() for arg in cmd if arg.startswith('--array='))


def good_result(root, m):
    b = np.arange(1, 41, dtype=float)
    f = b * .9
    return {'ckpt': str(root / 'parent.pkl'), 'ckpt_in': campaign.read(m['entries'][0]['config'])['data']['baseline_checkpoint'],
        'evaluation_status': 'complete', 'evaluated_samples': 32, 'target_steps': 40,
        'eval_mode': 'cold_full', 'eval_feedback': 'full', 'residual_state_init': 'zero',
        'chosen_idx': m['chosen_idx'], 'original_graphcast_loss': {
            'baseline_per_step': b.tolist(), 'full_per_step': f.tolist(),
            'baseline_rollout': float(b.mean()), 'full_rollout': float(f.mean()),
            'improvement_pct_rollout': 10.}}


@pytest.mark.parametrize('field,value', [('chosen_idx', [0]), ('evaluated_samples', 31),
    ('eval_feedback', 'baseline'), ('residual_state_init', 'ckpt'), ('ckpt', '/wrong.pkl')])
def test_wrong_cohort_feedback_state_or_checkpoint_rejected(setup, field, value):
    root, m = setup
    r = good_result(root, m)
    r[field] = value
    with pytest.raises(ValueError):
        campaign.validate_evaluation(r, m, root / 'parent.pkl')


def test_aggregation_checked_before_acceptance(monkeypatch, setup):
    root, m = setup
    r = good_result(root, m)
    monkeypatch.setattr(campaign, 'pinned', lambda p: {'path': str(p)})
    assert campaign.validate_evaluation(r, m, root / 'parent.pkl')['passed']
    r['original_graphcast_loss']['improvement_pct_rollout'] = 9.
    with pytest.raises(ValueError, match='aggregation'):
        campaign.validate_evaluation(r, m, root / 'parent.pkl')


@pytest.mark.parametrize('single_arm', [False, True])
def test_terminal_report_handles_results_and_baseline_drift(setup, single_arm):
    root, m = setup
    if single_arm:
        m['entries'] = [dict(m['entries'][0], arm='w512_di64')]
        m['residual_width'] = 512
    arms = [entry['arm'] for entry in m['entries']]
    m['workspace'] = str(root)
    for arm_index, arm in enumerate(arms):
        for step in (0, *campaign.STEPS):
            cp = root / f'{arm}_{step}.pkl'
            cp.write_bytes(f'{arm}_{step}'.encode())
            r = good_result(root, m)
            r['ckpt'] = str(cp)
            g = r['original_graphcast_loss']
            # Distinguish pooled ratios from the arithmetic mean of lead ratios.
            b = np.arange(1, 41, dtype=float) * (1 + .001 * arm_index)
            f = np.arange(1, 41, dtype=float) * .9 - step * .00001
            g.update(baseline_per_step=b.tolist(), full_per_step=f.tolist(),
                     baseline_rollout=float(b.mean()), full_rollout=float(f.mean()),
                     improvement_pct_rollout=float(100 * (1 - f.sum() / b.sum())))
            out = root / 'evaluation' / f'{arm}_step{step:05d}.json'
            campaign.write(out, r)
            campaign.write(out.with_suffix('.proof.json'), {'passed': True,
                'checkpoint': campaign.pinned(cp), 'output': campaign.pinned(out)})
    campaign.summarize(root, m)
    report = campaign.read(root / 'results/summary.json')
    assert report['status'] == 'complete' and len(report['rows']) == len(arms) * 5
    assert report['canonical_baseline'] == f'{arms[0]} parent execution'
    row = report['rows'][-1]
    assert row['loss_reduction_pct_canonical'] == pytest.approx(100 * (1 - (18.45 - .02) / 20.5))
    assert row['pooled_baseline_drift_pct'] == pytest.approx(0 if single_arm else .1)
    assert row['day10_loss_reduction_pct_canonical'] == pytest.approx(100 * (1 - 35.98 / 40))
    assert (root / 'plots/analyze_models/images/resolution_eval' / root.name / 'closed_loop_comparison.pdf').is_file()


def test_width512_native_architecture_and_single_arm_submission(tmp_path, monkeypatch):
    parent = campaign.read(campaign.ROOT / campaign.W512_PARENT / 'run_config.json')
    path = tmp_path / 'config.json'
    campaign.write(path, campaign.make_config(parent, tmp_path, 'w512_di64'))
    entry = {'arm': 'w512_di64', 'config': str(path)}
    m = {'entries': [entry], 'eval_data': '/data.zarr', 'workspace': str(campaign.ROOT),
         'source_root': str(tmp_path / 'source'), 'residual_width': 512,
         'resources': {'w512_di64': {'memory': 416, 'minutes': 600},
                       'eval': {'memory': 20, 'minutes': 120}}}
    cfg = load_training_config(path)
    assert asdict(cfg.architecture) == parent['architecture']
    args = campaign.eval_command(m, entry, tmp_path / 'checkpoint.pkl', tmp_path / 'out.json')
    arch = parse_args(list(map(str, args[3:]))).architecture
    assert asdict(arch) == parent['architecture']
    monkeypatch.setattr(campaign, 'submit_once', lambda *args: pytest.fail('Dry run submitted'))
    campaign.submit(tmp_path, m, dry_run=True)
    commands = campaign.read(tmp_path / 'submission_dry_run.json')['commands']
    assert set(commands) == {'parents', 'train_w512_di64', 'eval_w512_di64', 'report'}
    assert '--array=0' in commands['parents']
    assert '--qos=gpu-medium' in commands['train_w512_di64']
    assert '--array=1-4' in commands['eval_w512_di64']
    assert '--dependency=afterok:<train_w512_di64>' in commands['eval_w512_di64']
    assert '--dependency=afterok:<parents>:<eval_w512_di64>' in commands['report']
