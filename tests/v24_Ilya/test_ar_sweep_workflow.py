"""Scientific controls and fail-fast behavior of the AR training campaign."""
import copy
import json

import pytest

from scripts.experiments import run_v24_res1_ar_sweep as sweep
from src.models.mamba.v24_Ilya.training.config import load_training_config


def test_only_horizon_and_output_identity_change_across_arms(tmp_path):
    normalized = {}
    for k in sweep.HORIZONS:
        configs = sweep.configs_for(tmp_path, k)
        for stage, config in configs.items():
            path = tmp_path / f'{k}_{stage}.json'
            path.write_text(json.dumps(config))
            resolved = load_training_config(path)
            assert resolved.bptt_steps * resolved.distributed.global_batch_size == 96
            assert resolved.loss_mode == 'all_steps'
            assert resolved.feedback_mode == 'closed_loop_sg'
            assert resolved.temporal_state_policy == 'carry'
            # Single-step control must never consume its own weather predictions.
            feedback_positions = [i for i in range(24)
                                  if resolved.truth_prefix_steps - 1 <= i < 23]
            assert len(feedback_positions) == (0 if k == 1 else k)
            comparable = copy.deepcopy(config)
            del comparable['output']
            del comparable['sequence']['ar_tail_k']
            if stage in normalized:
                assert comparable == normalized[stage]
            normalized[stage] = comparable
    assert normalized['warmup']['optimizer']['max_steps'] == 500
    assert normalized['train']['optimizer']['max_steps'] == 2000
    assert normalized['train']['optimizer']['mamba_lr_multiplier'] == 3


def test_failed_smoke_prevents_production_training(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sweep.base, 'train', lambda *args: calls.append(args))
    def reject(*args):
        raise ValueError('Nonfinite parameters')
    monkeypatch.setattr(sweep, 'check_checkpoint', reject)
    entry = {'name': 'k01', 'smoke_config': 'smoke', 'warmup_config': 'warmup',
             'config': 'train', 'init_from': 'warmup-checkpoint'}
    with pytest.raises(ValueError, match='Nonfinite'):
        sweep.train(entry, tmp_path)
    assert calls == [('smoke',)]


def test_branch_starts_from_its_own_horizon_and_swa_excludes_warmup(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sweep.base, 'train', lambda *args: calls.append(('train', args)))
    monkeypatch.setattr(sweep.base, 'run', lambda *args: calls.append(('swa', args)))
    monkeypatch.setattr(sweep, 'check_checkpoint', lambda *args: {'passed': True})
    entry = {'name': 'k08', 'smoke_config': 'k08-smoke', 'warmup_config': 'k08-warmup',
             'config': 'k08-train', 'init_from': tmp_path / 'warmup/k08/checkpoint500.pkl'}
    sweep.train(entry, tmp_path)
    assert calls[:3] == [('train', ('k08-smoke',)), ('train', ('k08-warmup',)),
                         ('train', ('k08-train', entry['init_from']))]
    args = calls[-1][1]
    inputs = args[args.index('--inputs') + 1:args.index('--source-steps')]
    assert len(inputs) == 4
    assert all(p.is_relative_to(tmp_path / 'runs/k08/checkpoints') for p in inputs)
    assert [p.name for p in inputs] == [f'checkpoint_step{s:08d}.pkl' for s in (500, 1000, 1500, 2000)]
