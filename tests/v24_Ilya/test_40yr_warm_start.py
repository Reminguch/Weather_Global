"""Warm-start identity and single-width release checks for the larger-data run."""
from types import SimpleNamespace

import numpy as np

from scripts.experiments import run_v24_cached_40yr as campaign
from scripts.experiments.cached_40yr_qualification import training_initialization_checks
from src.models.mamba.v24_Ilya.training.data import TrainingCursor


def test_warm_start_accepts_trained_heads_and_rejects_weight_optimizer_or_state_changes(monkeypatch):
    from src.models.mamba.v24_Ilya import checkpoint
    from src.models.mamba.v24_Ilya.training import endpoint_step
    params = {'temporal_residual_head': {'w': np.array([1., 2.], np.float32)}}
    parent = SimpleNamespace(residual_params=params, baseline_checkpoint_fingerprint='baseline',
                             resolved_training_config={'architecture': {'width': 512}})
    source = SimpleNamespace(residual_params=params, baseline_checkpoint_fingerprint='baseline',
        parameter_overlay_metadata={'init_from_sha256': 'parent'}, completed_step=0,
        residual_state={'s': np.zeros(2)}, optimizer_state={'count': np.array(0)},
        training_cursor=TrainingCursor().to_dict())
    config = SimpleNamespace(to_dict=lambda: {'architecture': {'width': 512}})
    monkeypatch.setattr(checkpoint, 'load_v24_Ilya_training_checkpoint', lambda _: parent)
    monkeypatch.setattr(checkpoint, 'file_sha256', lambda _: 'parent')
    monkeypatch.setattr(endpoint_step, 'build_optimizer', lambda _: (
        SimpleNamespace(init=lambda _: {'count': np.array(0)}), None))
    assert training_initialization_checks(source, config, '/parent.pkl')['passed']
    source.residual_params = {'temporal_residual_head': {'w': np.zeros(2, np.float32)}}
    assert not training_initialization_checks(source, config, '/parent.pkl')['checks']['all_parent_weights']
    source.residual_params = params
    source.optimizer_state = {'count': np.array(4)}
    assert not training_initialization_checks(source, config, '/parent.pkl')['checks']['fresh_optimizer']
    source.optimizer_state = {'count': np.array(0)}
    source.residual_state = {'s': np.ones(2)}
    assert not training_initialization_checks(source, config, '/parent.pkl')['checks']['zero_recurrent_state']


def test_single_width_cache_release_and_terminal_report(tmp_path, monkeypatch):
    flow = {'root': str(tmp_path), 'widths': [512]}
    campaign.write(tmp_path / 'state.json', {'automation_enabled': True,
        'jobs': {'cache_verify': {'job_id': '10'}, 'evaluate_w512': {'job_id': '20'}}})
    calls = []
    monkeypatch.setattr(campaign, 'launch_pilots', lambda *args, **kw: calls.append(kw))
    campaign.advance(flow, 'cache_verify', campaign.spec('cache_verify'), {})
    assert calls == [{'family': 'training', 'dependency': '10', 'width': 512}]
    monkeypatch.setattr(campaign, 'submit_job', lambda *args, **kw: calls.append((args[1], kw)))
    campaign.maybe_submit_report(flow)
    assert calls[-1] == ('report', {'dependencies': ['20']})
