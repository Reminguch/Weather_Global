"""Regression coverage for recovering the warm-up to LR-branch transition."""
import json
import pickle

import numpy as np
import pytest

from scripts.experiments import run_v24_res1_batch4_lr as driver
from src.models.mamba.v24_Ilya.checkpoint import TRAINING_CHECKPOINT_FORMAT
from src.models.mamba.v24_Ilya.config import ARCHITECTURE_ID, SCHEMA_VERSION, V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig


@pytest.fixture
def warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'EXPERIMENT', tmp_path)
    config = V24IlyaTrainConfig(
        prepared_root=tmp_path / 'prepared', anchor_manifest_root=tmp_path / 'anchors',
        baseline_checkpoint=tmp_path / 'baseline', output_root=tmp_path / 'seeds',
        run_name='warmup', max_steps=500, architecture=V24IlyaArchitectureConfig(),
    )
    config_path = tmp_path / 'warmup.json'
    config_path.write_text(json.dumps(config.to_dict()))
    checkpoint = tmp_path / 'checkpoint_step00000500.pkl'
    payload = dict(
        architecture_id=ARCHITECTURE_ID, schema_version=SCHEMA_VERSION,
        checkpoint_format=TRAINING_CHECKPOINT_FORMAT, checkpoint_kind='training',
        completed_step=500, residual_params={'module': {'w': np.ones(2, np.float32)}},
        residual_state={}, optimizer_state={'count': np.array(500, np.int32)},
        rng_key=np.zeros(2, np.uint32), training_cursor={},
        resolved_training_config=config.to_dict(), baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint='baseline', anchor_manifest_fingerprint='anchors',
        parameter_overlay_metadata={},
    )
    entry = dict(name='di16_slow_mamba', warmup_config=str(config_path), init_from=str(checkpoint))
    return entry, checkpoint, payload


@pytest.mark.parametrize('reuse', [False, True])
def test_branch_start_loads_string_manifest_path_and_reuses_warmup(warmup, monkeypatch, reuse):
    entry, checkpoint, payload = warmup
    calls = []
    def train(config):
        calls.append(config)
        checkpoint.write_bytes(pickle.dumps(payload))
    monkeypatch.setattr(driver, 'train', train)
    if reuse:
        checkpoint.write_bytes(pickle.dumps(payload))
    driver.prepare_branch_start(entry)
    assert calls == ([] if reuse else [entry['warmup_config']])
    audit = json.loads((driver.EXPERIMENT / 'checks/start_di16_slow_mamba.json').read_text())
    assert audit['reused_warmup'] is reuse
    assert audit['completed_step'] == 500
    assert len(audit['residual_params_sha256']) == 64
    digest = audit['residual_params_sha256']
    driver.prepare_branch_start(entry)
    assert json.loads((driver.EXPERIMENT / 'checks/start_di16_slow_mamba.json').read_text())['residual_params_sha256'] == digest
    assert calls == ([] if reuse else [entry['warmup_config']])


@pytest.mark.parametrize('invalid', ['incomplete', 'configuration mismatch'])
def test_branch_start_rejects_wrong_existing_warmup(warmup, monkeypatch, invalid):
    entry, checkpoint, payload = warmup
    if invalid == 'incomplete':
        payload['completed_step'] = 250
    else:
        payload['resolved_training_config']['optimizer']['seed'] += 1
    checkpoint.write_bytes(pickle.dumps(payload))
    monkeypatch.setattr(driver, 'train', lambda *args: pytest.fail('Must not overwrite warm-up'))
    with pytest.raises(ValueError, match=invalid):
        driver.prepare_branch_start(entry)
    assert not (driver.EXPERIMENT / 'checks/start_di16_slow_mamba.json').exists()
