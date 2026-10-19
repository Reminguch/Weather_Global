"""Pinned width variants and physical diagnostics across all GraphCast channels."""
import copy
import json

import numpy as np
import pytest
import xarray as xr

from scripts.analyze_models.evaluate_v24_amse_regime import task_architecture
from src.models.mamba.v24_Ilya.spherical_observer import SphericalObserver


def test_width_override_is_explicit_and_cannot_change_other_architecture_fields():
    reference = {'temporal_d_inner': 16, 'width': 512, 'mesh_size': 5}
    wider = dict(reference, temporal_d_inner=32)
    assert task_architecture({'architecture': wider}, reference, wider) == wider
    assert reference['temporal_d_inner'] == 16
    with pytest.raises(ValueError, match='pinned evaluation task'):
        task_architecture({}, reference, wider)
    with pytest.raises(ValueError, match='pinned evaluation task'):
        task_architecture({'architecture': wider}, reference, reference)
    for wrong in (dict(wider, width=128), dict(wider, mesh_size=4)):
        with pytest.raises(ValueError, match='only temporal_d_inner'):
            task_architecture({'architecture': wrong}, reference, wrong)


def full_fields():
    lat, lon = np.linspace(-90, 90, 9), np.arange(16) * 22.5
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    coords = {'batch': [0], 'time': [np.timedelta64(6, 'h')],
              'lat': lat, 'lon': lon, 'level': levels}
    surface = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
               'mean_sea_level_pressure', 'total_precipitation_6hr']
    upper = ['temperature', 'specific_humidity', 'geopotential', 'u_component_of_wind',
             'v_component_of_wind', 'vertical_velocity']
    data = {}
    smooth = np.sin(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    for index, name in enumerate(surface):
        data[name] = (('batch', 'time', 'lat', 'lon'),
                      (smooth + index + 1)[None, None].astype(np.float32))
    for index, name in enumerate(upper):
        values = smooth + np.arange(13)[:, None, None] + index + 1
        data[name] = (('batch', 'time', 'level', 'lat', 'lon'), values[None, None].astype(np.float32))
    return xr.Dataset(data, coords=coords)


def test_full_spherical_observer_saves_83_channels_and_physical_errors(tmp_path):
    truth = full_fields()
    observer = SphericalObserver(tmp_path, {'model_id': 'di32'}, target_steps=1,
                                 lmax=5, all_channels=True)
    observer.begin_sample({'initialization_id': '20230101T000000Z',
                           'initialization_time': '2023-01-01T00:00:00Z', 'lead_hours': [6]})
    observer.update_step(0, truth, truth + 3, truth + 2)
    observer.finish_sample()
    with np.load(tmp_path / '20230101T000000Z.npz') as record:
        assert record['truth_power'].shape == (1, 83, 6)
        names = record['variables'].tolist()
        assert len(set(names)) == 83
        assert 'geopotential_level500' in names and 'temperature_level850' in names
        assert record['units'][names.index('geopotential_level500')] == 'm2/s2'
        np.testing.assert_allclose(record['full_grid_mse'], 4, rtol=2e-6)
        np.testing.assert_allclose(record['baseline_grid_mse'], 9, rtol=2e-6)
        assert json.loads(str(record['metadata']))['field_selection'] == 'all_primitive_channels'


def test_full_spherical_observer_rejects_changed_channel_layout(tmp_path):
    observer = SphericalObserver(tmp_path, {}, all_channels=True)
    truth = full_fields()
    observer._fields(truth)
    with pytest.raises(ValueError, match='channel layout changed'):
        observer._fields(truth.drop_vars('geopotential'))
    with pytest.raises(ValueError, match='one field per lead'):
        observer._fields(xr.concat([truth, truth], dim='batch'))


def test_specification_uses_pinned_width_in_protocol_hash(tmp_path, monkeypatch):
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluation
    reference = {'temporal_d_inner': 16, 'width': 512, 'mesh_size': 5}
    wider = dict(reference, temporal_d_inner=32)
    config = {'architecture': wider, 'data': {'baseline_checkpoint': '/baseline', 'stats_dir': '/stats'}}
    prototype = {'architecture': reference, 'baseline': {'checkpoint_sha256': 'baseline'},
                 'statistics': [], 'stage': 'exact'}
    task = {'id': 'di32_step02000', 'architecture': wider, 'run_config': '/config', 'checkpoint': '/checkpoint'}
    manifest = {'task_specs': [task], 'protocols': {'exact': prototype},
                'workspace': str(tmp_path), 'evaluation_role': 'development'}
    monkeypatch.setattr(evaluation, 'load_manifest', lambda path: copy.deepcopy(manifest))
    monkeypatch.setattr(evaluation, 'read_json', lambda path: config)
    monkeypatch.setattr(evaluation, 'frozen_file', lambda path: {'path': str(path), 'sha256': str(path).split('/')[-1]})
    monkeypatch.setattr(evaluation, 'verify_specification', lambda spec: None)
    result = evaluation.make_specification(tmp_path / 'manifest.json', 0, 'exact')
    assert result['architecture'] == wider
    assert result['protocol_hash'] == evaluation.protocol_digest(result)
    assert prototype['architecture'] == reference
