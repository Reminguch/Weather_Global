from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_smoothing as campaign
from scripts.experiments.run_v24_mgda import pinned, seal, write
from src.analysis.smoothing_observer import atomic_npz


def test_cohort_identity_and_submission_dependencies(tmp_path):
    ids = campaign.expected_ids({'initializations':[{'initialization_time':'2023-01-01T00:00:00Z'},
                                                   '2023-01-04T00:00:00+00:00']})
    assert ids == ['20230101T000000Z','20230104T000000Z']
    with pytest.raises(ValueError, match='Duplicate'):
        campaign.expected_ids({'initializations':['2023-01-01T00:00:00Z']*2})
    m = {'workspace':str(tmp_path),'source_root':str(tmp_path/'source'),'model_id':'M1'}
    command = campaign.sbatch(tmp_path,m,'evaluate',memory=24,minutes=90,array='1-7',dependency='1234')
    assert '--dependency=afterok:1234' in command
    assert '--array=1-7' in command and '--gres=gpu:1' in command
    assert not any('%' in arg for arg in command if arg.startswith('--array'))
    assert command[-4:] == ['evaluate',str(tmp_path),str(tmp_path/'source'),'0']


def test_resources_follow_measurement_not_scientific_metric(tmp_path):
    (tmp_path/'checks').mkdir()
    seal(tmp_path/'checks/calibrate_00.json',{'elapsed_seconds':300.,'max_rss_gib':12.,'finite_loss_increase':3.})
    result = campaign.resource_budget(tmp_path,{'shards':{'calibrate':[{'count':1},{'count':11}]}},'calibrate')
    assert result['memory_gib'] == 20
    assert result['minutes'] == 85


def test_missing_duplicate_and_corrupt_statistics_are_errors(tmp_path):
    cohort = tmp_path/'cohort.json'
    write(cohort,{'initializations':['2023-01-01T00:00:00Z']})
    source = tmp_path/'20230101T000000Z.npz'
    atomic_npz(source,{'e0':np.ones((2,4,1))}, {'campaign_hash':'x','mode':'calibrate',
        'initialization_id':source.stem,'status':'complete'})
    (tmp_path/'checks').mkdir()
    proof = {'campaign_hash':'x','count':1,'statistics':[pinned(source)]}
    seal(tmp_path/'checks/calibrate_00.json',proof)
    manifest = {'record_hash':'x','shards':{'calibrate':[{'count':1}]},
                'cohorts':{'calibrate':pinned(cohort)}}
    assert campaign.statistics_paths(tmp_path,manifest,'calibrate') == [source]
    arrays,_ = campaign.load_statistics(source,manifest,'calibrate')
    assert arrays['e0'].shape == (2,4,1)
    with pytest.raises(ValueError,match='provenance'):
        campaign.load_statistics(source,{**manifest,'record_hash':'wrong'},'calibrate')
    write(cohort,{'initializations':['2023-01-01T00:00:00Z','2023-01-04T00:00:00Z']})
    with pytest.raises(ValueError,match='Missing'):
        campaign.statistics_paths(tmp_path,manifest,'calibrate')
    source.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='changed'):
        campaign.statistics_paths(tmp_path,manifest,'calibrate')


def test_ratio_does_not_floor_zero_power():
    ratio = campaign.ratio(np.array([0.,1.,2.]),np.array([0.,0.,1.]))
    assert np.isnan(ratio[:2]).all() and ratio[2] == 2


def test_annual_report_pools_power_and_retains_finite_closure_warning(tmp_path,monkeypatch):
    from src.analysis.spherical_smoothing import json_digest
    (tmp_path/'checks').mkdir()
    paths=[]
    filters=seal(tmp_path/'filters.json', {'layout':{'channels':[['temperature',850]]},'campaign_hash':'x'})
    for i in range(2):
        path=tmp_path/f'2023010{1+3*i}T000000Z.npz'
        shape=(40,2,3,1)
        truth=np.ones((40,1,3))*(1+3*i)
        pred=np.broadcast_to(truth[:,None,None],shape+(3,)).copy()
        if i==0:
            pred*=2
        arrays={'truth_power':truth,'prediction_power':pred,'cross_power':pred*.4,
            'error_power':pred*.6,'grid_mse':np.ones(shape),
            'projected_grid_mse':np.ones(shape)*.9,'unresolved_grid_mse':np.ones(shape)*.2,
            'exact_gc_loss':np.ones((40,2,3)), 'mean_drift':np.zeros(shape),
            'positive_field_minimum':np.ones((40,2,3,1)),
            'negative_fraction':np.zeros((40,2,3,1))}
        atomic_npz(path,arrays,{'campaign_hash':'x','mode':'evaluate','initialization_id':path.stem,
            'status':'complete','filter_digest':json_digest(filters),'layout':filters['layout']})
        paths.append(path)
    monkeypatch.setattr(campaign,'statistics_paths',lambda *args:paths)
    monkeypatch.setattr(campaign,'plot_summary',lambda *args:None)
    result=campaign.report(tmp_path,{'record_hash':'x','model_id':'M1'})
    assert result['status']=='complete'
    assert result['numerical_diagnostics']['warnings']
    with np.load(tmp_path/'annual_pooled.npz') as archive:
        # Ratio of pooled powers: (2+4)/(1+4)=1.2, not mean(2,1)=1.5.
        np.testing.assert_allclose(archive['power_ratio'],1.2)
        np.testing.assert_allclose(archive['macro_power_mismatch'],.2)
        assert archive['per_initialization_exact_gc_loss'].shape==(2,40,2,3)
