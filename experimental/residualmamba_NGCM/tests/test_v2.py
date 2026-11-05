"""v2 exclusion, independent loss reduction, reporting and migration contracts."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from residualmamba_ngcm.config import Config, P, SCORED_FIELDS
from residualmamba_ngcm.loss import contributions, weighted_mse, target_norm, correct
from residualmamba_ngcm.train import validate
from residualmamba_ngcm.reuse import check_receipts, import_inputs
from residualmamba_ngcm.artifacts import sha256

LAT=np.arcsin(np.polynomial.legendre.leggauss(4)[0])


def test_v2_config_rejects_v1_checkpoint_protocol():
    config=Config.load(Path(__file__).parents[1]/'configs/default.json')
    assert config.schema_version==2 and config.output_channels==145
    assert 'cloud_alpha' not in asdict(config)
    with pytest.raises(ValueError):Config(schema_version=1)
    with pytest.raises(ValueError):Config(output_channels=203)
    with pytest.raises(TypeError):Config(cloud_alpha=0)


def test_independent_five_field_gaussian_pressure_reduction_and_gradient():
    rng=np.random.default_rng(22)
    prediction=rng.normal(size=(5,29,3,4)).astype(np.float32)
    truth=rng.normal(size=prediction.shape).astype(np.float32)
    # Independent normalized quadrature integral, then p/sum(p), then field sum.
    quadrature=np.polynomial.legendre.leggauss(4)[1]/2
    expected=((prediction.astype(np.float64)-truth)**2*quadrature).sum(-1).mean(-1)
    expected*=np.asarray(P)[None,:]/sum(P)
    np.testing.assert_allclose(contributions(prediction,truth,LAT),expected,rtol=3e-7)
    np.testing.assert_allclose(weighted_mse(prediction,truth,LAT),expected.sum(),rtol=3e-7)
    gradient=jax.grad(lambda p:weighted_mse(p,jnp.asarray(truth),LAT))(jnp.asarray(prediction))
    expected_gradient=2*(prediction-truth)*quadrature/3*np.asarray(P)[None,:,None,None]/sum(P)
    np.testing.assert_allclose(gradient,expected_gradient,rtol=4e-7,atol=1e-9)
    with pytest.raises(ValueError):weighted_mse(np.zeros((7,29,3,4)),np.zeros((7,29,3,4)),LAT)


def test_cloud_targets_and_scales_cannot_enter_loss_or_correction():
    baseline=np.zeros((7,37,3,4),np.float32); baseline[5:]=-0.
    truth=np.ones_like(baseline);stats={'diff_std':np.ones((7,37),np.float32)}
    expected=target_norm(truth,baseline,stats)
    truth[5:]=1e30;stats['diff_std'][5:]=1e-30
    np.testing.assert_array_equal(target_norm(truth,baseline,stats),expected)
    result=np.asarray(correct(baseline,np.ones((5,29,3,4),np.float32),stats))
    np.testing.assert_array_equal(result[5:].view(np.uint32),baseline[5:].view(np.uint32))
    np.testing.assert_array_equal(result[:,:8].view(np.uint32),baseline[:,:8].view(np.uint32))
    np.testing.assert_array_equal(result[:5,8:],1)


def test_validation_field_contributions_sum_to_total():
    model=SimpleNamespace(apply=lambda p,s,k,x:(x,s))
    x=np.ones((24,5,29,1,4),np.float32)
    report=validate(model,{}, {}, [np.arange(96)],lambda _:dict(features=x,target=x*2),LAT,Config())
    assert list(report['per_field'])==list(SCORED_FIELDS)
    np.testing.assert_allclose(sum(report['per_field'].values()),report['mean_loss'])
    np.testing.assert_allclose(sum(report['baseline_per_field'].values()),report['baseline_mean_loss'])
    np.testing.assert_allclose(report['mean_loss'],5,rtol=1e-6)
    np.testing.assert_allclose(report['baseline_mean_loss'],20,rtol=1e-6)


def test_reuse_receipts_detect_changed_source_and_never_overwrite(tmp_path):
    from residualmamba_ngcm.reuse import V1_SOURCE
    path=tmp_path/'stats.npz';path.write_bytes(b'original')
    manifest={'reused_inputs':{'producer_source_identity':V1_SOURCE,'donor_root':str(tmp_path),
                               'files':{'stats.npz':sha256(path)}}}
    check_receipts(manifest)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Reused input changed'):check_receipts(manifest)
    manifest['reused_inputs']['producer_source_identity']='unknown'
    with pytest.raises(ValueError,match='Unrecognized'):check_receipts(manifest)
    with pytest.raises((ValueError,FileExistsError)):import_inputs(tmp_path,tmp_path,Config())


def test_evaluation_reports_five_fields_and_matching_j(monkeypatch,tmp_path):
    from residualmamba_ngcm import evaluate
    from residualmamba_ngcm.artifacts import load_arrays
    baseline=np.ones((7,37,2,4),np.float32)
    truth=baseline*2; corrected=baseline.copy();corrected[:5,8:]+=0.5
    monkeypatch.setattr(evaluate,'forecast',lambda *a,**kw:iter(
        (k,baseline,corrected,truth) for k in range(1,41)))
    monkeypatch.setattr(evaluate,'spectra',lambda *a:np.zeros((3,3,2)))
    monkeypatch.setattr(evaluate,'bootstrap',lambda origins,c,b:np.zeros((2,*c.shape[1:])))
    bb=SimpleNamespace(model=SimpleNamespace(data_coords=SimpleNamespace(
        horizontal=SimpleNamespace(latitudes=LAT))))
    stats={'diff_std':np.ones((7,37),np.float32),'climatology':np.zeros_like(baseline)}
    evaluate.run(bb,None,None,None,None,stats,Config(),[np.datetime64('2023-01-02')],tmp_path,'test-v2')
    arrays,meta=load_arrays(tmp_path/'summary.npz')
    assert meta['field_order']==list(SCORED_FIELDS)
    assert arrays['rmse_corrected'].shape==(40,5,37)
    assert arrays['J_per_field'].shape==(40,2,5)
    np.testing.assert_allclose(arrays['J_per_field'][:,0].sum(-1),arrays['J_native'])
    np.testing.assert_allclose(arrays['J_per_field'][:,1].sum(-1),arrays['J_corrected'])
    np.testing.assert_allclose(arrays['J_field_share'].sum(-1),1,rtol=1e-6)
