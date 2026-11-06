import copy
from types import SimpleNamespace
import numpy as np
import pytest
from matched_v2_eval import matching_context,protocol,summarize


def plan(k):
    return {'ar_tail_k':k,'identity':'run','source_identity':'source',
            'prerequisites':{'stats.npz':'stats','preflight.json':'native20','tail1-inputs.json':'native1'}}


def test_uses_training_loader():
    calls=[]
    worker=SimpleNamespace(context=lambda root,evaluation:calls.append(evaluation) or ('native20' if evaluation else 'native1'))
    assert matching_context(worker,'/run')=='native1'
    assert calls==[False]


@pytest.mark.parametrize('k,prefix,leads',[(1,22,[6,12]),(20,3,list(range(6,127,6)))])
def test_actual_training_origins(k,prefix,leads):
    p=protocol(plan(k),'script')
    assert p['training_ar_tail_k']==p['evaluation_ar_tail_k']==k
    assert p['independent_prefix_predictions']==prefix
    assert p['tail_lead_hours']==leads


def records():
    p=protocol(plan(1),'script');baseline=np.ones((24,5,29));baseline[23]*=4
    row={'protocol':p,'checkpoint_sha256':'ck','loss_by_step_field_level':(baseline*.8).tolist(),
         'baseline_by_step_field_level':baseline.tolist()}
    return [copy.deepcopy(row) for _ in range(15)],p


def test_all_positions_and_separate_tail():
    rows,p=records();r=summarize(rows,p,'ck',2000)
    assert r['improvement_percent']==pytest.approx(20)
    assert r['baseline_mean_loss']==pytest.approx(145*27/24)
    assert r['baseline_tail_mean_loss']==pytest.approx(145*5/2)
    assert sum(r['per_field'].values())==pytest.approx(r['mean_loss'])


@pytest.mark.parametrize('problem',['wrong_k','wrong_checkpoint','incomplete','nan'])
def test_rejects_invalid_or_mixed_results(problem):
    rows,p=records()
    if problem=='wrong_k':rows[0]['protocol']['evaluation_ar_tail_k']=20
    if problem=='wrong_checkpoint':rows[0]['checkpoint_sha256']='other'
    if problem=='incomplete':rows.pop()
    if problem=='nan':rows[0]['loss_by_step_field_level'][0][0][0]=float('nan')
    with pytest.raises(ValueError):summarize(rows,p,'ck',2000)
