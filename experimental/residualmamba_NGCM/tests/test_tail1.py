from dataclasses import asdict,replace
import numpy as np
import pytest
from tail1_extension import Tail1Config,window,origin_index,chunk_baseline
from residualmamba_ngcm.config import Config


def test_only_tail_and_derived_prefix_change():
    a,b=asdict(Config()),asdict(Tail1Config())
    assert {k for k in a if a[k]!=b[k]}=={'ar_tail_k','truth_prefix_steps'}
    v=Tail1Config().v24()
    assert (v.bptt_steps,v.ar_tail_k,v.truth_prefix_steps,v.segment_steps)==(24,1,23,96)
    assert Tail1Config().architecture()==Config().architecture()
    with pytest.raises(ValueError):replace(Tail1Config(),learning_rate=2e-4)


def test_tail1_matches_independent_gc_window_and_forcing_origins():
    a0=8;truth=np.arange(100,dtype=np.float32)
    frames=list(truth[a0-1:a0+23]);baseline=[];expected=[]
    for i in range(24):
        pair=(frames[i],frames[i+1]);expected.append(pair)
        prediction=.3*pair[0]+.6*pair[1];baseline.append(prediction)
        if 22<=i<23:frames.append(prediction)
    prefix=truth[a0-1:a0+23]
    for i in range(24):
        np.testing.assert_array_equal(window(prefix,baseline,i),expected[i])
        assert origin_index(a0,i)==a0+min(i,22)
    assert window(prefix,baseline,23)==(truth[a0+22],baseline[22])
    # No target at or beyond the tail origin's future enters the tail window.
    changed=truth.copy();changed[a0+23:]+=1e6
    for i in range(24):np.testing.assert_array_equal(window(changed[a0-1:a0+23],baseline,i),expected[i])


def test_native_rollout_call_count_and_no_reencoding(monkeypatch):
    import tail1_extension as ext
    from types import SimpleNamespace
    calls=[]
    def fake(bb,store,origin,steps):
        calls.append((origin,steps))
        for k in range(1,steps+1):yield np.asarray([origin,k],np.float32),None
    monkeypatch.setattr(ext,'rollout',fake)
    result=chunk_baseline(None,SimpleNamespace(times=np.arange(100)),4)
    assert calls==[(i,1) for i in range(4,26)]+[(26,2)]
    np.testing.assert_array_equal(result[-2:],[[26,1],[26,2]])


def test_controller_stage_and_resume(tmp_path):
    from scripts.training.ngcm_tail1_controller import next_actions
    assert [x[0] for x in next_actions(tmp_path)]==['qa','cache-0','cache-1']
    (tmp_path/'qa').mkdir();(tmp_path/'qa/COMPLETE.json').write_text('{}')
    for i in range(2):(tmp_path/f'cache-lane-{i}.json').write_text('{}')
    assert next_actions(tmp_path)==[('checks',True,[])]
    (tmp_path/'cache-verified.json').write_text('{}')
    assert next_actions(tmp_path)==[('train-1000',True,['--endpoint','1000'])]


def test_qa_runs_in_distinct_processes(tmp_path):
    from scripts.training.ngcm_tail1_controller import batch
    args,script=batch({'root':str(tmp_path),'python':'/scratch/python','code_snapshot':'/scratch/code'},'qa',True,[])
    assert '--qos=gpu-test' in args and '--gres=gpu:h200:1' in args
    for stage in ('qa-full','qa-split','qa-resume','qa-compare'):assert 'tail1_2k.py '+stage in script
    assert script.count('tail1_2k.py ')==4


def test_tail1_training_resume_and_memory_segment_boundary(tmp_path,monkeypatch):
    import test_contracts
    monkeypatch.setattr(test_contracts,'Config',Tail1Config)
    test_contracts.test_exact_training_resume_and_segment_carry(tmp_path)
