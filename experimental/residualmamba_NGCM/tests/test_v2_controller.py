"""Scheduling checks: atomic checkpoints, dense eval, endpoint and explicit stop."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/training'))
from v2_ngcm_controller import next_actions,pending_checkpoints


def checkpoint(root,step,receipt=True):
    p=root/'checkpoints'/f'step_{step:05d}.pkl';p.parent.mkdir(exist_ok=True)
    p.write_bytes(b'test')
    if receipt:p.with_suffix('.json').write_text(json.dumps({'step':step}))
    return p


def test_atomic_checkpoint_discovery_and_concurrent_eval(tmp_path):
    assert next_actions(tmp_path)==[('train-2000',True,['--endpoint','2000'])]
    p=checkpoint(tmp_path,100,receipt=False)
    assert pending_checkpoints(tmp_path)==[]
    p.with_suffix('.json').write_text(json.dumps({'step':100}))
    assert pending_checkpoints(tmp_path)==[p]
    assert [a[0] for a in next_actions(tmp_path)]==['train-2000','validate-dense']
    p=tmp_path/'eval/tail20/step_00100/summary.json';p.parent.mkdir(parents=True);p.write_text('{}')
    assert pending_checkpoints(tmp_path)==[]


def test_final_eval_precedes_report_and_explicit_stop(tmp_path):
    checkpoint(tmp_path,2000)
    assert next_actions(tmp_path)==[('validate-dense',True,[])]
    p=tmp_path/'eval/tail20/step_02000/summary.json';p.parent.mkdir(parents=True);p.write_text('{}')
    assert next_actions(tmp_path)==[('report',False,[])]
    p=tmp_path/'report/COMPLETE.json';p.parent.mkdir();p.write_text('{}')
    assert next_actions(tmp_path)==[]
    p=tmp_path/'controller/user-stop.json';p.parent.mkdir();p.write_text('{}')
    with pytest.raises(SystemExit):next_actions(tmp_path)


def test_tail1_checkpoint_labels_and_shared_protocol(tmp_path):
    from v2_diagnostic import training_config
    cfg=training_config({'ar_tail_k':1})
    assert (cfg.schema_version,cfg.output_channels,cfg.ar_tail_k,cfg.truth_prefix_steps)==(2,145,1,23)
    assert (cfg.bptt_steps,cfg.segment_steps,cfg.v24().ar_tail_k)==(24,96,1)
    (tmp_path/'plan.json').write_text(json.dumps({'arm':'tail1'}))
    p=checkpoint(tmp_path,100)
    wrong=tmp_path/'eval/tail20/step_00100/summary.json';wrong.parent.mkdir(parents=True);wrong.write_text('{}')
    assert pending_checkpoints(tmp_path)==[p]
    right=tmp_path/'eval/tail1/step_00100/summary.json';right.parent.mkdir(parents=True);right.write_text('{}')
    assert pending_checkpoints(tmp_path)==[]


def test_tail1_training_uses_tail1_inputs_but_comparison_uses_tail20(tmp_path,monkeypatch):
    import v2_diagnostic as worker
    import v2_tail1_inputs
    import tail1_extension
    from residualmamba_ngcm import cache,model
    plan={'ar_tail_k':1,'arm':'tail1','identity':'verified','orchestration_hashes':{},'prerequisites':{}}
    monkeypatch.setattr(worker,'read_json',lambda p:plan if p.name=='plan.json' else {'cache':'/tail1'})
    monkeypatch.setattr(worker,'digest',lambda _: 'verified')
    monkeypatch.setattr(worker,'source_identity',lambda: 'source')
    monkeypatch.setattr(worker,'pipeline_context',lambda *a:({'cache':'/tail20','gc_checkpoint':'unused','gc_sha256':'sha'},object(),{'static':0,'latitude':0,'longitude':0}))
    monkeypatch.setattr(worker,'load_arrays',lambda _:({},{}))
    monkeypatch.setattr(v2_tail1_inputs,'check_receipts',lambda _:None)
    monkeypatch.setattr(model,'read_model_config',lambda *a:None)
    monkeypatch.setattr(model,'make_model',lambda *a:None)
    monkeypatch.setattr(cache,'load_chunk',lambda *a:'tail20')
    monkeypatch.setattr(tail1_extension,'load_chunk',lambda *a:'tail1')
    assert worker.context(tmp_path)[-1](4)=='tail1'
    assert worker.context(tmp_path,evaluation=True)[-1](4)=='tail20'
