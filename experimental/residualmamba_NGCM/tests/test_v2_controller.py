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
