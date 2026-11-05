"""The loss report compares repeated weather anchors, including a partial epoch."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'diagnostic_2k.py'
SPEC=importlib.util.spec_from_file_location('diagnostic_2k_report',PATH)
worker=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(worker)


def test_matched_anchor_curve_and_duplicate_recovery_rows(tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib'))
    directory=tmp_path/'checkpoints';directory.mkdir()
    rows=[]
    for step in range(1,2001):
        index=(step-1)%424;epoch=(step-1)//424
        rows.append({'step':step,'a0':4+24*index,'loss':(1+.01*index)/(1+.1*epoch)})
    (directory/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows+[rows[-1]]))
    (directory/'step_02000.pkl').write_bytes(b'synthetic report fixture, not a real checkpoint')
    for step in (1000,2000):
        (directory/f'validation_{step:05d}.json').write_text(json.dumps({
            'mean_loss':.8,'baseline_mean_loss':1.,'improvement_percent':20.}))
    monkeypatch.setattr(worker,'training_identity',lambda *a:'test')
    monkeypatch.setattr(worker,'load_checkpoint',lambda *a:{'step':2000})
    worker.report(tmp_path,{'scope':'synthetic report unit test'})
    result=json.loads((tmp_path/'report/summary.json').read_text())
    assert result['matched_anchor_count']==304
    assert result['duplicate_log_rows_after_recovery']==1
    assert result['matched_loss_reduction_percent']==pytest.approx(100*(1-1/1.4))
    assert (tmp_path/'report/loss-curves.png').stat().st_size>1000
    assert (tmp_path/'report/COMPLETE.json').is_file()
