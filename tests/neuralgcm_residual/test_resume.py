from pathlib import Path
from types import SimpleNamespace
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.neuralgcm_residual.pretrain import run_pretrain
from src.models.neuralgcm_residual.checkpoint import load_checkpoint
from test_gradients import fixture, assert_tree


class Reader:
    times=[str(t) for t in np.arange(np.datetime64('2020-01-02T00'),np.datetime64('2020-01-26T00'),np.timedelta64(6,'h'))]
    def __getitem__(self,i):
        return {'origin_state':jnp.float32(i/100), 'baseline_state':jnp.float32(.5),
                'forcing':jnp.float32(1), 'target':jnp.float32(.4)}


def test_resume_replays_rng_memory_optimizer_cursor_and_archives_logs(tmp_path):
    trainer,p,h,key=fixture()
    runtime=SimpleNamespace(trainer=trainer,params=p,memory=h,known=lambda s,f:jnp.float32(0))
    cfg=SimpleNamespace(seed=22,segment_steps=96,bptt_steps=24,pretrain_epochs=2)
    ids={'source':'one','dataset':'same'}
    uninterrupted=tmp_path/'full'
    run_pretrain(runtime,Reader(),cfg,uninterrupted,ids)
    interrupted=tmp_path/'resume'
    original=trainer.cached_gradients
    calls=0
    def fail_after_five(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==6: raise RuntimeError('simulated interruption')
        return original(*a,**kw)
    trainer.cached_gradients=fail_after_five
    with pytest.raises(RuntimeError):
        run_pretrain(runtime,Reader(),cfg,interrupted,ids)
    trainer.cached_gradients=original
    run_pretrain(runtime,Reader(),cfg,interrupted,ids,resume=interrupted/'checkpoint_pass_01.pkl')
    left=load_checkpoint(uninterrupted/'checkpoint_pass_02.pkl',stage='pretrain',identities=ids)
    right=load_checkpoint(interrupted/'checkpoint_pass_02.pkl',stage='pretrain',identities=ids)
    assert left['cursor']==right['cursor']
    for k in ('params','optimizer','memory','rng'): assert_tree(left[k],right[k])
    assert len((interrupted/'metrics.jsonl').read_text().splitlines())==8
    assert list(interrupted.glob('*.archive'))
