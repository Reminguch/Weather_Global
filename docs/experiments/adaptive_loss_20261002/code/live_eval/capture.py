"""Archive an atomic committed checkpoint while training continues; never edit training files."""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
from common import read,write,atomic,digest


def capture(root,force=False):
    plan=read(root/'plan.json');sys.path.insert(0,plan['source_root'])
    from experimental.adaptive_loss.checkpoint import validate
    for hardware,folder in plan['campaigns'].items():
        for mode in ('fixed','calibrated','adaptive'):
            run=Path(folder)/('trial_'+mode)
            if not (run/'last.pkl').exists():continue
            # Writer atomically replaces last.pkl. One read observes one complete inode.
            raw=(run/'last.pkl').read_bytes();state=pickle.loads(raw);config=read(run/'config.json')
            validate(state,digest(config),state['params'],state['optimizer'],config['updates'])
            steps=[r['update'] for r in state['metrics']];seconds=[float(r['seconds']) for r in state['metrics']]
            if any(s<0 or not __import__('math').isfinite(s) for s in seconds):raise ValueError('Invalid recorded training time')
            import numpy as np
            cumulative=np.r_[0.,np.cumsum(seconds)]
            history=dict(hardware=hardware,mode=mode,config=config,update=state['update'],
                         captured_at=time.time(),cumulative_step_seconds=cumulative.tolist(),validations=state['validations'])
            write(root/'history'/(hardware+'_'+mode+'.json'),history)
            prior=[read(p) for p in (root/'snapshots').glob(hardware+'_'+mode+'_*/metadata.json')] if (root/'snapshots').exists() else []
            previous=max([x['update'] for x in prior if x['kind']=='last'] or [-plan['snapshot_interval_updates']])
            due=force or state['update']==config['updates'] or state['update']>=previous+plan['snapshot_interval_updates']
            if not due:continue
            for kind,item in [('last',state),('best',state['best'])]:
                step=item['update']
                if step==0:continue  # Frozen baseline is evaluated independently, once.
                identity=hardware+'_'+mode+'_step'+str(step)
                directory=root/'snapshots'/identity
                payload=pickle.dumps(dict(params=item['params'],update=step),protocol=5)
                payload_hash=hashlib.sha256(payload).hexdigest()
                import jax
                def parameter_hash(params):
                    h=hashlib.sha256(str(jax.tree_util.tree_structure(params)).encode())
                    for leaf in jax.tree_util.tree_leaves(params):
                        x=np.asarray(leaf);h.update(str((x.shape,x.dtype.str)).encode());h.update(x.tobytes())
                    return h.hexdigest()
                canonical_hash=parameter_hash(item['params'])
                if (directory/'metadata.json').exists():
                    existing=read(directory/'metadata.json')
                    existing_hash=existing.get('parameter_sha256')
                    if existing_hash is None:existing_hash=parameter_hash(pickle.loads((directory/'checkpoint.pkl').read_bytes())['params'])
                    if existing_hash!=canonical_hash:raise ValueError('Conflicting saved parameters at '+identity)
                    continue
                atomic(directory/'checkpoint.pkl',payload)
                meta=dict(id=identity,hardware=hardware,mode=mode,kind=kind,update=step,
                          training_compute_seconds=float(cumulative[step]),captured_at=time.time(),
                          original_last_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=payload_hash,parameter_sha256=canonical_hash,config=config)
                write(directory/'metadata.json',meta)
                print(json.dumps(dict(captured=identity,kind=kind,training_compute_seconds=meta['training_compute_seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--force',action='store_true');a=p.parse_args();capture(a.root,a.force)
