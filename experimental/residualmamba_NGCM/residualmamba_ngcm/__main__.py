"""Run with PYTHONPATH=experimental/residualmamba_NGCM python -m residualmamba_ngcm."""
import argparse
from pathlib import Path
from .config import Config
from . import artifacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['reuse-v1','preflight','stats','cache','verify-cache','census','qualify','baseline-audit','train','swa','evaluate','bias','report'])
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--run-root',required=True,type=Path)
    for arg in ('donor-root','store','checkpoint','gc-checkpoint','cache','quota-audit','resume','model-checkpoint','eval-root','e1-manifest','gc-eval-times','gc-logged-origins','bias-table','gpu-smoke','native-smoke','resume-smoke'):
        p.add_argument('--'+arg,type=Path)
    p.add_argument('--checkpoint-sha256');p.add_argument('--gc-sha256')
    p.add_argument('--steps',type=int,help='Stop at this update without changing the fixed 10k LR schedule')
    p.add_argument('--indices',type=int,nargs='+',help='Build selected cache chunks; READY still requires all chunks')
    p.add_argument('--warm',action='store_true');p.add_argument('--native',action='store_true')
    a=p.parse_args();cfg=Config.load(a.config);root=a.run_root.resolve()
    if not str(root).startswith('/scratch/'):
        raise ValueError('Run artifacts must be on scratch')
    if a.eval_root is not None and not str(a.eval_root.resolve()).startswith('/scratch/'):
        raise ValueError('Evaluation and report outputs must be on scratch')
    from . import pipeline
    if a.stage=='reuse-v1':
        if not a.donor_root:p.error('reuse-v1 requires --donor-root')
        from .reuse import import_inputs
        import_inputs(a.donor_root,root,cfg);return
    if a.stage=='qualify':
        if not a.gpu_smoke or not a.native_smoke or not a.resume_smoke:p.error('qualify requires --gpu-smoke, --native-smoke and --resume-smoke evidence')
        pipeline.qualify(root,a.gpu_smoke,a.native_smoke,a.resume_smoke);return
    if a.stage=='preflight':
        for key in ('donor-root','store','checkpoint','checkpoint_sha256','gc_checkpoint','gc_sha256','cache','quota_audit'):
            if getattr(a,key) is None:p.error('--'+key.replace('_','-')+' is required for preflight')
        pipeline.preflight(a,cfg);return
    if a.stage=='report':
        from .report import render
        if not a.eval_root:p.error('--eval-root is required')
        render(a.eval_root/'summary.npz',a.eval_root/'report');return
    manifest,store,geometry=pipeline.context(root,cfg)
    if manifest.get('reused_inputs') and a.stage in ('stats','cache'):
        raise ValueError('Imported v1 statistics/cache are immutable; skip their build stages')
    def backbone():
        from .ngcm import load
        return load(manifest['checkpoint'],manifest['checkpoint_sha256'],manifest['store'])[0]
    if a.stage=='stats':
        from .stats import compute
        compute(backbone(),store,cfg,root/'stats.npz',artifacts.sha256(root/'preflight.json'));return
    if a.stage=='cache':
        from .cache import build
        build(backbone(),store,manifest['cache'],artifacts.sha256(root/'preflight.json'),a.indices);return
    stats,_=artifacts.load_arrays(root/'stats.npz')
    if a.stage=='verify-cache':
        from .cache import verify
        verify(backbone(),store,manifest['cache'],stats,root/'verify-cache.json');return
    from .timeline import split_segments
    from .cache import load_chunk
    segments=split_segments(store.times,'train')[0]
    val_segments=split_segments(store.times,'val')[0]
    load=lambda a0:load_chunk(store,manifest['cache'],a0,stats,cfg,geometry['static'],geometry['latitude'],geometry['longitude'])
    if a.stage=='census':
        from .train import census
        census(val_segments,load,geometry['latitude'],root/'census.json',
            {'stats':artifacts.sha256(root/'stats.npz'),'cache':artifacts.sha256(Path(manifest['cache'])/'READY.json')});return
    if a.stage=='bias':
        from .evaluate import bias_table
        bias_table(backbone(),store,root/'bias.npz',
            {'preflight':artifacts.sha256(root/'preflight.json'),'stats':artifacts.sha256(root/'stats.npz')},
            stats,root/'baseline-audit');return
    if a.stage=='baseline-audit':
        from .evaluate import baseline_audit
        baseline_audit(backbone(),store,stats,root/'baseline-audit',
            {'preflight':artifacts.sha256(root/'preflight.json'),'stats':artifacts.sha256(root/'stats.npz')});return
    identity,manifest,store,geometry,stats=pipeline.training_identity(root,cfg)
    if a.stage=='swa':
        from .swa import average
        average(root/'checkpoints',cfg,root/'swa_02000_08000.pkl');return
    from .model import read_model_config,make_model,zero_memory
    base=read_model_config(manifest['gc_checkpoint'],manifest['gc_sha256'])
    model=make_model(base,cfg,geometry['latitude'],geometry['longitude'])
    if a.stage=='train':
        from .train import run
        run(model,cfg,segments,val_segments,load,geometry['latitude'],root/'checkpoints',identity,a.steps,a.resume);return
    from . import evaluate
    from .timeline import evaluation_origins
    import jax
    import numpy as np
    if a.e1_manifest:
        if not a.gc_eval_times or not a.gc_logged_origins:p.error('E1 requires actual GC eval times and logged origins')
        origins=evaluate.parity_origins(a.e1_manifest,a.gc_eval_times,a.gc_logged_origins)
    else:
        origins=evaluation_origins(store.times,2023)
    if not a.eval_root:p.error('--eval-root is required')
    bias=None
    if a.bias_table:
        bias,biasmeta=artifacts.load_arrays(a.bias_table);bias=bias['bias']
        if not biasmeta.get('train_only') or biasmeta['upstream']!={'preflight':artifacts.sha256(root/'preflight.json'),'stats':artifacts.sha256(root/'stats.npz')}:
            raise ValueError('Bias table provenance mismatch')
    if a.native or bias is not None:
        model=None;params=None;memory={}
    else:
        if not a.model_checkpoint:p.error('--model-checkpoint, --native or --bias-table is required')
        saved=artifacts.load_checkpoint(a.model_checkpoint,identity);params=saved['params']
        _,memory=model.init(jax.random.PRNGKey(cfg.seed),np.zeros((128,64,cfg.channels),np.float32))
        memory=zero_memory(memory)
    eval_identity=artifacts.digest({'run':identity,'checkpoint':artifacts.sha256(a.model_checkpoint) if a.model_checkpoint else None,
        'native':a.native,'bias':artifacts.sha256(a.bias_table) if a.bias_table else None,
        'origins':[str(t) for t in origins],'warm':a.warm})
    evaluate.run(backbone(),store,model,params,memory,stats,cfg,origins,a.eval_root,eval_identity,a.warm,bias)


if __name__=='__main__':main()
