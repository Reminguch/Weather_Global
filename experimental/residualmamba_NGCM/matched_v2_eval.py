#!/usr/bin/env python3
"""Training-matched v2 evaluation using the immutable qualified training snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matching_context(worker, root):
    # The historical evaluation=True branch forces tail20. Use the training loader.
    return worker.context(root, evaluation=False)


def protocol(plan, script_sha):
    k = plan['ar_tail_k']
    if k not in (1, 20):
        raise ValueError('Only qualified K1 and K20 runs are supported')
    return {'kind': 'training_matched_validation', 'training_ar_tail_k': k,
        'evaluation_ar_tail_k': k, 'run_identity': plan['identity'],
        'training_source_identity': plan['source_identity'], 'evaluator_sha256': script_sha,
        'stats_sha256': plan['prerequisites']['stats.npz'],
        'input_receipt_sha256': plan['prerequisites'].get('tail1-inputs.json', plan['prerequisites']['preflight.json']),
        'truth_prefix_steps': 24-k, 'independent_prefix_predictions': 23-k,
        'tail_lead_hours': list(range(6, 6*(k+1)+1, 6)),
        'validation': 'all 15 fixed 2022 segments / 60 chunks',
        'memory': 'reset each 96-anchor segment; carry through four 24-step chunks',
        'aggregation': 'mean all 24 positions and all chunks/segments; sum five weighted normalized fields',
        'baseline': 'official pretrained NeuralGCM with identical origins, valid times and leads'}


def verify_record(row, contract, checkpoint_sha):
    if row['protocol'] != contract or row['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Mixed evaluation protocol or checkpoint')


def summarize(rows, contract, checkpoint_sha, step):
    import numpy as np
    if len(rows) != 15:
        raise ValueError('All 15 segments are required')
    for row in rows:
        verify_record(row, contract, checkpoint_sha)
    arrays = []
    for key in ('loss_by_step_field_level', 'baseline_by_step_field_level'):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        if values.shape != (15,24,5,29) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or malformed contributions')
        arrays.append(values.mean(axis=0))
    loss, baseline = arrays
    fields = ('temperature','geopotential','u_component_of_wind','v_component_of_wind','specific_humidity')
    values, refs = loss.mean(axis=0).sum(axis=-1), baseline.mean(axis=0).sum(axis=-1)
    total, ref = float(values.sum()), float(refs.sum())
    if ref <= 0:
        raise ValueError('Baseline must be positive')
    start = contract['independent_prefix_predictions']
    return {'protocol': contract, 'step': step, 'checkpoint_sha256': checkpoint_sha,
        'segments': 15, 'chunks': 60, 'mean_loss': total, 'baseline_mean_loss': ref,
        'improvement_percent': 100*(1-total/ref),
        'per_field': dict(zip(fields,values.tolist())), 'baseline_per_field': dict(zip(fields,refs.tolist())),
        'per_step': loss.sum(axis=(1,2)).tolist(), 'baseline_per_step': baseline.sum(axis=(1,2)).tolist(),
        'tail_mean_loss': float(loss[start:].sum(axis=(1,2)).mean()),
        'baseline_tail_mean_loss': float(baseline[start:].sum(axis=(1,2)).mean()),
        'selection': 'all saved checkpoints; final endpoint not validation-selected'}


def evaluate(root, output, step):
    plan = read(root/'plan.json')
    sys.path.insert(0,str(Path(plan['code_snapshot'])/'experimental/residualmamba_NGCM'))
    import v2_diagnostic as worker
    from residualmamba_ngcm.artifacts import load_checkpoint,write_json
    from residualmamba_ngcm.timeline import split_segments
    from dense_eval import predictor,evaluate_segment
    import jax
    import numpy as np
    if jax.devices()[0].device_kind != 'NVIDIA H200':
        raise RuntimeError('Use qualified H200 execution')
    plan,store,geometry,model,load = matching_context(worker,root)
    contract = protocol(plan,sha(Path(__file__)))
    checkpoint = root/'checkpoints'/f'step_{step:05d}.pkl'
    saved = load_checkpoint(checkpoint,plan['identity']);cksha = sha(checkpoint)
    directory = output/f'step_{step:05d}';directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'protocol.json',contract,immutable=True)
    params,memory = jax.device_put((saved['params'],saved['memory']))
    predict = predictor(model,geometry['latitude'],22)
    segments = split_segments(store.times,'val')[0]
    assert len(segments)==15
    deadline = time.monotonic()+3000
    if not (directory/'smoke.json').exists():
        from residualmamba_ngcm.model import zero_memory
        from residualmamba_ngcm.loss import contributions
        batch = load(int(segments[0][0]))
        _,actual,native = predict(params,zero_memory(memory),batch)
        actual,native = np.asarray(actual),np.asarray(native)
        direct = np.stack([np.asarray(contributions(np.zeros_like(t),t,geometry['latitude'])) for t in batch['target']])
        if actual.shape!=(24,5,29) or not np.isfinite(actual).all():
            raise ValueError('Invalid real-batch forward pass')
        np.testing.assert_allclose(native,direct,rtol=2e-6,atol=1e-8)
        write_json(directory/'smoke.json',{'protocol':contract,'checkpoint_sha256':cksha,
            'finite':True,'baseline_scan_vs_direct_max_abs':float(np.max(np.abs(native-direct))),
            'first_chunk_baseline_loss':float(native.mean(axis=0).sum()),
            'first_chunk_model_loss':float(actual.mean(axis=0).sum())},immutable=True)
        print('H200 real-batch smoke passed',flush=True)
        del batch,actual,native,direct
    else:
        verify_record(read(directory/'smoke.json'),contract,cksha)
    for i,segment in enumerate(segments):
        path = directory/f'segment_{i:02d}.json'
        if path.exists():
            verify_record(read(path),contract,cksha);continue
        if time.monotonic()+180>=deadline:
            print('Soft deadline; saved segments will resume',flush=True);return
        loss,baseline = evaluate_segment(predict,params,memory,segment,load,geometry['latitude'],22)
        if not np.isfinite(loss).all() or not np.isfinite(baseline).all():
            raise FloatingPointError('Nonfinite matched evaluation')
        write_json(path,{'protocol':contract,'checkpoint_sha256':cksha,
            'loss_by_step_field_level':loss.tolist(),'baseline_by_step_field_level':baseline.tolist()},immutable=True)
        print(f'matched trainK={plan["ar_tail_k"]} evalK={plan["ar_tail_k"]} checkpoint={step} segment={i+1}/15',flush=True)
    result = summarize([read(directory/f'segment_{i:02d}.json') for i in range(15)],contract,cksha,step)
    write_json(directory/'summary.json',result,immutable=True)
    print(json.dumps({k:result[k] for k in ('step','mean_loss','baseline_mean_loss','improvement_percent')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);a=p.parse_args()
    if not all(str(x.resolve()).startswith('/scratch/') for x in (a.root,a.output)):
        raise ValueError('Use scratch')
    evaluate(a.root.resolve(),a.output.resolve(),a.step)
