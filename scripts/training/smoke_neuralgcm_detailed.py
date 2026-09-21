#!/usr/bin/env python3
"""Full-size GPU integration checks against an immutable production snapshot.

Smoke artifacts are isolated from production checkpoints. Only the number of
training updates is shortened; architecture, optimizer, BPTT and rollout lengths
come from the locked production configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

REQUIRED_CHECKS = {
    'zero_residual_40', 'cache_live_24', 'chunk_carry', 'bptt_reference',
    'physical_stop_gradient', 'pretrain_96', 'pretrain_resume', 'fixed_sample_fit',
    'stage_transfer', 'live20_cold', 'live20_warm', 'finetune_resume',
    'validation_cold', 'validation_warm', 'frozen_backbone',
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate_reports(manifest, root, runner_hash):
    """Fail closed on missing, failed, stale, partial, or CPU-only arm reports."""
    reports = {}
    for run_id, config in manifest['configs'].items():
        path = Path(root) / run_id / 'report.json'
        report = json.loads(path.read_text())
        expected = {'passed': True, 'run_id': run_id, 'source_id': manifest['source_id'],
                    'runner_sha256': runner_hash, 'config_sha256': config['sha256'],
                    'gpu_measured': True}
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f'{run_id}: invalid {key}')
        if report.get('not_a_production_gate', False):
            raise ValueError(f'{run_id}: pilot-only statistics cannot release production')
        if not REQUIRED_CHECKS <= report.get('checks', {}).keys():
            raise ValueError(f'{run_id}: incomplete smoke coverage')
        if any(report['checks'][key].get('passed') is not True for key in REQUIRED_CHECKS):
            raise ValueError(f'{run_id}: failed smoke check')
        reports[run_id] = {'sha256': file_hash(path), 'seconds': report['seconds']}
    return reports



def compare_gradient_reference(actual, reference):
    """Compare distinct FP32 AD lowerings with scale-aware error bounds.

    Independent AD/rematerialization changes reduction order. Relative error at
    an individual cancellation-heavy entry is ill-conditioned. Bound every
    leaf's maximum error, global relative L2 error, and gradient direction.
    Exact replay and cached/live checks retain their original stricter checks.
    """
    import jax
    import numpy as np
    left, left_tree = jax.tree_util.tree_flatten(actual)
    right, right_tree = jax.tree_util.tree_flatten(reference)
    if left_tree != right_tree or not left:
        raise AssertionError('Reference gradient structure differs')
    differences, xs, ys, ratios = [], [], [], []
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('Reference gradient shape or finiteness differs')
        error = float(np.max(np.abs(a-b), initial=0))
        scale = max(float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
        ratio = error / (1e-6 + 1e-5*scale)
        if ratio > 1:
            raise AssertionError(f'Reference gradient leaf error exceeds FP32 budget: {ratio}')
        ratios.append(ratio)
        differences.append(error)
        xs.append(a.ravel())
        ys.append(b.ravel())
    x, y = np.concatenate(xs), np.concatenate(ys)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    relative_l2 = float(np.linalg.norm(x-y) / max(nx, ny, 1e-30))
    cosine = float(np.clip(x@y/(nx*ny), -1, 1)) if nx and ny else (1.0 if nx == ny else 0.0)
    if relative_l2 > 1e-5 or cosine < .99999:
        raise AssertionError(f'Reference gradient norm/direction differs: {relative_l2}, {cosine}')
    return dict(max_abs=max(differences), relative_l2=relative_l2, cosine=cosine,
                max_leaf_error_budget_ratio=max(ratios), relative_l2_limit=1e-5,
                leaf_atol=1e-6, leaf_scale_rtol=1e-5, cosine_minimum=.99999)


def run_arm(args, manifest, config, resources):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.cache import CacheReader, live_record
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint, save_checkpoint, transfer_parent
    from src.models.neuralgcm_residual.checks import tree_comparison, backbone_parameter_digest
    from src.models.neuralgcm_residual.data import STEP
    from src.models.neuralgcm_residual.evaluate import evaluate_origins, select_checkpoint
    from src.models.neuralgcm_residual import finetune
    from src.models.neuralgcm_residual.io import append_jsonl, read_json, write_json
    from src.models.neuralgcm_residual.kernels import DecodedTrainer, make_optimizer
    from src.models.neuralgcm_residual.model import zero_memory, parameter_counts
    from src.models.neuralgcm_residual.native_state import stop
    from src.models.neuralgcm_residual.pretrain import run_pretrain
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities

    started = time.perf_counter()
    output = args.output / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if not devices or any(d.platform != 'gpu' for d in devices):
        raise RuntimeError('These integration checks require an actual GPU')
    runtime = build_runtime(config, resources, stage='pretrain')
    b, trainer = runtime.backbone, runtime.trainer
    reader = CacheReader(resources['cache_root'], runtime.store, 'train')
    ids = runtime_identities(runtime, config, manifest['source_id'], reader.manifest['producer_id'])
    frozen_hash = backbone_parameter_digest(b.model)
    report = dict(passed=False, run_id=args.run_id, source_id=manifest['source_id'],
                  runner_sha256=file_hash(__file__), config_sha256=manifest['configs'][args.run_id]['sha256'],
                  identities=ids, resolved_config=config.to_dict(), checks={}, gpu_measured=True,
                  devices=[d.device_kind for d in devices], **parameter_counts(runtime.params, runtime.memory))

    def finite(tree, *, fp32=False):
        for value in jax.device_get(jax.tree_util.tree_leaves(tree)):
            a = np.asarray(value)
            if not np.isfinite(a).all():
                raise FloatingPointError('Nonfinite state, loss, gradient, or optimizer value')
            if fp32 and a.dtype.kind == 'f' and a.dtype != np.float32:
                raise AssertionError(f'Unexpected training dtype: {a.dtype}')

    def norm(tree):
        return float(np.sqrt(sum(np.sum(np.asarray(x, np.float64)**2)
                                 for x in jax.device_get(jax.tree_util.tree_leaves(tree))
                                 if np.asarray(x).dtype != jax.dtypes.float0)))

    def distance(left, right):
        return norm(jax.tree_util.tree_map(lambda x, y: np.asarray(x)-np.asarray(y), left, right))

    def passed(name, **value):
        report['checks'][name] = dict(passed=True, **value)
        append_jsonl(output / 'progress.jsonl', dict(check=name, elapsed=time.perf_counter()-started, **value))
        write_json(output / 'in_progress.json', report)
        print(json.dumps({'passed': name, **value}), flush=True)

    def record(value):
        return trainer.record(value['origin_state'], value['baseline_state'], value['forcing'],
                              value['target'], runtime.known(value['origin_state'], value['forcing']))

    finite((runtime.params, runtime.memory), fp32=True)
    val = read_json(resources['validation_origins'])
    origin = val['origins'][0]
    inputs, forcing = runtime.store.inputs_and_forcing(b.model, origin)
    initial = b.encode(inputs, forcing)
    physical = baseline = initial
    memory = zero_memory(runtime.memory)
    apply = jax.jit(runtime.branch.apply)
    zero_checks = []
    for lead in range(1, 41):
        delta, memory = apply(runtime.params, memory, jax.random.PRNGKey(lead),
                              runtime.normalization.inputs(runtime.adapter.features(physical)),
                              runtime.known(physical, forcing))
        np.testing.assert_array_equal(delta, 0)
        physical = runtime.adapter.apply_increment(b.advance_6h(physical, forcing),
                                                   runtime.normalization.increment(delta))
        baseline = b.advance_6h(baseline, forcing)
        finite((physical, baseline, memory), fp32=True)
        tree_comparison(physical, baseline, exact=True)
        b.assert_time(initial, physical, lead)
        if lead in (1, 20, 40):
            zero_checks.append(dict(lead=lead, **tree_comparison(b.decode(physical, forcing),
                                                                b.decode(baseline, forcing), exact=True)))
    passed('zero_residual_40', origin=origin, comparisons=zero_checks)

    # Exercise all upstream gradients using a deliberately nonzero test head.
    probe = jax.tree_util.tree_map(lambda x: x, runtime.params)
    heads = [name for name in probe if 'native_zero_head' in name]
    if len(heads) != 1:
        raise AssertionError('Expected one native increment head')
    name = heads[0]
    probe[name] = dict(probe[name], w=jnp.full_like(probe[name]['w'], 1e-5),
                       b=jnp.full_like(probe[name]['b'], 1e-3))
    h = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), runtime.memory)
    key = jax.random.PRNGKey(config.seed)
    cached, fresh = [], []
    for i in range(24):
        value = reader[i]
        cached.append(record(value))
        live = live_record(b, runtime.store, value['origin'])
        live['target'] = value['target']
        fresh.append(record(live))
    a = trainer.cached_gradients(probe, h, key, cached)
    c = trainer.cached_gradients(probe, h, key, fresh)
    finite((a, c), fp32=False)
    comparisons = {label: tree_comparison(x, y) for label, x, y in zip(
        ('loss', 'memory', 'rng', 'gradients'), a, c, strict=True)}
    if comparisons['gradients']['cosine'] is None or comparisons['gradients']['cosine'] < .99999:
        raise AssertionError('Cached/live gradient direction differs')
    opt = trainer.optimizer.init(probe)
    comparisons['update'] = tree_comparison(trainer.checked_update(probe, opt, a[3]),
                                           trainer.checked_update(probe, opt, c[3]))
    passed('cache_live_24', records=24, comparisons=comparisons)

    def forward(params, memory, rng, records):
        losses = []
        for item in records:
            rng, subkey = jax.random.split(rng)
            loss, memory, _ = trainer.forward(params, memory, subkey, *item)
            losses.append(loss)
        return losses, memory, rng

    whole = forward(probe, h, key, cached)
    left = forward(probe, h, key, cached[:12])
    right = forward(probe, stop(left[1]), left[2], cached[12:])
    passed('chunk_carry', comparison=tree_comparison(whole, (left[0]+right[0], right[1], right[2]), exact=True))
    reference = jax.grad(lambda p: jnp.mean(jnp.stack(forward(p, h, key, cached[:2])[0])))(probe)
    tape = trainer.cached_gradients(probe, h, key, cached[:2])[3]
    recurrent = norm({n: v for n, v in tape.items() if 'temporal' in n})
    if recurrent <= 0:
        raise AssertionError('No gradient reaches the recurrent layers')
    passed('bptt_reference', comparison=compare_gradient_reference(tape, reference), temporal_gradient_norm=recurrent)
    native, known, base, force, target = cached[0]
    physical_grads = jax.grad(lambda x, k, s, f: trainer.forward(probe, h, key, x, k, s, f, target)[0],
                             argnums=(0, 1, 2, 3), allow_int=True)(native, known, base, force)
    if norm(physical_grads) != 0:
        raise AssertionError('Physical-input gradient was not stopped')
    passed('physical_stop_gradient', norm=0.0)
    del fresh, a, c, reference, tape, whole, left, right, physical_grads

    # Execute the actual production pretraining loop over one full 96-record segment.
    class Segment:
        times = reader.times[:96]
        def __getitem__(self, index):
            return reader[index]
    if len(Segment.times) != 96 or any(np.datetime64(y)-np.datetime64(x) != STEP
                                      for x, y in zip(Segment.times, Segment.times[1:])):
        raise AssertionError('Smoke segment is not 96 consecutive six-hour records')
    short = SimpleNamespace(**vars(config))
    short.pretrain_epochs = 1
    original_cached, original_update = trainer.cached_gradients, trainer.checked_update
    observed, last_opt = [], None
    midpoint = output / 'pretrain_midpoint.pkl'
    def capture_cached(params, memory, rng, records):
        if len(observed) == 2:
            save_checkpoint(midpoint, stage='pretrain', identities=ids, params=params, optimizer=last_opt,
                            memory=memory, rng=rng,
                            cursor=dict(epoch=0, segment=0, chunk=2, update=2, processed_timestamps=48))
        result = original_cached(params, memory, rng, records)
        finite((result[1], result[3]), fp32=True)
        observed.append(dict(loss=float(result[0]), temporal_gradient_norm=norm(
            {n: v for n, v in result[3].items() if 'temporal' in n})))
        return result
    def capture_update(params, optimizer, gradients):
        nonlocal last_opt
        result = original_update(params, optimizer, gradients)
        finite(result, fp32=True)
        last_opt = result[1]
        return result
    trainer.cached_gradients, trainer.checked_update = capture_cached, capture_update
    try:
        run_pretrain(runtime, Segment(), short, output/'pretrain_full', ids)
    finally:
        trainer.cached_gradients, trainer.checked_update = original_cached, original_update
    parent = output/'pretrain_full/checkpoint_pass_01.pkl'
    full = load_checkpoint(parent, stage='pretrain', identities=ids)
    if len(observed) != 4 or max(x['temporal_gradient_norm'] for x in observed[1:]) <= 0:
        raise AssertionError('Four production updates did not train the recurrent branch')
    changed = distance(runtime.params, full['params'])
    if changed <= 0:
        raise AssertionError('Pretraining did not update branch parameters')
    passed('pretrain_96', updates=4, bptt_steps=24, records=96, parameter_change_l2=changed, metrics=observed)
    run_pretrain(runtime, Segment(), short, output/'pretrain_resumed', ids, resume=midpoint)
    resumed = load_checkpoint(output/'pretrain_resumed/checkpoint_pass_01.pkl', stage='pretrain', identities=ids)
    if full['cursor'] != resumed['cursor']:
        raise AssertionError('Pretraining resume cursor differs')
    passed('pretrain_resume', resumed_from_update=2, final_update=4,
           comparison=tree_comparison(tuple(full[k] for k in ('params','optimizer','memory','rng')),
                                      tuple(resumed[k] for k in ('params','optimizer','memory','rng')), exact=True))

    # A fixed real sample must be learnable using the production optimizer.
    p = runtime.params
    opt = trainer.optimizer.init(p)
    fits = []
    for _ in range(12):
        result = trainer.cached_gradients(p, zero_memory(runtime.memory), key, cached[:1])
        fits.append(float(result[0]))
        p, opt, _ = trainer.checked_update(p, opt, result[3])
        finite((p, opt), fp32=True)
    final_loss = float(forward(p, zero_memory(runtime.memory), key, cached[:1])[0][0])
    if not np.isfinite(fits+[final_loss]).all() or final_loss >= fits[0]:
        raise AssertionError(f'Fixed-sample loss did not decrease: {fits[0]} -> {final_loss}')
    passed('fixed_sample_fit', updates=12, initial_loss=fits[0], final_loss=final_loss, losses=fits)
    del p, opt, resumed

    transferred, transfer = transfer_parent(parent, {k: ids[k] for k in
        ('architecture', 'backbone', 'native_schema', 'normalization', 'dataset')})
    tree_comparison(transferred, full['params'], exact=True)
    passed('stage_transfer', resets=transfer['reset'], parent_sha256=transfer['sha256'])
    fine_ids = dict(ids, parent=transfer['sha256'])
    runtime.trainer = DecodedTrainer(runtime.branch, runtime.adapter, runtime.normalization, b,
                                     trainer.weather_loss, make_optimizer(config.finetune_optimizer))
    fine_trainer = runtime.trainer
    short.finetune_updates, short.finetune_validate_every = 2, 1
    original_episode = finetune.episode_gradients
    episode_audits = []

    def audited_episode(tr, params, initial_memory, rng, **kwargs):
        warm, t0 = kwargs.get('warm', False), np.datetime64(kwargs['origin'], 'h')
        calls = dict(inputs=[], targets=[], advances=0, encodes=0, feedback=0, nonzero_corrections=0)
        if not episode_audits:
            tree_comparison(params, transferred, exact=True)
            tree_comparison(rng, jax.random.PRNGKey(config.seed), exact=True)
        last = {}
        actual_b, actual_store, actual_forward = kwargs['backbone'], kwargs['store'], tr.forward
        class AuditedBackbone:
            model = actual_b.model
            def encode(self, inputs, forcing):
                calls['encodes'] += 1
                return actual_b.encode(inputs, forcing)
            def advance_6h(self, state, forcing):
                if calls['advances']:
                    tree_comparison(state, last['corrected'], exact=True)
                    tree_comparison(forcing, last['forcing'], exact=True)
                    calls['feedback'] += 1
                last['input'] = state
                last['baseline'] = actual_b.advance_6h(state, forcing)
                last['forcing'] = forcing
                calls['advances'] += 1
                return last['baseline']
            def assert_time(self, *values):
                return actual_b.assert_time(*values)
        class AuditedStore:
            def inputs_and_forcing(self, model, time):
                calls['inputs'].append(str(np.datetime64(time, 'h')))
                return actual_store.inputs_and_forcing(model, time)
            def frame(self, time):
                calls['targets'].append(str(np.datetime64(time, 'h')))
                return actual_store.frame(time)
        def audited_forward(*values):
            tree_comparison(values[3], tr.normalization.inputs(tr.adapter.features(last['input'])), exact=True)
            tree_comparison(values[4], runtime.known(last['input'], last['forcing']), exact=True)
            tree_comparison(values[5], last['baseline'], exact=True)
            if calls['advances'] == 1:
                calls['initial_memory_norm'] = norm(values[1])
                if (warm and calls['initial_memory_norm'] <= 0) or (not warm and calls['initial_memory_norm'] != 0):
                    raise AssertionError('Cold/warm memory initialization is incorrect')
            result = actual_forward(*values)
            finite(result, fp32=True)
            last['corrected'] = result[2]
            if distance(result[2], last['baseline']) > 0:
                calls['nonzero_corrections'] += 1
            return result
        tr.forward = audited_forward
        try:
            result = original_episode(tr, params, initial_memory, rng,
                                      **dict(kwargs, backbone=AuditedBackbone(), store=AuditedStore()))
            jax.block_until_ready(result)
        finally:
            tr.forward = actual_forward
        expected_inputs = ([str(t0-np.timedelta64(lag,'h')) for lag in (24,18,12,6)] if warm else [])+[str(t0)]
        expected_targets = [str(t0+lead*STEP) for lead in range(1,21)]
        if (calls['inputs'] != expected_inputs or calls['targets'] != expected_targets
                or calls['encodes'] != (5 if warm else 1) or calls['advances'] != 20
                or calls['feedback'] != 19 or calls['nonzero_corrections'] != 20
                or result['valid_hours'] != list(range(6,121,6))):
            raise AssertionError(f'Closed-loop causal audit failed: {calls}')
        finite((result['gradients'],result['physical'],result['memory']), fp32=True)
        if norm(result['gradients']) <= 0 or norm(result['memory']) <= 0:
            raise AssertionError('Live forecast has no branch gradient or recurrent memory')
        episode_audits.append(dict(warm=warm, origin=str(t0), loss=result['loss'],
                                   gradient_norm=norm(result['gradients']), lead_losses=result['lead_losses'], **calls))
        return result

    fine_update = fine_trainer.checked_update
    fine_update_count = 0
    def checked_fine_update(params, optimizer, gradients):
        nonlocal fine_update_count
        if fine_update_count == 0:
            tree_comparison(optimizer, fine_trainer.optimizer.init(params), exact=True)
        result = fine_update(params, optimizer, gradients)
        finite(result, fp32=True)
        fine_update_count += 1
        return result
    fine_trainer.checked_update = checked_fine_update
    finetune.episode_gradients = audited_episode
    try:
        finetune.run_finetune(runtime, short, output/'finetune_full', ids, parent)
        for label, audit in zip(('live20_cold','live20_warm'), episode_audits, strict=True):
            passed(label, **audit)
        finetune.run_finetune(runtime, short, output/'finetune_resumed', ids, parent,
                              resume=output/'finetune_full/checkpoint_000001.pkl')
    finally:
        finetune.episode_gradients = original_episode
        fine_trainer.checked_update = fine_update
    fine_full = load_checkpoint(output/'finetune_full/checkpoint_000002.pkl', stage='finetune', identities=fine_ids)
    fine_resumed = load_checkpoint(output/'finetune_resumed/checkpoint_000002.pkl', stage='finetune', identities=fine_ids)
    if fine_full['cursor'] != fine_resumed['cursor'] or norm(fine_full['memory']) != 0:
        raise AssertionError('Fine-tuning resume cursor or episode reset differs')
    if distance(full['params'], fine_full['params']) <= 0:
        raise AssertionError('Fine-tuning did not update the residual parameters')
    passed('finetune_resume', comparison=tree_comparison(
        tuple(fine_full[k] for k in ('params','optimizer','memory','rng')),
        tuple(fine_resumed[k] for k in ('params','optimizer','memory','rng')), exact=True))

    for warm in (False, True):
        label = 'warm' if warm else 'cold'
        metrics = evaluate_origins(runtime, fine_full['params'], dict(val, origins=[origin]),
                                   output/('validation_'+label), warm=warm)
        if metrics['failures'] or metrics['completed_origins'] != 1 or not np.isfinite(metrics['loss']):
            raise AssertionError('Validation failed')
        if not warm:
            select_checkpoint(output/'smoke_selection.json', output/'finetune_full/checkpoint_000002.pkl',
                              metrics, split='val', epoch_or_update=2)
        passed('validation_'+label, loss=metrics['loss'], baseline_loss=metrics['baseline_loss'], origin=origin)
    if backbone_parameter_digest(b.model) != frozen_hash:
        raise AssertionError('Frozen NeuralGCM weights changed')
    passed('frozen_backbone', sha256=frozen_hash)
    if set(report['checks']) != REQUIRED_CHECKS:
        raise AssertionError('Incomplete detailed smoke report')
    report.update(passed=True, seconds=time.perf_counter()-started,
                  gpu_allocator_stats=[d.memory_stats() for d in devices],
                  peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  test_only_budgets={'pretrain_passes':1,'pretrain_records':96,'finetune_updates':2,
                                     'validation_origins':1},
                  production_checkpoints_modified=False)
    write_json(output/'report.json', report, immutable=True)
    print(json.dumps({'passed':True,'report':str(output/'report.json'),'seconds':report['seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if bool(args.run_id) == args.aggregate:
        parser.error('Choose exactly one of --run-id or --aggregate')
    raw = json.loads((args.experiment_root/'manifest.json').read_text())
    source = Path(raw['source_root'])
    for path in (source/'third_party/graphcast', source/'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.config import load_config
    from src.models.neuralgcm_residual.io import write_json, versions
    from src.models.neuralgcm_residual.worker import require_receipt
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']:
        raise ValueError('Smoke-test libraries differ from production')
    require_receipt(args.experiment_root, 'verify-cache', 'res2p8', manifest['source_id'])
    if args.aggregate:
        reports = validate_reports(manifest, args.output, file_hash(__file__))
        write_json(args.output/'PASSED.json', dict(passed=True, source_id=manifest['source_id'],
                   runner_sha256=file_hash(__file__), reports=reports), immutable=True)
        print(json.dumps({'all_four_arms_passed': True, 'reports':reports}), flush=True)
    else:
        config = load_config(manifest['configs'][args.run_id]['path'])
        run_arm(args, manifest, config, resolve_resources(manifest, config.resolution_id))


if __name__ == '__main__':
    main()
