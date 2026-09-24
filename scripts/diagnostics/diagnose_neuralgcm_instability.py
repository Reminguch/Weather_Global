#!/usr/bin/env python3
"""Read-only checkpoint ablations and independent K=1 validation audit on a GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--origins', type=int, default=4)
    parser.add_argument('--audit-k1', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((args.experiment_root / 'manifest.json').read_text())
    source = Path(raw['source_root'])
    for p in (source / 'third_party/graphcast', source / 'third_party/neuralgcm', source):
        sys.path.insert(0, str(p))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.cache import CacheReader, live_record
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.config import load_config, FIELDS
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.native_state import field_value
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.io import sha256, versions
    started = time.perf_counter()
    if any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('Real GPU required')
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']:
        raise ValueError('Library versions differ from training')
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest, config.resolution_id)
    rt = build_runtime(config, resources, stage='pretrain')
    b, a, n, trainer = rt.backbone, rt.adapter, rt.normalization, rt.trainer
    reader = CacheReader(resources['cache_root'], rt.store, 'val')
    identities = runtime_identities(rt, config, manifest['source_id'], reader.manifest['producer_id'])
    stage = args.experiment_root / 'runs' / args.run_id / 'seed22/pretrain'
    checkpoint = stage / f'checkpoint_pass_{args.epoch:02d}.pkl'
    saved = load_checkpoint(checkpoint, stage='pretrain', identities=identities)
    params = saved['params']
    frozen_before = backbone_parameter_digest(b.model)
    del saved
    report = dict(run_id=args.run_id, epoch=args.epoch, checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint), identities=identities,
                  script_sha256=sha256(__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                  devices=[d.device_kind for d in jax.devices()],
                  started_at=datetime.now(timezone.utc).isoformat(), variants=[], impulses=[])

    def plain(value):
        if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [plain(v) for v in value]
        if isinstance(value, np.ndarray): return plain(value.tolist())
        if isinstance(value, np.generic): return plain(value.item())
        if isinstance(value, float) and not np.isfinite(value): return str(value)
        return value

    def emit(kind, value):
        entry = plain(dict(kind=kind, elapsed_seconds=time.perf_counter()-started, **value))
        with (args.output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(entry, allow_nan=False) + '\n')
        print(json.dumps(entry, allow_nan=False), flush=True)

    def save():
        tmp = args.output / 'report.tmp'
        tmp.write_text(json.dumps(plain(report), indent=2, allow_nan=False) + '\n')
        tmp.replace(args.output / 'report.json')

    @jax.jit
    def correction(p, memory, key, state, forcing):
        return rt.branch.apply(p, memory, key, n.inputs(a.features(state)), rt.known(state, forcing))

    apply_increment = jax.jit(a.apply_increment)

    @jax.jit
    def score_state(state, forcing, truth):
        decoded = b.decode(state, forcing)
        fields = trainer.weather_loss.field_scores(decoded, truth)
        return dict(loss=trainer.weather_loss(decoded, truth), fields=fields,
                    decoded_finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(decoded[k])) for k in FIELDS])))

    @jax.jit
    def native_stats(state):
        features = a.features(state)
        stats = {}
        offset = 0
        for name, units, count in a.channels:
            x = features[..., offset:offset+count]
            z = (x-n.mean[offset:offset+count])/n.scale[offset:offset+count]
            stats[name] = dict(min=jnp.min(x), max=jnp.max(x), mean=jnp.mean(x),
                               rms=jnp.sqrt(jnp.mean(x*x)), negative_fraction=jnp.mean(x < 0),
                               input_z_rms=jnp.sqrt(jnp.mean(z*z)), input_z_max=jnp.max(jnp.abs(z)))
            offset += count
        ps = b.model.from_nondim_units(jnp.exp(a.grid.to_nodal(field_value(state, 'log_surface_pressure'))), 'Pa')
        stats['surface_pressure_pa'] = dict(min=jnp.min(ps), max=jnp.max(ps), mean=jnp.mean(ps))
        return stats

    def score(state, forcing, truth):
        return plain(jax.device_get(score_state(state, forcing, truth)))

    channel_masks = {}
    offset = 0
    for name, _, count in a.channels:
        mask = np.zeros(a.output_size, np.float32)
        mask[offset:offset+count] = 1
        channel_masks[name] = mask
        offset += count
    origins = json.loads(Path(resources['validation_origins']).read_text())['origins'][:args.origins]
    emit('loaded', dict(origins=origins, checkpoint_sha256=report['checkpoint_sha256']))
    for origin_index, origin in enumerate(origins):
        inputs, forcing = rt.store.inputs_and_forcing(b.model, origin)
        initial = b.encode(inputs, forcing)
        truths = [rt.store.frame(np.datetime64(origin, 'h')+np.timedelta64(6*i, 'h')) for i in range(1,21)]
        # Every intervention shares the same initial state, forcing, and targets.
        # postprocess_only scores corrections but feeds back the uncorrected solver state.
        variants = [('baseline', 0.0), ('full', 1.0), ('first_only', 1.0),
                    ('reset_memory', 1.0), ('postprocess_only', 1.0), ('scale_0p1', 0.1)]
        for mode, scale in variants:
            state, memory, key = initial, zero_memory(rt.memory), jax.random.PRNGKey(22)
            steps = []
            for lead in range(1,21):
                old = state
                advanced = b.advance_6h(old, forcing)
                b.assert_time(initial, advanced, lead)
                before = score(advanced, forcing, truths[lead-1])
                corrected = advanced
                delta = None
                if mode != 'baseline' and (mode != 'first_only' or lead == 1):
                    key, subkey = jax.random.split(key)
                    if mode == 'reset_memory': memory = zero_memory(rt.memory)
                    delta, memory = correction(params, memory, subkey, old, forcing)
                    corrected = apply_increment(advanced, n.increment(delta) * scale)
                state = advanced if mode == 'postprocess_only' else corrected
                after = score(corrected, forcing, truths[lead-1])
                step = dict(lead_hours=lead*6, before_correction=before, after_correction=after,
                            native=plain(jax.device_get(native_stats(corrected))))
                if delta is not None:
                    d = np.asarray(delta)
                    step['normalized_delta_rms'] = float(np.sqrt(np.mean(d.astype(np.float64)**2)))
                    step['normalized_delta_max_abs'] = float(np.max(np.abs(d)))
                steps.append(step)
                emit('lead', dict(origin=origin, mode=mode, lead_hours=lead*6,
                                  before_loss=before['loss'], after_loss=after['loss'],
                                  fields=after['fields']))
                if not isinstance(after['loss'], (float, int)) or not after['decoded_finite']:
                    break
            report['variants'].append(dict(origin=origin, mode=mode, scale=scale, steps=steps))
            save()
        # A single impulse followed by an unmodified NGCM advance separates native
        # channel sensitivity from recurrent-memory or second-correction effects.
        base6 = b.advance_6h(initial, forcing)
        _, subkey = jax.random.split(jax.random.PRNGKey(22))
        delta, _ = correction(params, zero_memory(rt.memory), subkey, initial, forcing)
        physical_delta = n.increment(delta)
        if origin_index == 0:
            np.savez_compressed(args.output/'first_correction.npz',
                                delta=np.asarray(delta), physical_delta=np.asarray(physical_delta),
                                base_features=np.asarray(a.features(base6)),
                                corrected_features=np.asarray(a.features(apply_increment(base6, physical_delta))),
                                correction_scale=np.asarray(n.correction_scale),
                                input_scale=np.asarray(n.scale), input_mean=np.asarray(n.mean))
        masks = [('only_'+name, mask) for name, mask in channel_masks.items()]
        masks += [('drop_'+name, 1-mask) for name, mask in channel_masks.items()]
        masks += [('all_scale_0p01', np.full(a.output_size, .01, np.float32))]
        for name, mask in masks:
            state6 = apply_increment(base6, physical_delta*jnp.asarray(mask))
            state12 = b.advance_6h(state6, forcing)
            item = dict(origin=origin, intervention=name,
                        at_6h=score(state6, forcing, truths[0]),
                        at_12h_no_second_correction=score(state12, forcing, truths[1]),
                        native6=plain(jax.device_get(native_stats(state6))),
                        native12=plain(jax.device_get(native_stats(state12))))
            report['impulses'].append(item)
            emit('impulse',dict(origin=origin, intervention=name, loss6=item['at_6h']['loss'],
                                loss12=item['at_12h_no_second_correction']['loss'],
                                fields12=item['at_12h_no_second_correction']['fields']))
        save()
    if args.audit_k1:
        # Repeat every original validation record in its original order and memory policy.
        memory, previous, since_reset = zero_memory(rt.memory), None, 0
        losses, baselines, independent_checks, live_checks = [], [], [], []
        field_sums = {k: np.zeros(2, np.float64) for k in FIELDS}
        sample_indices = set(np.linspace(0, len(reader)-1, 8, dtype=int).tolist())
        native_sample_stats = []
        for i in range(len(reader)):
            record = reader[i]
            if since_reset == 96 or (previous is not None and np.datetime64(record['origin'])-previous != np.timedelta64(6,'h')):
                memory, since_reset = zero_memory(rt.memory), 0
            item = trainer.record(record['origin_state'], record['baseline_state'], record['forcing'],
                                  record['target'], rt.known(record['origin_state'], record['forcing']))
            value, memory, corrected = trainer.forward(params, memory, jax.random.PRNGKey(i), *item)
            baseline = score(record['baseline_state'], record['forcing'], record['target'])
            rescored = score(corrected, record['forcing'], record['target'])
            losses.append(float(value)); baselines.append(baseline['loss'])
            for k in FIELDS:
                field_sums[k] += [baseline['fields'][k], rescored['fields'][k]]
            if i in sample_indices:
                # Independent float64 NumPy reduction, using raw predictions/targets.
                decoded = jax.device_get(b.decode(corrected, record['forcing']))
                lat = b.model.data_coords.horizontal.latitudes
                nodes, weights = np.polynomial.legendre.leggauss(len(lat))
                np.testing.assert_allclose(nodes, np.sin(lat), atol=2e-6)
                weights = weights/weights.sum()
                levels = np.asarray(b.model.data_coords.vertical.centers,np.float64)
                levels /= levels.sum()
                independent = {}
                for k in FIELDS:
                    err = (np.asarray(decoded[k],np.float64)-record['target'][k])/np.asarray(n.manifest['loss_scales'][k])[:,None,None]
                    independent[k] = float(np.sum(np.mean(np.sum(err**2*weights,axis=-1),axis=-1)*levels))
                independent_checks.append(dict(index=i, origin=record['origin'], original=float(value),
                                                numpy_float64=float(np.mean(list(independent.values()))),fields=independent))
                live = live_record(b, rt.store, record['origin'])
                differences = {}
                for name in ['origin_state','baseline_state','forcing']:
                    pairs=zip(jax.tree_util.tree_leaves(live[name]),jax.tree_util.tree_leaves(record[name]))
                    differences[name]=max(float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)))) for x,y in pairs)
                live_checks.append(dict(origin=record['origin'], max_abs_difference=differences))
                native_sample_stats.append(dict(origin=record['origin'], corrected=plain(jax.device_get(native_stats(corrected)))))
            previous, since_reset = np.datetime64(record['origin']), since_reset+1
            if (i+1)%200 == 0:emit('audit_progress',dict(records=i+1, loss=float(np.mean(losses))))
        stored = json.loads((stage/'validation'/f'{args.epoch:06d}'/'one_step.json').read_text())
        initial = json.loads((stage/'validation/000000/one_step.json').read_text())
        report['k1_audit']=dict(records=len(losses), stored=stored, recomputed_loss=float(np.mean(losses)),
                                original_baseline=initial,recomputed_baseline=float(np.mean(baselines)),
                                fields_mean={k:(v/len(losses)).tolist() for k,v in field_sums.items()},
                                numpy_checks=independent_checks,live_cache_checks=live_checks,
                                native_samples=native_sample_stats)
        emit('k1_audit',report['k1_audit'])
        save()
        np.testing.assert_allclose(np.mean(losses),stored['loss'],rtol=1e-5,atol=1e-3)
        np.testing.assert_allclose(np.mean(baselines),initial['loss'],rtol=1e-5,atol=1e-3)
        for check in independent_checks:
            np.testing.assert_allclose(check['original'],check['numpy_float64'],rtol=1e-5,atol=1e-3)
    if backbone_parameter_digest(b.model) != frozen_before or sha256(checkpoint) != report['checkpoint_sha256']:
        raise AssertionError('Diagnostic unexpectedly changed model weights')
    report.update(completed=True, elapsed_seconds=time.perf_counter()-started,
                  frozen_backbone_unchanged=True, production_checkpoint_unchanged=True)
    save()
    emit('completed',dict(report=str(args.output/'report.json')))


if __name__ == '__main__':
    main()
