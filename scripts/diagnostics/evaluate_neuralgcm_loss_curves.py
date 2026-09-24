#!/usr/bin/env python3
"""Replay all frozen epoch checkpoints for per-field K=1 loss curves."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['experiment_root'])
    raw = json.loads((root / 'manifest.json').read_text())
    source = Path(raw['source_root'])
    for path in (source / 'third_party/graphcast', source / 'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.cache import CacheReader, live_record
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.config import load_config, FIELDS
    from src.models.neuralgcm_residual.io import sha256, versions
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities

    started = time.perf_counter()
    if any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('Run this model replay in a Slurm GPU allocation')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = load_experiment(root)
    if versions() != manifest['libraries']:
        raise ValueError('Frozen environment versions differ')
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest, config.resolution_id)
    rt = build_runtime(config, resources, stage='pretrain')
    reader = CacheReader(resources['cache_root'], rt.store, 'val')
    identities = runtime_identities(rt, config, manifest['source_id'], reader.manifest['producer_id'])
    choices = plan['runs'][args.run_id]
    if args.smoke:
        choices = [choices[0], choices[-1]]
    params = []
    for choice in choices:
        path = Path(choice['checkpoint'])
        if sha256(path) != choice['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed: {path}')
        saved = load_checkpoint(path, stage='pretrain', identities=identities)
        if saved['cursor']['update'] != choice['update']:
            raise ValueError('Checkpoint optimizer update mismatch')
        params.append(jax.device_put(saved['params']))
        del saved
    b, trainer = rt.backbone, rt.trainer
    before = backbone_parameter_digest(b.model)
    latitude = np.asarray(b.model.data_coords.horizontal.latitudes)
    levels = np.asarray(b.model.data_coords.vertical.centers)
    nodes, weights64 = np.polynomial.legendre.leggauss(len(latitude))
    np.testing.assert_allclose(nodes, np.sin(latitude), atol=2e-6)
    weights64 /= weights64.sum()
    area = jnp.asarray(weights64, jnp.float32)

    def mse(decoded, target):
        return jnp.stack([jnp.mean(jnp.sum((decoded[f] - target[f]) ** 2 * area, axis=-1), axis=-1)
                          for f in FIELDS])

    @jax.jit
    def baseline_mse(item):
        return mse(b.decode(item[2], item[3]), item[4])

    @jax.jit
    def inference(p, memory, key, item):
        native, known, baseline, forcing, target = item
        delta, memory = rt.branch.apply(p, memory, key, native, known)
        state = rt.adapter.apply_increment(baseline, rt.normalization.increment(delta))
        return memory, mse(b.decode(state, forcing), target)

    @jax.jit
    def reference(p, memory, key, item):
        loss, next_memory, state = trainer.forward(p, memory, key, *item)
        return loss, next_memory, b.decode(state, item[3])

    def tree_max_difference(x, y):
        return max((float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(c, np.float64))))
                    for a, c in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y), strict=True)
                    if np.asarray(a).size), default=0.)

    score_weights = np.stack([np.asarray(trainer.weather_loss.level_weights, np.float64) /
                              np.asarray(trainer.weather_loss.scales[f], np.float64) ** 2 / len(FIELDS)
                              for f in FIELDS])
    samples = set(np.linspace(0, len(reader) - 1, 8, dtype=int).tolist())
    indices = sorted(samples) if args.smoke else range(len(reader))
    total = np.zeros((len(params) + 1, len(FIELDS), len(levels)), np.float64)
    memories = [zero_memory(rt.memory) for _ in params]
    previous, since_reset, checks = None, 0, []
    for count, i in enumerate(indices, 1):
        record = reader[i]
        origin = np.datetime64(record['origin'], 'h')
        if np.datetime64(record['target_time'], 'h') != origin + np.timedelta64(6, 'h'):
            raise ValueError('Invalid six-hour target time')
        if since_reset == 96 or (previous is not None and origin - previous != np.timedelta64(6, 'h')):
            memories = [zero_memory(rt.memory) for _ in params]
            since_reset = 0
        item = trainer.record(record['origin_state'], record['baseline_state'], record['forcing'],
                              record['target'], rt.known(record['origin_state'], record['forcing']))
        item = jax.device_put(item)
        key = jax.random.PRNGKey(i)
        values = [baseline_mse(item)]
        prior_last = memories[-1]
        for k, p in enumerate(params):
            memories[k], errors = inference(p, memories[k], key, item)
            values.append(errors)
        values = np.asarray(jax.device_get(jnp.stack(values)))
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite error at {origin}')
        np.testing.assert_allclose(values[1], values[0], rtol=2e-5, atol=1e-12)
        if i in samples:
            ref_loss, ref_mem, decoded = jax.device_get(reference(params[-1], prior_last, key, item))
            independent = np.stack([np.mean(np.sum((np.asarray(decoded[f], np.float64) -
                               np.asarray(record['target'][f], np.float64)) ** 2 * weights64, axis=-1), axis=-1)
                                    for f in FIELDS])
            np.testing.assert_allclose(values[-1], independent, rtol=2e-4, atol=1e-12)
            np.testing.assert_allclose(np.sum(values[-1] * score_weights), ref_loss, rtol=2e-5)
            diff = tree_max_difference(memories[-1], ref_mem)
            if diff != 0:
                raise AssertionError('Production memory differs')
            check = dict(origin=str(origin), production_memory_max_abs=diff,
                         independent_numpy_mse=True, production_forward_loss=float(ref_loss))
            if args.smoke:
                live = live_record(b, rt.store, record['origin'])
                differences = {k: tree_max_difference(live[k], record[k])
                               for k in ('origin_state', 'baseline_state', 'forcing')}
                if any(differences.values()):
                    raise AssertionError('Live and cached model inputs differ')
                check['live_cache_max_abs'] = differences
            checks.append(check)
        total += values
        previous, since_reset = origin, since_reset + 1
        if count % 100 == 0 or count == len(indices):
            value = dict(records=count, total=len(indices), checkpoints=len(params),
                         elapsed_seconds=time.perf_counter() - started)
            print(json.dumps(value), flush=True)
            (args.output / 'progress.json').write_text(json.dumps(value, indent=2) + '\n')
    total /= len(indices)
    reconstructed = np.sum(total * score_weights[None], axis=(1, 2))
    if not args.smoke:
        np.testing.assert_allclose(reconstructed[0], choices[0]['validation_loss'], rtol=2e-5)
        np.testing.assert_allclose(reconstructed[1:], [c['validation_loss'] for c in choices], rtol=2e-5)
    for choice in choices:
        if sha256(choice['checkpoint']) != choice['checkpoint_sha256']:
            raise AssertionError('Checkpoint changed during evaluation')
    if backbone_parameter_digest(b.model) != before:
        raise AssertionError('Frozen backbone changed')
    np.savez_compressed(args.output / 'physical_mse.npz', baseline_mse=total[0], residual_mse=total[1:],
                        fields=np.asarray(FIELDS), pressure_hpa=levels,
                        updates=np.asarray([c['update'] for c in choices]),
                        epochs=np.asarray([c['epoch'] for c in choices]))
    report = dict(completed=True, smoke=args.smoke, run_id=args.run_id, checkpoints=choices,
                  records=len(indices), plan_sha256=sha256(args.plan), script_sha256=sha256(__file__),
                  source_id=manifest['source_id'], identities=identities,
                  job_id=os.environ.get('SLURM_JOB_ID'), devices=[d.device_kind for d in jax.devices()],
                  captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started, checks=checks,
                  zero_checkpoint_matches_baseline=True, checkpoint_and_backbone_unchanged=True,
                  reconstructed_current_v2_loss=reconstructed.tolist(),
                  memory_policy='independent chronological memory per checkpoint; reset every 96 records or at gaps',
                  physical_initialization='ERA5 at each origin; K=1 six-hour forecast',
                  artifacts={'physical_mse.npz': sha256(args.output / 'physical_mse.npz')})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed': True, 'report': str(args.output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
