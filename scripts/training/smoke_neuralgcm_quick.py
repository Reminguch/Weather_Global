#!/usr/bin/env python3
"""Independent real-data smoke with 96-record statistics, never a production gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--detailed-runner', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.experiment_root/'manifest.json').read_text())
    source = Path(raw['source_root'])
    for path in (source/'third_party/graphcast', source/'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.worker import native_setup
    from src.models.neuralgcm_residual.config import load_config
    from src.models.neuralgcm_residual.cache import (CacheReader, initialize_cache, producer_identity,
        produce_shard, valid_k1_origins, verify_cache, live_record)
    from src.models.neuralgcm_residual.data import origin_manifest
    from src.models.neuralgcm_residual.normalization import fit_statistics
    from src.models.neuralgcm_residual.io import read_json, write_json, sha256, versions
    if not all(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('Independent smoke requires an actual GPU')
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']:
        raise ValueError('Dependency versions differ from the frozen experiment')
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest, config.resolution_id)
    production_cache = Path(resources['cache_root'])
    pilot = args.output/'pilot'/args.run_id
    pilot.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(pilot/'SCOPE.json', dict(not_a_production_gate=True, training_records=96,
        normalization_scope='first_96_eligible_training_records_only',
        production_source_id=manifest['source_id'], runner_sha256=sha256(args.detailed_runner),
        wrapper_sha256=sha256(__file__)), immutable=True)
    backbone, store, adapter = native_setup(resources)
    origins = valid_k1_origins(store, 'train')[:96]
    if len(origins) != 96:
        raise AssertionError('Insufficient real training data')
    producer = producer_identity(backbone, adapter, store)
    mini = pilot/'cache'
    cache = initialize_cache(mini, producer, {'train': origins})
    production = read_json(production_cache/'manifest.json')
    if production['producer_id'] != cache['producer_id']:
        raise ValueError('Existing cache is incompatible with this frozen source')
    for shard in cache['shards']:
        sid = shard['id']
        original = next(s for s in production['shards'] if s['id'] == sid)
        if original != shard:
            raise ValueError('Pilot cache timestamp selection changed')
        meta, archive = production_cache/(sid+'.json'), production_cache/(sid+'.zip')
        if meta.exists() and archive.exists():
            receipt = read_json(meta)
            if (receipt['producer_id'] != cache['producer_id'] or receipt['records'] != len(shard['origins'])
                    or receipt['sha256'] != sha256(archive)):
                raise ValueError('Existing cache shard failed identity/hash checks')
            os.symlink(archive, mini/archive.name)
            shutil.copyfile(meta, mini/meta.name)
            method = 'reuse_verified_readonly_archive'
        else:
            produce_shard(mini, sid, backbone, adapter, store)
            method = 'generate_in_isolated_pilot_cache'
        print(json.dumps({'pilot_cache_shard':sid,'method':method}), flush=True)
    verify_cache(mini, fresh_record=lambda origin: live_record(backbone, store, origin))
    reader = CacheReader(mini, store, 'train')
    def records():
        for i in range(len(reader)):
            record = reader[i]
            inputs, forcing = store.inputs_and_forcing(backbone.model, record['valid'])
            record['next_truth_state'] = jax.device_get(backbone.encode(inputs, forcing))
            record['truth'] = store.frame(record['origin'])
            yield record
            if (i+1) % 24 == 0:
                print(json.dumps({'pilot_statistics_records':i+1}), flush=True)
    statistics = pilot/'statistics.json'
    fit_statistics(records(), adapter, backbone.model.data_coords.horizontal.latitudes,
                   dataset_id=store.identity, output=statistics)
    validation = pilot/'validation_origins.json'
    write_json(validation, origin_manifest(store.times, 'val', 4, 20, store.identity), immutable=True)
    resources = dict(resources, cache_root=str(mini), statistics=str(statistics), validation_origins=str(validation))
    write_json(pilot/'READY.json', dict(not_a_production_gate=True, records=96,
        seconds=time.perf_counter()-started, statistics_sha256=sha256(statistics),
        producer_id=cache['producer_id']), immutable=True)
    print(json.dumps({'pilot_preparation_passed':True,'seconds':time.perf_counter()-started}), flush=True)
    spec = importlib.util.spec_from_file_location('independent_detailed_smoke', args.detailed_runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_arm(args, manifest, config, resources)
    report = read_json(args.output/args.run_id/'report.json')
    if report.get('not_a_production_gate') is not True or not report['passed']:
        raise AssertionError('Invalid independent smoke scope/report')
    print(json.dumps({'independent_smoke_passed':True,'run_id':args.run_id,
                      'not_a_production_gate':True,'seconds':time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
