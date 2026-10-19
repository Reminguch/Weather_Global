#!/usr/bin/env python3
"""Prepare the fixed K22 comparison away from recovered development windows."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(workflow, recovery_of=None):
    import numpy as np
    from scripts.experiments.frozen_runtime import freeze_source
    from scripts.experiments.v24_2023_qualification import development_initialization_times
    workflow = workflow.resolve()
    if workflow.exists():
        raise FileExistsError(workflow)
    principal = ROOT / 'artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009'
    previous = ROOT / 'artifacts/evaluations/graphcast_common_grid_2022_20261016'
    original = read(principal / 'protocol.json')
    development_file = ROOT / ('artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/'
                              'runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/'
                              'swa_step00500-02000.json')
    archived = read(development_file)
    development = development_initialization_times(original, archived)
    # Recovered K22 checkpoint/EMA/alpha sweeps use seed=0, 32 samples,
    # 1460 six-hour 2022 times, and a common 64-future-step eligibility pool.
    recovered_indices = np.sort(np.random.default_rng(0).choice(
        np.arange(1, 1396), size=32, replace=False)).tolist()
    if recovered_indices != archived['chosen_idx']:
        raise ValueError('Recovered K22 sampling rule differs from development32')
    exposed = sorted(set(development + original['pilot']['initialization_times']))
    parse = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    first = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
    starts = [(first + dt.timedelta(days=i)).strftime('%Y-%m-%dT%H:%M:%SZ') for i in range(365)]
    retained = [s for s in starts if all(abs(parse(s)-parse(d)) > dt.timedelta(hours=246)
                                       for d in exposed)]
    if len(retained) != 69:
        raise ValueError(f'Unexpected eligible-date count: {len(retained)}')
    workflow.mkdir(parents=True)
    for name in ('logs', 'shards', 'pilot', 'references', 'report', 'submissions', 'checkpoints', 'audit'):
        (workflow / name).mkdir()
    write(workflow / 'initializations.json', {'schema_version': 1, 'initializations': retained})
    write(workflow / 'initializations_smoke.json', {'schema_version': 1, 'initializations': retained[:1]})
    shared = Path('/scratch/gpfs/DABANIN/lm8598/Weather_Global')
    evidence = [shared / 'logs/v22cl_K22_wb2grid_10950439.out']
    clone = shared / 'results/eval2023_code_test/fresh_clone'
    evidence += [clone / 'slurm_pilots' / name for name in (
        'v22cl_K22_ema_eval.slurm', 'v22cl_K22_ema_sweep_eval.slurm',
        'v22cl_r1_K22_sweep_eval.slurm', 'v22cl_K22_alpha_sweep.slurm')]
    recovered = []
    for path in evidence:
        destination = workflow / 'audit' / path.name
        shutil.copy2(path, destination)
        recovered.append({'source': str(path), 'copy': str(destination), 'sha256': sha(destination)})
    write(workflow / 'development_audit.json', {
        'development32': development, 'numerical_pilots': original['pilot']['initialization_times'],
        'excluded_origins': exposed, 'development_file': str(development_file),
        'development_file_sha256': sha(development_file), 'chosen_idx': recovered_indices,
        'recovered_k22_rule': 'seed0; 32 without replacement from indices1..1395 in 2022',
        'evidence': recovered,
        'exclusion': 'closed input/verification intervals [start-6h,start+240h] must not intersect',
        'scope': 'recovered development32 and four numerical pilots; not a claim of untouched 2022',
        'remaining_limits': 'Unrecorded historical exposure cannot be excluded; five seasonal gaps remain.',
    })
    checkpoint_source = shared / 'results/v22cl_r1_EMA/K22_ema.pkl'
    checkpoint = workflow / 'checkpoints/K22_swa_5k_10k.pkl'
    shutil.copy2(checkpoint_source, checkpoint)
    if sha(checkpoint) != sha(checkpoint_source):
        raise ValueError('K22 checkpoint copy differs')
    old = read(previous / 'protocol.json')
    snapshot = freeze_source(ROOT, workflow / 'source')
    data = workflow / 'data'
    recovery_record = None
    if recovery_of is not None:
        recovery_of = recovery_of.resolve()
        prior = read(recovery_of / 'protocol.json')
        if prior['initialization_times'] != retained or prior['checkpoint']['sha256'] != sha(checkpoint):
            raise ValueError('Recovery must preserve dates and checkpoint')
        destination = data / 'inputs1'
        destination.mkdir(parents=True)
        for start in retained:
            name = start.replace('-', '').replace(':', '')+'.nc'
            original_input = Path(prior['inputs1']) / name
            if not original_input.is_file():
                raise FileNotFoundError(original_input)
            (destination / name).symlink_to(original_input.resolve())
        recovery_record = {'workflow': str(recovery_of), 'protocol_sha256': sha(recovery_of / 'protocol.json'),
                           'protocol_hash': prior['protocol_hash'],
                           'reason': 'Repair extra-truth writer argument; reuse all69 verified inputs',
                           'reused_input_count': len(retained)}
    protocol = {
        'schema_version': 1, 'experiment': 'k22_common_grid_development_excluded_2022',
        'prepared_date': '2026-10-18', 'workspace': str(ROOT),
        'source_root': str(workflow / 'source'), 'source_digest': snapshot['source_digest'],
        'initialization_manifest': str(workflow / 'initializations.json'),
        'initialization_manifest_sha256': sha(workflow / 'initializations.json'),
        'initialization_times': retained, 'lead_hours': list(range(6, 241, 6)), 'target_steps': 40,
        'development_audit': str(workflow / 'development_audit.json'),
        'development_audit_sha256': sha(workflow / 'development_audit.json'),
        'checkpoint': {'path': str(checkpoint), 'sha256': sha(checkpoint),
                       'historical_path': str(checkpoint_source),
                       'resolved_historical_path': str(checkpoint_source.resolve()),
                       'averaging': 'historical filename EMA; symlink points to SWA5k-10k; exact recipe unverified'},
        'baseline': {**old['gc1'], 'sha256': old['gc1'].get('checkpoint_sha256', old['gc1'].get('sha256'))},
        'statistics': old['statistics'], 'stats_dir': old['stats_dir'],
        'inputs1': str(data / 'inputs1'), 'common_truth': str(data / 'common_truth'),
        'reuse_data_root': old['data_root'],
        'architecture': {'resolution': 1., 'mesh_size': 5, 'width': 512, 'baseline_msg_steps': 16,
                         'residual_msg_steps': 2, 'temporal_location': 'mesh_processor_interleaved',
                         'temporal_hidden_size': 128, 'temporal_d_inner': 128, 'temporal_d_state': 16,
                         'temporal_d_conv': 4, 'temporal_dt_rank': 'auto', 'temporal_layers': 2,
                         'temporal_conv_bias': True, 'temporal_zero_init_out': True,
                         'temporal_bias': False, 'temporal_dropout': 0.},
        'forecast': 'legacy K22 factories; independent full feedback; zero memory at each start; BF16 network',
        'reference': 'datasets/graphcast_v2/2022-240x121_equiangular_with_poles_conservative.zarr/',
        'ens_reference': 'datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr/',
        'verification': old['verification'],
        'bootstrap': {'draws': 10000, 'seed': 20261018, 'block_days': [5, 10],
                      'method': 'paired stratified nonwrapping blocks within each contiguous date window',
                      'intervals': 'pointwise; conditional on these five periods; limited independent weather'},
        'plot_directory': str(ROOT / 'plots/analyze_models/images/resolution_eval/k22_common_grid_2022_20261018'),
        'table_directory': str(ROOT / 'plots/analyze_models/data/resolution_eval/k22_common_grid_2022_20261018'),
        'interpretation': 'Historical K22 supplementary comparison; recovered-development-window exclusion, '
                          'uneven seasonal coverage, not annual or untouched-test-year performance.',
        'recovery_of': recovery_record,
        'resources': {'evidence_job': '15750888', 'elapsed': '00:08:31', 'max_rss_gib': 10.87,
                      'pilot': {'mem': '16G', 'time': '00:20:00'},
                      'production': {'shards': 9, 'mem': '16G', 'time': '00:45:00'}},
    }
    protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write(workflow / 'protocol.json', protocol)
    print(json.dumps({'workflow': str(workflow), 'starts': len(retained),
                      'protocol_hash': protocol['protocol_hash']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--recovery-of', type=Path)
    args = parser.parse_args()
    prepare(args.workflow, args.recovery_of)
