#!/usr/bin/env python3
"""Package the submitted computation plus portable launchers; never submit jobs."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.frozen_runtime import file_sha256, verify_source

CAMPAIGN = ROOT / "artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112"
NOVEMBER = ROOT / "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
CONSTANT_GC = ROOT / "artifacts/checkpoints/v24_Ilya/res1_full_gc_constant_lr3em7_20261112"
PACKAGE_NAME = "lianghong_weather_20261112_constant_lr3em7"
WEIGHTS_DOWNLOAD = ROOT / "artifacts/handoffs" / PACKAGE_NAME / f"{PACKAGE_NAME}_weights.tar.gz"
PILOT = ROOT / "artifacts/evaluations/v24_raw_checkpoint_pilot_20261110"
README = ROOT / "docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_sha256(source) != file_sha256(target):
        raise ValueError(f"Transfer identity differs: {target}")


def runtime_lock():
    """Lock the installed dependency closure, without URLs or local pip paths."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    installed = {canonicalize_name(d.metadata['Name']): d for d in metadata.distributions()
                 if d.metadata.get('Name')}
    pending = ['jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'optax',
               'dm-haiku', 'dm-tree', 'jraph', 'jmp', 'numpy', 'scipy', 'xarray',
               'pandas', 'netCDF4', 'zarr', 'dask', 'matplotlib', 'trimesh', 'rtree',
               'fsspec', 'gcsfs', 'pytest', 'packaging']
    pending += [name for name in installed if name.startswith('nvidia-')]
    selected = {}
    environment = {**default_environment(), 'extra': ''}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in selected:
            continue
        distribution = installed[name]
        selected[name] = distribution.version
        for specification in distribution.requires or []:
            requirement = Requirement(specification)
            if requirement.marker is None or requirement.marker.evaluate(environment):
                pending.append(requirement.name)
    return ''.join(f'{name}=={version}\n' for name, version in sorted(selected.items()))


def checksum_file(root):
    records = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(
        f'{file_sha256(p)}  {p.relative_to(root)}\n' for p in records))


def build(destination, *, weights=True):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new handoff directory; existing packages are immutable')
    upstream = CAMPAIGN / 'source'
    original = verify_source(upstream)
    destination.mkdir(parents=True)
    package = destination / PACKAGE_NAME
    package.mkdir()
    for relative in original['files']:
        copy(upstream / relative, package / relative)
    adapters = sorted((ROOT / 'scripts/experiments').glob('*v24_portable*'))
    required = {'run_v24_portable_gcm_lambda.py', 'run_v24_portable_full_gc_ar.py',
                'run_v24_portable_comparison.py', 'run_v24_portable_smoothing.py'}
    if not required.issubset({p.name for p in adapters}):
        raise FileNotFoundError('Portable adapters are not all present')
    for path in adapters:
        copy(path, package / 'scripts/experiments' / path.name)
    for path in sorted((ROOT / 'tests').glob('test_portable*.py')):
        copy(path, package / 'tests' / path.name)
    for path in (ROOT / 'LICENSE', ROOT / 'third_party/graphcast/LICENSE'):
        if path.exists():
            copy(path, package / path.relative_to(ROOT))
    copy(README, package / 'README.md')
    for stage in ('initial', 'main'):
        copy(CAMPAIGN / f'configs/gcm_m1matched_s22_l000_{stage}.json',
             package / f'templates/{stage}.json')
    full_config = CONSTANT_GC / 'configs/gcft_s22_l000_lr3em7_constant_20261112.json'
    optimizer = read(full_config)['optimizer']
    if not (optimizer['learning_rate'] == 3e-7
            and optimizer['learning_rate_schedule'] == 'constant'
            and optimizer['warmup_steps'] == 0
            and optimizer.get('end_learning_rate') is None):
        raise ValueError('Full-GC handoff requires constant3e-7 from the first update')
    copy(full_config, package / 'templates/full_gc.json')
    copy(NOVEMBER / 'manifests/train_order_seed22.json', package / 'portable/train_order.json')
    copy(NOVEMBER / 'manifests/development_2022_32.json', package / 'portable/development32.json')
    copy(ROOT / 'artifacts/evaluations/v24_res1_2023_stride3_memory_20261004/initializations_2023.json',
         package / 'portable/annual122.json')
    copy(PILOT / 'cohort.json', package / 'portable/pilot4.json')
    data = read(CAMPAIGN / 'configs/gcm_m1matched_s22_l000_initial.json')['data']
    input_records = []
    paths = [('baseline_checkpoint', ''), ('prepared_root', 'metadata.json'),
             ('prepared_root', 'validity.json')]
    paths += [('prepared_root', f'coords/{name}.npy') for name in ('time', 'lat', 'lon', 'level')]
    paths += [('anchor_manifest_root', 'metadata.json')]
    paths += [('anchor_manifest_root', f'anchors/{name}.npy')
              for name in ('anchor_indices', 'anchor_times', 'split_train', 'split_val')]
    paths += [('stats_dir', name) for name in
              ('mean_by_level.nc', 'stddev_by_level.nc', 'diffs_stddev_by_level.nc')]
    for name, relative in paths:
        source = Path(data[name]) / relative if relative else Path(data[name])
        input_records.append({'input': name, 'relative_path': relative,
                              'sha256': file_sha256(source), 'bytes': source.stat().st_size})
        if name in ('anchor_manifest_root', 'stats_dir'):
            target = 'anchors' if name == 'anchor_manifest_root' else 'stats'
            copy(source, package / 'inputs' / target / relative)
    write(package / 'portable/input_reference.json', {
        'source_host': 'Della', 'source_campaign': str(CAMPAIGN),
        'source_digest': original['source_digest'], 'files': input_records,
        'note': 'Hashes cover metadata, coordinates, anchors, statistics and backbone. '
                'Prepared variable arrays are external; their full contents are not rehashed by prepare.'})
    prepared = Path(data['prepared_root'])
    inventory = [{'relative_path': str(p.relative_to(prepared)), 'bytes': p.stat().st_size}
                 for p in sorted(prepared.rglob('*')) if p.is_file()]
    pilot = read(PILOT / 'manifest.json')
    references = {m['id']: m['checkpoint'] for m in pilot['models'] if m['id'] in ('M1_raw', 'M1_swa')}
    assets = {'prepared_training': {'host': 'Della', 'path': str(prepared),
              'bytes': sum(p['bytes'] for p in inventory), 'files': inventory},
              'evaluation_2022': {'host': 'Della', 'path': pilot['development_data']},
              'evaluation_2023': {'host': 'Della', 'path': str(ROOT / 'data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr')},
              'weights': {'public_graphcast': {'path': data['baseline_checkpoint'],
                          'sha256': file_sha256(data['baseline_checkpoint'])}, **references}}
    write(package / 'portable/external_assets.json', assets)
    download_digest = file_sha256(WEIGHTS_DOWNLOAD)
    original_checksums = (WEIGHTS_DOWNLOAD.parent / 'SHA256SUMS').read_text().splitlines()
    if f'{download_digest}  {WEIGHTS_DOWNLOAD.name}' not in original_checksums:
        raise ValueError('Existing handoff weights archive differs from its recorded checksum')
    write(package / 'portable/weights_download.json', {
        'source_host': 'Della', 'path': str(WEIGHTS_DOWNLOAD),
        'filename': WEIGHTS_DOWNLOAD.name, 'sha256': download_digest,
        'bytes': WEIGHTS_DOWNLOAD.stat().st_size,
        'access': 'An authorized SSH account able to read this project directory is required; this is not a public URL.',
        'included_in_this_delivery': weights,
        'contents': ['weights/public_graphcast.npz', 'weights/M1_raw.pkl',
                     'weights/M1_swa.pkl', 'weights/SHA256SUMS']})
    (package / 'portable/weights_download.SHA256SUMS').write_text(
        f'{download_digest}  {WEIGHTS_DOWNLOAD.name}\n')
    (package / 'portable/prepared_files.txt').write_text(''.join(x['relative_path'] + '\n' for x in inventory))
    (package / 'runtime').mkdir()
    (package / 'runtime/requirements.lock.txt').write_text(runtime_lock())
    write(package / 'runtime/environment.json', {'python': sys.version, 'platform': sys.platform,
          'source': 'Installed Della graphcast311 runtime, recorded 2026-11-12',
          'remote_installation_tested': False, 'graphcast': 'Use the included vendored source via PYTHONPATH'})
    write(package / 'portable/source_provenance.json', {
        'upstream_source_digest': original['source_digest'], 'upstream_files': original['files'],
        'adapters': {str(p.relative_to(ROOT)): file_sha256(p) for p in adapters},
        'computation': 'All original submitted source files copied byte-for-byte; portable adapters are additional files.',
        'scope': 'Five M1-recipe GC–Mamba lambda arms and four fresh full-GC horizon arms.',
        'full_gc_schedule': 'Constant3e-7 from the first update; no warmup or decay.',
        'supersedes': 'lianghong_weather_20261112; prior full-GC template used cosine decay.'})
    checksum_file(package)
    code_archive = destination / f'{PACKAGE_NAME}_code.tar.gz'
    with tarfile.open(code_archive, 'w:gz', compresslevel=1) as archive:
        archive.add(package, arcname=package.name)
    archives = [code_archive]
    if weights:
        weights_dir = destination / 'weights'
        weights_dir.mkdir()
        copy(Path(data['baseline_checkpoint']), weights_dir / 'public_graphcast.npz')
        for model, record in references.items():
            copy(Path(record['path']), weights_dir / f'{model}.pkl')
        checksum_file(weights_dir)
        weight_archive = destination / f'{PACKAGE_NAME}_weights.tar.gz'
        with tarfile.open(WEIGHTS_DOWNLOAD, 'r:gz') as archive:
            archived_checksums = archive.extractfile('weights/SHA256SUMS').read().decode()
        if archived_checksums != (weights_dir / 'SHA256SUMS').read_text():
            raise ValueError('Reference archive weights differ from the requested training/evaluation inputs')
        # Reuse the exact archive so bundled and separately downloaded weights
        # have the same checksum, independent of gzip/tar creation timestamps.
        copy(WEIGHTS_DOWNLOAD, weight_archive)
        archives.append(weight_archive)
    (destination / 'SHA256SUMS').write_text(''.join(f'{file_sha256(p)}  {p.name}\n' for p in archives))
    return {'package': str(package), 'archives': [str(p) for p in archives],
            'upstream_source_digest': original['source_digest'],
            'prepared_external_gib': assets['prepared_training']['bytes'] / 1024**3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--without-weights', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.output, weights=not args.without_weights), indent=2))


if __name__ == '__main__':
    main()
