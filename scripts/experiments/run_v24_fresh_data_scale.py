"""Immutable fresh residual data-scale campaigns with reusable seven-year controls."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_cached_40yr import read, write, sha, digest, locked, now
from scripts.experiments.frozen_runtime import freeze_source, verify_source

FORMAT = 'v24_fresh_data_scale_v1'
DRIVER = 'scripts/experiments/run_v24_fresh_data_scale.py'
CONTROL = 'artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007/di64_carry'
CONTROL_CL = 'artifacts/checkpoints/v24_Ilya/res1_cached_w128_closed_loop_k20_20261013'
PRODUCER = 'artifacts/checkpoints/v24_Ilya/res1_cached_40yr_w512_di64_20261014_v5'
NEW_ARMS = ('w512_7y', 'w128_40y', 'w512_40y')
REUSED_CAMPAIGN = 'artifacts/checkpoints/v24_Ilya/res1_fresh_data_scale_20261015_v2'


def pinned(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def job(stage, *, arm=None, memory=16, minutes=30, cpus=2, gpu=None, array=None, stop=None):
    return dict(stage=stage, arm=arm, memory=memory, minutes=minutes, cpus=cpus,
                gpu=gpu, array=array, stop=stop)


def freeze_runtime(workspace, destination, control_source):
    """Preserve the control's scientific training implementation, with bounded I/O."""
    snapshot = freeze_source(workspace, destination)
    historical = read(control_source / 'snapshot.json')
    overrides = {}
    # New drivers, evaluation, and bounded/year-sharded prepared I/O are retained.
    prefixes = ('src/models/mamba/v24_Ilya/training/', 'src/models/mamba/modules/',
                'src/models/mamba/training/', 'third_party/graphcast/')
    exact = {'src/models/mamba/v24_Ilya/model.py'}
    for relative, expected in historical['files'].items():
        if not (relative.startswith(prefixes) or relative in exact):
            continue
        source = control_source / relative
        if sha(source) != expected:
            raise ValueError(f'Historical training source changed: {source}')
        target = destination / relative
        if sha(target) != expected:
            shutil.copy2(source, target)
            overrides[relative] = expected
        snapshot['files'][relative] = expected
    snapshot['source_digest'] = digest(snapshot['files'])
    write(destination / 'snapshot.json', snapshot)
    verify_source(destination)
    return dict(source_digest=snapshot['source_digest'], historical_source=str(control_source),
                historical_source_digest=historical['source_digest'], historical_overrides=overrides,
                retained_current_io='src/models/graphcast/training/core/prepared_array.py',
                purpose='Historical batch1 training arithmetic; bounded reads and year shards are I/O changes')


def prepare(root, workspace, *, large_training_years=40, producer_root=None, reused_campaign=None):
    if large_training_years == 20:
        if producer_root is None:
            raise ValueError('The twenty-year campaign requires its dedicated producer root')
        return prepare_twenty(root, workspace, producer_root, reused_campaign or Path(workspace) / REUSED_CAMPAIGN)
    if large_training_years != 40 or producer_root is not None or reused_campaign is not None:
        raise ValueError('Custom producer/reused controls are supported by the new twenty-year campaign')
    from scripts.experiments.fresh_data_scale_training import open_config, closed_config
    from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    if root.exists():
        raise FileExistsError(f'Use an empty new campaign root: {root}')
    control, cl = workspace / CONTROL, workspace / CONTROL_CL
    template = read(control / 'runs/cached/run_config.json')
    closed_template = read(cl / 'runs/di64/run_config.json')
    if template['optimizer']['warmup_steps'] != 200:
        raise ValueError('Historical control warmup changed')
    init_audit = read(control / 'reports/initialization_identity.json')['initialization_policy']
    if not init_audit['passed'] or init_audit['residual_copied'] != 0:
        raise ValueError('Control is not verified fresh')
    producer = read(workspace / PRODUCER / 'campaign.json')
    paths = dict(large_prepared=producer['paths']['prepared_root'],
                 producer_cache=producer['paths']['cache_root'],
                 large_anchor=str(root / 'data/aligned_anchors'),
                 large_cache=str(workspace / 'data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_1982_2022_aligned7_bptt24_k20'),
                 small_cache=str(workspace / 'data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20'),
                 producer_root=str(workspace / PRODUCER), download_status=producer['paths']['download_status'])
    if Path(paths['large_cache']).exists():
        raise FileExistsError('Dedicated aligned cache already exists')
    for name in ('configs', 'reports', 'logs', 'intents', 'data', 'results'):
        (root / name).mkdir(parents=True, exist_ok=True)
    arms = {}
    for name in NEW_ARMS:
        width = 128 if name.startswith('w128') else 512
        years = 40 if name.endswith('40y') else 7
        ar = root / 'arms' / name
        data = (dict(prepared_root=paths['large_prepared'], anchor_manifest_root=paths['large_anchor'])
                if years == 40 else None)
        op = root / 'configs' / f'{name}_open.json'
        cp = root / 'configs' / f'{name}_closed.json'
        write(op, open_config(template, ar / 'open', 'cached', width, paths=data))
        write(cp, closed_config(closed_template, ar / 'closed/runs', 'closed', width))
        init_source = (str(control / 'shared/checkpoint_step00000000.pkl') if width == 128 else
                       str(root / 'arms/w512_7y/init/checkpoint_step00000000.pkl') if years == 40 else None)
        arms[name] = dict(name=name, width=width, years=years, root=str(ar),
                          open_config=str(op), closed_config=str(cp), init_dir=str(ar / 'init'),
                          init_source=init_source,
                          cache_root=paths['large_cache' if years == 40 else 'small_cache'])
    # Cache generation has the identical baseline and sequence settings.
    cache_config = root / 'configs/cache.json'
    write(cache_config, read(arms['w128_40y']['open_config']))
    arms['w128_7y'] = dict(name='w128_7y', width=128, years=7, reused=True,
        root=str(root / 'arms/w128_7y'), open_config=str(control / 'runs/cached/run_config.json'),
        closed_config=str(cl / 'runs/di64/run_config.json'),
        parent=str(control / 'runs/cached/checkpoints/checkpoint_step00014000.pkl'),
        final=str(cl / 'runs/di64/checkpoints/checkpoint_step00002000.pkl'))
    prior = read(cl / 'manifest.json')
    init2023 = workspace / 'artifacts/evaluations/v24_res1_2023_stride3_memory_20261004/initializations_2023.json'
    eval2023 = workspace / 'data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr'
    inputs = [pinned(p) for p in (control / 'shared/checkpoint_step00000000.pkl',
               control / 'runs/cached/run_config.json', cl / 'runs/di64/run_config.json',
               arms['w128_7y']['parent'], arms['w128_7y']['final'], init2023,
               workspace / PRODUCER / 'campaign.json', template['data']['baseline_checkpoint'])]
    inputs += [pinned(p) for p in Path(template['data']['stats_dir']).glob('*.nc')]
    inputs += [pinned(Path(paths['small_cache']) / name) for name in ('manifest.json', 'READY.json')]
    inputs += [pinned(Path(template['data']['prepared_root']) / 'metadata.json')]
    inputs += [pinned(p) for p in Path(template['data']['anchor_manifest_root']).rglob('*') if p.is_file()]
    configs = [pinned(p) for p in sorted((root / 'configs').glob('*.json'))]
    runtime = freeze_runtime(workspace, root / 'source', control / 'source')
    write(root / 'source_provenance.json', runtime)
    m = dict(format=FORMAT, root=str(root), workspace=str(workspace), source=str(root / 'source'),
             source_digest=runtime['source_digest'], paths=paths, arms=arms, configs=configs, inputs=inputs,
             cache_config=str(cache_config), cache_shards=40, wait_hours=143,
             eval_data2022=prior['eval_data'], eval_data2023=str(eval2023), init2023=str(init2023),
             eval_metadata2022=frozen_zarr_metadata(prior['eval_data']),
             eval_metadata2023=frozen_zarr_metadata(eval2023), chosen_idx=prior['chosen_idx'],
             open_steps=20000, closed_steps=2000, warmup_steps=200, seed=22,
             common_closed_training_years=[2015, 2021], numerical_policy='advisory',
             sampling='historical chronological segments; record selected-parent cursor',
             created_at=now())
    m['workflow_id'] = digest(m)
    write(root / 'manifest.json', m)
    write(root / 'state.json', dict(workflow_id=m['workflow_id'], automation_enabled=False, jobs={}))
    return verify(root)


def prepare_twenty(root, workspace, producer_root, reused_campaign):
    """Two new twenty-year lineages, preserving both completed seven-year arms."""
    from scripts.experiments.fresh_data_scale_training import open_config, closed_config
    from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata
    root, workspace, producer_root, reused_campaign = map(
        lambda p: Path(p).resolve(), (root, workspace, producer_root, reused_campaign))
    if root.exists():
        raise FileExistsError(f'Use an empty new campaign root: {root}')
    old = verify(reused_campaign)
    producer = read(producer_root / 'campaign.json')
    if (producer.get('years') != list(range(2002, 2023))
            or producer.get('training_years') != [2002, 2021]
            or producer.get('validation_year') != 2022
            or producer.get('stop_after') != 'cache_verify'):
        raise ValueError('Producer must prepare 2002–2021 training plus 2022 validation and stop after its cache')
    control, closed_control = workspace / CONTROL, workspace / CONTROL_CL
    template = read(control / 'runs/cached/run_config.json')
    closed_template = read(closed_control / 'runs/di64/run_config.json')
    paths = dict(large_prepared=producer['paths']['prepared_root'],
                 producer_cache=producer['paths']['cache_root'],
                 large_anchor=str(root / 'data/aligned_anchors'),
                 large_cache=str(root / 'data/aligned_cache'),
                 small_cache=old['paths']['small_cache'], producer_root=str(producer_root),
                 download_status=producer['paths']['download_status'])
    for name in ('configs', 'reports', 'logs', 'intents', 'data', 'results'):
        (root / name).mkdir(parents=True, exist_ok=True)
    arms, inputs = {}, [pinned(reused_campaign / 'manifest.json'), pinned(producer_root / 'campaign.json')]
    # Fixed checkpoints and completed development evaluations are reused as evidence.
    for name in ('w128_7y', 'w512_7y'):
        prior = old['arms'][name]
        if name == 'w128_7y':
            parent, final = prior['parent'], prior['final']
            evidence = control / 'reports/integrity.json'
            benchmark = control / 'reports/benchmark.json'
            init = control / 'shared/checkpoint_step00000000.pkl'
            evaluations = {f'2022_{label}': str(closed_control / f'evaluation/di64_step{step:05d}.json')
                           for label, step in (('parent', 0), ('final', 2000))}
        else:
            selection = require_result(old, f'select_{name}')
            trained = require_result(old, f'closed_{name}')
            if selection.get('step') != 8000 or trained.get('completed_step') != 2000:
                raise ValueError('Completed width512 control endpoint/parent changed')
            parent, final = selection['checkpoint'], trained['checkpoint']
            if sha(parent) != selection['checkpoint_sha256'] or sha(final) != trained['checkpoint_sha256']:
                raise ValueError('Completed seven-year checkpoint changed')
            evidence = Path(prior['root']) / 'qualification.json'
            benchmark = None
            init = Path(prior['init_dir']) / 'checkpoint_step00000000.pkl'
            evaluations = {}
            for index, label in enumerate(('parent', 'final')):
                output = require_result(old, f'eval2022_{name}_{index:03d}')['output']
                if pinned(output['path']) != output:
                    raise ValueError('Completed seven-year evaluation changed')
                evaluations[f'2022_{label}'] = output['path']
        inputs += [pinned(p) for p in (prior['open_config'], prior['closed_config'], parent, final, init, evidence,
                                      *evaluations.values())]
        if benchmark is not None:
            inputs.append(pinned(benchmark))
        arms[name] = dict(name=name, width=prior['width'], years=7, reused=True,
                         root=str(root / 'arms' / name), open_config=prior['open_config'],
                         closed_config=prior['closed_config'], parent=parent, final=final,
                         initial_checkpoint=str(init), scientific_evidence=str(evidence),
                         benchmark_evidence=str(benchmark) if benchmark is not None else None,
                         reused_evaluations=evaluations, reused_from=str(reused_campaign))
        merged = Path(prior['root']) / 'evaluation/merged2023.json'
        if merged.exists():
            # Saved complete results can be imported after strict identity checking.
            arms[name]['reused_merged2023'] = str(merged)
            inputs.append(pinned(merged))
    for width in (128, 512):
        name = f'w{width}_20y'
        prior = arms[f'w{width}_7y']
        ar = root / 'arms' / name
        op, cp = root / 'configs' / f'{name}_open.json', root / 'configs' / f'{name}_closed.json'
        write(op, open_config(template, ar / 'open', 'cached', width,
                 paths=dict(prepared_root=paths['large_prepared'], anchor_manifest_root=paths['large_anchor'])))
        write(cp, closed_config(closed_template, ar / 'closed/runs', 'closed', width))
        arms[name] = dict(name=name, width=width, years=20, root=str(ar), open_config=str(op),
                         closed_config=str(cp), init_dir=str(ar / 'init'),
                         init_source=prior['initial_checkpoint'], cache_root=paths['large_cache'],
                         scientific_evidence=prior['scientific_evidence'],
                         benchmark_evidence=prior['benchmark_evidence'],
                         evidence_config=prior['open_config'])
    cache_config = root / 'configs/cache.json'
    write(cache_config, read(arms['w128_20y']['open_config']))
    inputs += [pinned(p) for p in (old['init2023'], template['data']['baseline_checkpoint'])]
    inputs += [pinned(p) for p in Path(template['data']['stats_dir']).glob('*.nc')]
    inputs += [pinned(Path(paths['small_cache']) / name) for name in ('manifest.json', 'READY.json')]
    inputs += [pinned(Path(template['data']['prepared_root']) / 'metadata.json')]
    inputs += [pinned(p) for p in Path(template['data']['anchor_manifest_root']).rglob('*') if p.is_file()]
    runtime = freeze_runtime(workspace, root / 'source', control / 'source')
    write(root / 'source_provenance.json', runtime)
    m = dict(format=FORMAT, root=str(root), workspace=str(workspace), source=str(root / 'source'),
             source_digest=runtime['source_digest'], paths=paths, arms=arms,
             configs=[pinned(p) for p in sorted((root / 'configs').glob('*.json'))], inputs=inputs,
             cache_config=str(cache_config), cache_shards=20, wait_hours=143,
             large_training_years=20, prepared_years=list(range(2002, 2023)),
             training_years=[2002, 2021], validation_year=2022, new_arms=['w128_20y', 'w512_20y'],
             reuse_qualification=True, producer_scheduler_bridge=False,
             eval_data2022=old['eval_data2022'], eval_data2023=old['eval_data2023'], init2023=old['init2023'],
             eval_metadata2022=frozen_zarr_metadata(old['eval_data2022']),
             eval_metadata2023=frozen_zarr_metadata(old['eval_data2023']), chosen_idx=old['chosen_idx'],
             open_steps=20000, closed_steps=2000, warmup_steps=200, seed=22,
             common_closed_training_years=[2015, 2021], numerical_policy='advisory',
             sampling='historical chronological segments; record selected-parent cursor', created_at=now())
    m['workflow_id'] = digest(m)
    write(root / 'manifest.json', m)
    write(root / 'state.json', dict(workflow_id=m['workflow_id'], automation_enabled=False, jobs={}))
    return verify(root)


def verify(root, *, runtime=False):
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    m = read(Path(root) / 'manifest.json')
    if m.get('format') != FORMAT or digest({k: v for k, v in m.items() if k != 'workflow_id'}) != m['workflow_id']:
        raise ValueError('Campaign identity changed')
    if Path(m['root']).resolve() != Path(root).resolve():
        raise ValueError('Campaign root changed')
    if verify_source(m['source'])['source_digest'] != m['source_digest']:
        raise ValueError('Frozen source changed')
    if runtime and ROOT.resolve() != Path(m['source']).resolve():
        raise ValueError('Execute through the frozen source')
    for item in m['configs'] + m['inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Pinned input changed: {item["path"]}')
    verify_frozen_zarr_metadata(m['eval_data2022'], m['eval_metadata2022'])
    verify_frozen_zarr_metadata(m['eval_data2023'], m['eval_metadata2023'])
    return m


def sbatch_args(m, key, spec, dependencies=()):
    if any(not str(x).isdigit() for x in dependencies):
        raise ValueError('Invalid dependency')
    args = ['sbatch', '--parsable', '--account=dabanin', f'--job-name=fds-{key}',
            f'--cpus-per-task={spec["cpus"]}', f'--mem={spec["memory"]}G',
            f'--time={spec["minutes"]}', f'--chdir={m["workspace"]}',
            f'--comment=fds:{digest([m["workflow_id"], key])[:24]}',
            f'--output={m["root"]}/logs/{key}_%A_%a.out',
            f'--error={m["root"]}/logs/{key}_%A_%a.err']
    if spec['gpu']:
        # Della assigns the GPU partition; an explicit --partition=gpu is rejected.
        args += ['--gres=gpu:1', f'--constraint={spec["gpu"]}']
        qos = 'gpu-short' if spec['minutes'] <= 1440 else 'gpu-medium' if spec['minutes'] <= 4320 else 'gpu-long'
        if spec['minutes'] > 8640:
            raise ValueError('GPU stage exceeds the six-day queue limit')
        args += [f'--qos={qos}']
    else:
        args += ['--partition=cpu']
        if spec['minutes'] > 4320:
            args += ['--qos=vlong']
    if spec['array'] is not None:
        args += [f'--array=0-{spec["array"] - 1}']
    if dependencies:
        args += ['--dependency=afterok:' + ':'.join(map(str, dependencies))]
    return args + [str(Path(m['source']) / 'scripts/experiments/v24_fresh_data_scale.slurm'),
                   m['root'], key, m['workspace']]


def submit_job(m, key, spec, dependencies=()):
    root = Path(m['root'])
    identity = dict(spec=spec, dependencies=list(dependencies))
    with locked(root):
        state = read(root / 'state.json')
        if not state['automation_enabled']:
            raise RuntimeError('Submission not enabled')
        if key in state['jobs']:
            old = state['jobs'][key]
            if any(old[k] != v for k, v in identity.items()):
                raise ValueError(f'Conflicting existing job: {key}')
            return old['job_id']
        intent = root / 'intents' / f'{key}.json'
        if intent.exists():
            raise RuntimeError(f'Unresolved sbatch intent; reconcile before retry: {intent}')
        command = sbatch_args(m, key, spec, dependencies)
        write(intent, dict(command=command, identity=identity, created_at=now()))
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            write(intent.with_suffix('.rejected.json'), dict(command=command, error=result.stderr, created_at=now()))
            intent.unlink()
            raise RuntimeError(result.stderr)
        jid = result.stdout.strip().split(';')[0]
        if not jid.isdigit():
            raise RuntimeError(f'Unrecognized sbatch output: {result.stdout}')
        state['jobs'][key] = dict(**identity, job_id=jid, status='submitted', submitted_at=now())
        write(root / 'state.json', state)
        intent.unlink()
        return jid


def initial_jobs(m):
    # Distinct aligned-cache producer; shared prepared data has one existing owner.
    if m.get('reuse_qualification'):
        return [
            ('prepared_wait', job('prepared_wait', memory=2, cpus=1, minutes=m['wait_hours'] * 60 + 5), []),
            ('aligned_prepare', job('aligned_prepare', memory=64, cpus=8, minutes=240), ['prepared_wait']),
            ('cache_generate', job('cache_generate', memory=8, cpus=2, minutes=240,
                                   array=m['cache_shards']), ['aligned_prepare']),
            ('cache_verify', job('cache_verify', memory=16, cpus=4, minutes=240), ['cache_generate']),
            *[(f'qualify_{arm}', job('startup', arm=arm, memory=64, cpus=4, minutes=60), ['cache_verify'])
              for arm in m['new_arms']],
        ]
    return [
        ('producer_scheduler_bridge', job('producer_scheduler_bridge', memory=2, cpus=1, minutes=m['wait_hours'] * 60 + 5), []),
        ('prepared_wait', job('prepared_wait', memory=2, cpus=1, minutes=m['wait_hours'] * 60 + 5), []),
        ('aligned_prepare', job('aligned_prepare', memory=64, cpus=8, minutes=240), ['prepared_wait']),
        ('cache_generate', job('cache_generate', memory=8, cpus=2, minutes=240, array=40), ['aligned_prepare']),
        ('cache_verify', job('cache_verify', memory=16, cpus=4, minutes=240), ['cache_generate']),
        ('qualify_w512_7y', job('qualify', arm='w512_7y', memory=128, cpus=4, minutes=240, gpu='gpu80&nomig'), []),
        ('qualify_w128_40y', job('qualify', arm='w128_40y', memory=128, cpus=4, minutes=240, gpu='gpu80&nomig'), ['cache_verify']),
        ('qualify_w512_40y', job('qualify', arm='w512_40y', memory=128, cpus=4, minutes=240, gpu='gpu80&nomig'), ['cache_verify', 'qualify_w512_7y']),
    ]


def submit(m, *, dry=False, producer_dependency=None):
    jobs = initial_jobs(m)
    if producer_dependency is not None:
        if not str(producer_dependency).isdigit() or not m.get('reuse_qualification'):
            raise ValueError('A producer dependency is a numeric job ID for the new twenty-year campaign')
        jobs = [(key, spec, [str(producer_dependency)] if key == 'prepared_wait' else deps)
                for key, spec, deps in jobs]
    if dry:
        return {'submitted': False, 'initial_jobs': jobs,
                'automatic': 'startup/reused qualification → 5×4000 open updates → select → closed2k → matched32; all finals frozen → 2023 eight shards/model or valid saved results → comparison'}
    with locked(m['root']):
        p = Path(m['root']) / 'state.json'
        state = read(p); state['automation_enabled'] = True; write(p, state)
    ids = {}
    for key, spec, deps in jobs:
        ids[key] = submit_job(m, key, spec, [ids.get(d, d) for d in deps])
    return ids


def require_result(m, key):
    p = read(Path(m['root']) / 'reports' / f'{key}.json')
    if p['workflow_id'] != m['workflow_id'] or p['sha256'] != digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError(f'Invalid stage proof: {key}')
    if p['result'].get('passed') is not True:
        raise ValueError(f'Failed prerequisite: {key}')
    return p['result']


def run_helper(name, args):
    command = [sys.executable, '-u', str(ROOT / f'scripts/experiments/fresh_data_scale_{name}.py'), *map(str, args)]
    subprocess.run(command, check=True)


def training_budget(result, width):
    # Conservative historical fallback; qualified bounded reads may justify less.
    result = result.get('benchmark', result)
    peak = float(result.get('peak_host_gib', 400 if width == 512 else 313))
    rate = float(result.get('p90_end_to_end_seconds', 12 if width == 512 else 8))
    validation = float(result.get('validation_seconds', 900))
    if not all(math.isfinite(x) and x > 0 for x in (peak, rate, validation)):
        raise ValueError('Invalid qualification resource measurements')
    return dict(memory=max(96, math.ceil(peak * 1.5 / 16) * 16),
                minutes=max(240, math.ceil((4000 * rate + 2 * validation + 900) * 1.4 / 1800) * 30),
                cpus=4, gpu='gpu80&nomig')


def advance(m, key, spec, result):
    state = read(Path(m['root']) / 'state.json')
    if not state['automation_enabled']:
        return
    jid = state['jobs'][key]['job_id']
    arm = spec['arm']
    if spec['stage'] in ('qualify', 'startup'):
        budget = training_budget(result, m['arms'][arm]['width'])
        previous = jid
        for stop in range(4000, 20001, 4000):
            previous = submit_job(m, f'open_{arm}_{stop:05d}', job('open', arm=arm, stop=stop, **budget), [previous])
        selection = submit_job(m, f'select_{arm}', job('select', arm=arm, memory=16), [previous])
        width = m['arms'][arm]['width']
        closed = submit_job(m, f'closed_{arm}', job('closed', arm=arm, memory=416 if width == 512 else 384,
                            minutes=600 if width == 512 else 480, cpus=8, gpu='gpu80&nomig'), [selection])
        submit_job(m, f'eval2022_{arm}', job('eval2022', arm=arm, memory=20, cpus=4,
                                          minutes=120, gpu='gpu80&nomig', array=2), [closed])
        maybe_freeze_finals(m)
    elif spec['stage'] == 'freeze_finals':
        merges = []
        for name in m['arms']:
            if m['arms'][name].get('reused_merged2023'):
                merges.append(submit_job(m, f'merge_{name}', job('reuse2023', arm=name, memory=8), [jid]))
                continue
            ev = submit_job(m, f'eval2023_{name}', job('eval2023', arm=name, memory=20, cpus=4,
                           minutes=90, gpu='gpu80&nomig', array=8), [jid])
            merges.append(submit_job(m, f'merge_{name}', job('merge', arm=name, memory=8), [ev]))
        submit_job(m, 'comparison', job('comparison', memory=16, minutes=60), merges)


def maybe_freeze_finals(m):
    jobs = read(Path(m['root']) / 'state.json')['jobs']
    required = [f'eval2022_{arm}' for arm in m.get('new_arms', NEW_ARMS)]
    if all(k in jobs for k in required):
        submit_job(m, 'freeze_finals', job('freeze_finals', memory=16), [jobs[k]['job_id'] for k in required])


def execute_stage(m, key, spec, index):
    root, paths, stage = Path(m['root']), m['paths'], spec['stage']
    arm = m['arms'].get(spec['arm'])
    if stage == 'producer_scheduler_bridge':
        from scripts.experiments.fresh_data_scale_producer_bridge import run
        return run(Path(paths['producer_root']), timeout_hours=m['wait_hours'])
    if stage == 'prepared_wait':
        deadline = time.monotonic() + m['wait_hours'] * 3600
        ready = Path(paths['large_prepared']) / 'READY.json'
        cache_ready = Path(paths['producer_cache']) / 'READY.json'
        while not (ready.exists() and cache_ready.exists()):
            own = read(root / 'state.json')['jobs'].get('producer_scheduler_bridge', {})
            if own.get('status') == 'failed':
                raise RuntimeError('Shared producer scheduler bridge failed; inspect its report')
            status = read(paths['download_status'])
            if status.get('status') == 'failed':
                raise RuntimeError(f'Download failed: {status.get("error")}')
            producer_state = read(Path(paths['producer_root']) / 'state.json')
            if any(x.get('status') == 'failed' and x['spec']['stage'] in
                   ('download_gate', 'prepare_pilot', 'prepare_year', 'prepare_verify',
                    'cache_pilot', 'cache_release', 'cache_generate', 'cache_verify')
                   for x in producer_state['jobs'].values()):
                raise RuntimeError('Shared prepared-data producer failed; downstream work stopped')
            if time.monotonic() > deadline:
                raise TimeoutError('Prepared-data wait exhausted; no downstream work released')
            print(json.dumps({'waiting_for': str(cache_ready), 'download': status.get('current_year'), 'at': now()}), flush=True)
            time.sleep(60)
        record = read(ready)
        if record.get('status') != 'complete' or record.get('years') != m.get('prepared_years', list(range(1982, 2023))):
            raise ValueError('Incomplete full prepared-data marker')
        if record.get('metadata_sha256') != sha(ready.parent / 'metadata.json') or (ready.parent / '.incomplete').exists():
            raise ValueError('Invalid prepared-data identity')
        cache_record = read(cache_ready)
        cache_manifest = read(cache_ready.parent / 'manifest.json')
        if cache_record.get('partial') or cache_record.get('manifest_sha256') != cache_manifest.get('manifest_sha256'):
            raise ValueError('Invalid existing full-cache identity')
        return dict(passed=True, ready=pinned(ready), producer_cache_ready=pinned(cache_ready),
                    producer_cache_manifest=pinned(cache_ready.parent / 'manifest.json'))
    if stage in ('aligned_prepare', 'cache_generate', 'cache_verify'):
        from scripts.experiments import fresh_data_scale_cache as cache
        if stage == 'aligned_prepare':
            require_result(m, 'prepared_wait')
            return cache.prepare_aligned_cache(Path(m['cache_config']), Path(paths['small_cache']), Path(paths['large_cache']),
                                               num_shards=m.get('cache_shards', 40), historical_cache_root=Path(paths['producer_cache']))
        require_result(m, 'aligned_prepare')
        if stage == 'cache_generate':
            return cache.generate_aligned_shard(Path(m['cache_config']), Path(paths['small_cache']), Path(paths['large_cache']), index,
                                               historical_cache_root=Path(paths['producer_cache']))
        return cache.verify_aligned_cache(Path(paths['large_cache']))
    if stage in ('qualify', 'startup'):
        if arm['years'] > 7:
            require_result(m, 'cache_verify')
        args = ['init', '--config', arm['open_config'], '--output-dir', arm['init_dir']]
        if arm['init_source']:
            args += ['--source-init', arm['init_source']]
        run_helper('training', args)
        report = Path(arm['root']) / 'qualification.json'
        args = [stage, '--config', arm['open_config'], '--init', str(Path(arm['init_dir']) / 'checkpoint_step00000000.pkl'),
                '--cache-root', arm['cache_root'], '--report', report]
        if stage == 'startup':
            args += ['--evidence', arm['scientific_evidence'], '--previous-config', arm['evidence_config'],
                     '--previous-init', arm['init_source']]
            if arm.get('benchmark_evidence'):
                args += ['--benchmark', arm['benchmark_evidence']]
        run_helper('training', args)
        return read(report)
    if stage == 'open':
        require_result(m, f'qualify_{arm["name"]}')
        report = Path(arm['root']) / f'open_{spec["stop"]:05d}.json'
        run_helper('training', ['train-open', '--config', arm['open_config'], '--init', str(Path(arm['init_dir']) / 'checkpoint_step00000000.pkl'),
                   '--cache-root', arm['cache_root'], '--report', report, '--stop-step', spec['stop'],
                   '--integrity-report', str(Path(arm['root']) / 'qualification.json')])
        return read(report)
    if stage == 'select':
        require_result(m, f'open_{arm["name"]}_20000')
        report = Path(arm['root']) / 'selection.json'
        run_helper('training', ['select', '--config', arm['open_config'], '--report', report])
        return read(report)
    if stage == 'closed':
        selected = require_result(m, f'select_{arm["name"]}')
        report = Path(arm['root']) / 'closed.json'
        run_helper('training', ['train-closed', '--config', arm['closed_config'], '--parent', selected['checkpoint'], '--report', report])
        return read(report)
    if stage == 'eval2022':
        selected = require_result(m, f'select_{arm["name"]}')
        trained = require_result(m, f'closed_{arm["name"]}')
        checkpoint = selected['checkpoint'] if index == 0 else trained['checkpoint']
        output = Path(arm['root']) / 'evaluation' / f'2022_{"parent" if index == 0 else "final"}.json'
        run_helper('evaluation', ['--config', arm['closed_config'], '--checkpoint', checkpoint,
                   '--out', output, '--year', 2022, '--data-path', m['eval_data2022']])
        return dict(passed=True, output=pinned(output))
    if stage == 'freeze_finals':
        finals = {}
        for name, a in m['arms'].items():
            if a.get('reused'):
                checkpoint = a['final']
            else:
                checkpoint = require_result(m, f'closed_{name}')['checkpoint']
                for i in range(2):
                    require_result(m, f'eval2022_{name}_{i:03d}')
            finals[name] = dict(checkpoint=pinned(checkpoint), config=pinned(a['closed_config']))
        result = dict(passed=True, finals=finals, frozen_at=now())
        write(root / 'final_models.json', result)
        return result
    if stage == 'eval2023':
        finals = require_result(m, 'freeze_finals')['finals']
        checkpoint = finals[arm['name']]['checkpoint']
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen final checkpoint changed')
        output = Path(arm['root']) / 'evaluation' / f'2023_shard_{index:02d}.json'
        run_helper('evaluation', ['--config', arm['closed_config'], '--checkpoint', checkpoint['path'],
                   '--out', output, '--year', 2023, '--data-path', m['eval_data2023'],
                   '--initialization-manifest', m['init2023'], '--index', index, '--shards', 8])
        return dict(passed=True, output=pinned(output))
    if stage == 'merge':
        from scripts.experiments.fresh_data_scale_evaluation import merge
        files = [Path(arm['root']) / 'evaluation' / f'2023_shard_{i:02d}.json' for i in range(8)]
        for i, path in enumerate(files):
            proof = require_result(m, f'eval2023_{arm["name"]}_{i:03d}')['output']
            if str(path) != proof['path'] or sha(path) != proof['sha256']:
                raise ValueError('Evaluation shard changed after completion')
        checkpoint = require_result(m, 'freeze_finals')['finals'][arm['name']]['checkpoint']
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen final checkpoint changed')
        result = merge(files, Path(m['init2023']), Path(arm['root']) / 'evaluation/merged2023.json',
                       expected_checkpoint=checkpoint['path'])
        return dict(passed=True, **result)
    if stage == 'reuse2023':
        from scripts.experiments.fresh_data_scale_evaluation import manifest_times, loss_arrays
        result = read(arm['reused_merged2023'])
        checkpoint = require_result(m, 'freeze_finals')['finals'][arm['name']]['checkpoint']
        if (result.get('status') != 'complete' or result['identity']['checkpoint'] != checkpoint
                or pinned(checkpoint['path']) != checkpoint
                or result['identity']['initializations'] != pinned(m['init2023'])
                or result['initialization_times'] != manifest_times(m['init2023'])):
            raise ValueError('Saved seven-year 2023 evaluation is incompatible')
        loss_arrays(result['baseline_loss_matrix'], result['full_loss_matrix'], (122, 40))
        write(Path(arm['root']) / 'evaluation/merged2023.json', result)
        return dict(passed=True, **result)
    if stage == 'comparison':
        from scripts.experiments.fresh_data_scale_evaluation import comparison_report
        reports = {name: str(Path(a['root']) / 'evaluation/merged2023.json') for name, a in m['arms'].items()}
        large = m.get('large_training_years', 40)
        result = comparison_report(reports, root / 'results',
                                   arm_keys={(w, y): f'w{w}_{y}y' for w in (128, 512) for y in (7, large)})
        return dict(passed=True, **result)
    raise ValueError(stage)


def execute(m, key, index=None):
    root = Path(m['root'])
    entry = read(root / 'state.json')['jobs'][key]
    spec = entry['spec']
    actual_job = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID'))
    if actual_job and actual_job != entry['job_id']:
        raise ValueError('Stale job attempt')
    if spec['array'] is not None and (index is None or not 0 <= index < spec['array']):
        raise ValueError('Invalid array index')
    report_key = f'{key}_{index:03d}' if spec['array'] is not None else key
    try:
        result = execute_stage(m, key, spec, index)
        if result.get('passed') is not True:
            raise ValueError('Stage did not pass')
    except BaseException as exc:
        with locked(root):
            state = read(root / 'state.json')
            state['jobs'][key].update(status='failed', error=str(exc), ended_at=now())
            write(root / 'state.json', state)
        raise
    payload = dict(workflow_id=m['workflow_id'], key=report_key, result=result, completed_at=now())
    payload['sha256'] = digest(payload)
    write(root / 'reports' / f'{report_key}.json', payload)
    if spec['array'] is None:
        with locked(root):
            state = read(root / 'state.json')
            state['jobs'][key].update(status='complete', ended_at=now())
            write(root / 'state.json', state)
        try:
            advance(m, key, spec, result)
        except BaseException as exc:
            with locked(root):
                state = read(root / 'state.json')
                state['jobs'][key].update(status='release_failed', error=str(exc))
                write(root / 'state.json', state)
            print(f'Automatic submission failed; resume advancement for {key}: {exc}', file=sys.stderr)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'verify', 'submit', 'execute', 'advance', 'status'])
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    p.add_argument('--stage')
    p.add_argument('--index', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--large-training-years', type=int, choices=(20, 40), default=40)
    p.add_argument('--producer-root', type=Path)
    p.add_argument('--reuse-campaign', type=Path)
    p.add_argument('--producer-dependency')
    a = p.parse_args()
    m = prepare(a.campaign_root, a.workspace, large_training_years=a.large_training_years,
                producer_root=a.producer_root, reused_campaign=a.reuse_campaign) if a.action == 'prepare' else verify(a.campaign_root, runtime=a.action == 'execute')
    if a.action in ('prepare', 'verify'):
        result = {'passed': True, 'root': m['root'], 'workflow_id': m['workflow_id']}
    elif a.action == 'submit':
        result = submit(m, dry=a.dry_run, producer_dependency=a.producer_dependency)
    elif a.action == 'execute':
        result = execute(m, a.stage, a.index)
    elif a.action == 'advance':
        entry = read(Path(m['root']) / 'state.json')['jobs'][a.stage]
        advance(m, a.stage, entry['spec'], require_result(m, a.stage))
        result = {'advanced': a.stage}
    else:
        result = read(Path(m['root']) / 'state.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
