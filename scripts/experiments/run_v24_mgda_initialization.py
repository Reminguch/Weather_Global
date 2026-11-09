"""Frozen full-BPTT continuation/fresh MGDA comparisons, with matched MSE controls.

Checkpointed stages bound execution without changing the optimization schedule.
Finite loss increases and repeatability drift remain advisory throughout.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_v24_mse_amse_excess_pilot as pilot
from scripts.experiments.frozen_runtime import (
    FORMAT as SOURCE_FORMAT, _digest, _source_files, file_sha256, verify_source,
)
from scripts.experiments.run_v24_mgda import (
    call, check, gpu_required, pinned, read, seal, submit_once, unseal, write,
)

FORMAT = 'v24_mgda_initialization_v1'
ARMS = ('mgda', 'mse_control')
DRIVER = 'scripts/experiments/run_v24_mgda_initialization.py'
WRAPPER = 'scripts/experiments/v24_mgda_initialization.slurm'
CONFIG_MODULE = 'src/models/mamba/v24_Ilya/training/config.py'
PARENT = 'artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028'
TRAINING_PARTS = 2


def endpoints(kind):
    if kind not in ('continuation', 'fresh'):
        raise ValueError('Unknown experiment kind')
    start, end = (200, 2000) if kind == 'continuation' else (0, 2500)
    return [(start + end) // TRAINING_PARTS, end]


def make_config(template, experiment, arm, kind):
    if kind not in ('continuation', 'fresh') or arm not in ARMS:
        raise ValueError('Unknown experiment kind or arm')
    config = copy.deepcopy(template)
    config['output'] = {'output_root': str(experiment / 'runs'), 'run_name': arm}
    config['optimizer'].update(max_steps=2000 if kind == 'continuation' else 2500,
                               checkpoint_every=100, mgda_direction=arm)
    if kind == 'fresh':
        config['optimizer'].update(learning_rate=1e-5, warmup_start_learning_rate=1e-6,
            end_learning_rate=3e-6, warmup_steps=100, decay_end_step=2000)
        config['architecture'].update(residual_initialization='baseline_overlay',
            temporal_init_scheme='mamba1', temporal_zero_init_out=True)
    return config


def freeze(parent_source, destination, workspace):
    original = verify_source(parent_source)
    if destination.exists():
        raise FileExistsError(destination)
    for relative in original['files']:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_source / relative, target)
    for relative in (DRIVER, WRAPPER, CONFIG_MODULE):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / relative, target)
    files = {str(p.relative_to(destination)): file_sha256(p) for p in _source_files(destination)}
    changed = sorted(name for name in files if files[name] != original['files'].get(name))
    if set(changed) - {DRIVER, WRAPPER, CONFIG_MODULE}:
        raise ValueError('Unexpected runtime changes')
    write(destination / 'snapshot.json', {'format': SOURCE_FORMAT, 'workspace': str(workspace),
        'files': files, 'source_digest': _digest(files)})
    result = verify_source(destination)
    return result, {'parent_source_digest': original['source_digest'], 'changed_files': changed,
        'runtime': 'Preserve the proven full-feedback runtime, including its historical prepared loader.',
        'resume_change': 'Permit output relocation only; original checkpoint tensors and training semantics unchanged.'}


def measured_budget(parent):
    import numpy as np
    histories = [read_lines(parent / 'runs' / arm / 'train_metrics.jsonl') for arm in ARMS]
    p90 = max(float(np.percentile([r.get('end_to_end_seconds', r['step_seconds'])
                                   for r in rows[5:]], 90)) for rows in histories)
    peaks = [unseal(parent / 'checks' / f'{arm}_train.json')['peak_rss_gib'] for arm in ARMS]
    observation = unseal(parent / 'checks/observation.json')
    startup = max(900., float(observation['startup_compile_seconds']) * 1.4)
    if not all(math.isfinite(x) and x > 0 for x in (p90, *peaks, startup)):
        raise ValueError('Invalid prior resource measurements')
    # Longer execution parts retain extra headroom over the measured host peak.
    memory = math.ceil(1.6 * max(234.71, *peaks) / 16) * 16
    return {'memory': memory, 'steady_p90_seconds': p90, 'startup_allowance_seconds': startup,
        'evidence_jobs': ['16402493_0', '16402493_1'],
        'historical_peak_rss_gib': max(234.71, *peaks),
        'memory_formula': 'ceil(1.6*max(234.71,measured_peak_GiB)/16)*16',
        'training_parts': TRAINING_PARTS, 'automatic_horizon_or_batch_reduction': False}


def stage_budget(manifest, phase):
    stop = manifest['stage_endpoints'][phase]
    start = (manifest['stage_endpoints'][phase - 1] if phase else manifest['initial_step'])
    evidence = manifest['resources']
    minutes = math.ceil(((stop - start) * evidence['steady_p90_seconds'] * 1.4
                         + evidence['startup_allowance_seconds']) / 1800) * 30
    return {'memory': evidence['memory'], 'minutes': max(60, minutes)}


def read_lines(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(experiment, parent, workspace, kind):
    from src.models.mamba.v24_Ilya.training.config import load_training_config, validate_resume_config
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest
    experiment, parent, workspace = map(lambda p: Path(p).resolve(), (experiment, parent, workspace))
    if experiment.exists():
        raise FileExistsError('Use a fresh campaign directory')
    prior = pilot.verify(parent)
    completed = unseal(parent / 'complete.json')
    check(completed['report'])
    if completed['status'] != 'complete' or prior['policy']['feedback_mode'] != 'closed_loop':
        raise ValueError('A completed full-feedback parent is required')
    if prior['source_digest'] != 'b71969ce37d1cf47b231d93719e24a21b01d01503717fe785a19baf6499cfd13':
        raise ValueError('Unexpected parent runtime')
    experiment.mkdir(parents=True)
    for name in ('logs', 'checks', 'submissions'):
        (experiment / name).mkdir()
    inputs = list(prior['inputs']) + [pinned(parent / 'manifest.json'), pinned(parent / 'complete.json'),
                                     completed['report'], pinned(parent / 'checks/observation.json')]
    initial = {}
    for arm in ARMS:
        proof = pilot.train_proof(parent, prior, arm)
        inputs.append(pinned(parent / 'checks' / f'{arm}_train.json'))
        config = make_config(read(parent / 'configs' / f'{arm}.json'), experiment, arm, kind)
        path = experiment / 'configs' / f'{arm}.json'
        write(path, config)
        cfg = load_training_config(path)
        inputs.append(pinned(path))
        if kind == 'continuation':
            checkpoint = load_v24_Ilya_training_checkpoint(Path(proof['checkpoint']['path']))
            validate_resume_config(cfg, checkpoint.resolved_training_config, completed_step=200)
            target = experiment / 'initial' / arm
            target.mkdir(parents=True)
            destination = experiment / 'runs' / arm / 'checkpoints/checkpoint_step00000200.pkl'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(proof['checkpoint']['path'], destination)
            if file_sha256(destination) != proof['checkpoint']['sha256']:
                raise ValueError('Continuation checkpoint copy differs')
            files = {}
            for name in ('train_metrics.jsonl', 'backtracking_trials.jsonl', 'checkpoint_metrics.jsonl'):
                source = parent / 'runs' / arm / name
                shutil.copy2(source, target / name)
                run_copy = experiment / 'runs' / arm / name
                run_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, run_copy)
                files[name] = pinned(target / name)
            initial[arm] = {'checkpoint': pinned(destination), 'history': files,
                            'source_checkpoint': proof['checkpoint']}
            inputs.extend([initial[arm]['checkpoint'], *files.values()])
            del checkpoint
    evaluation = copy.deepcopy(read(parent / 'evaluation/manifest.json'))
    evaluation.pop('manifest_hash', None)
    evaluation['output_dir'] = str(experiment / 'evaluation')
    reference = copy.deepcopy(evaluation['task_specs'][0])
    reference['label'] = 'Best trained MSE reference (not fresh initialization)'
    reference['reuse'] = {stage: str(parent / 'evaluation/results/parent' / stage)
                          for stage in ('exact', 'spectral')}
    for directory in reference['reuse'].values():
        inputs.extend(pinned(Path(directory) / name)
                      for name in ('specification.json', 'evaluation.json'))
    tasks = [reference]
    final = endpoints(kind)[-1]
    for arm in ARMS:
        run = experiment / 'runs' / arm
        path = experiment / 'configs' / f'{arm}.json'
        tasks.append({'id': f'{arm}_step{final:05d}', 'label': f'{kind} {arm} update{final}',
            'role': 'candidate', 'family': arm, 'step': final, 'seed': 22,
            'checkpoint': str(run / 'checkpoints' / f'checkpoint_step{final:08d}.pkl'),
            'source_run_dir': str(run), 'run_config': str(run / 'run_config.json'),
            'architecture': reference['architecture'], 'planned_config': str(path),
            'planned_config_record': pinned(path), 'endpoint': 'primary'})
    evaluation['task_specs'] = tasks
    evaluation['scope'] = f'{kind}: matched final checkpoints on the same eight2022 origins and40 leads.'
    evaluation['manifest_hash'] = protocol_digest(evaluation)
    write(experiment / 'evaluation/manifest.json', evaluation)
    evaluator.load_manifest(experiment / 'evaluation/manifest.json')
    inputs.append(pinned(experiment / 'evaluation/manifest.json'))
    source, provenance = freeze(Path(prior['source_root']), experiment / 'source', workspace)
    write(experiment / 'source_provenance.json', provenance)
    inputs.append(pinned(experiment / 'source_provenance.json'))
    return seal(experiment / 'manifest.json', {'format': FORMAT, 'experiment': str(experiment),
        'workspace': str(workspace), 'kind': kind, 'source_root': str(experiment / 'source'),
        'source_digest': source['source_digest'], 'inputs': inputs,
        'evaluation_data': prior['evaluation_data'], 'initial': initial,
        'initial_step': 200 if kind == 'continuation' else 0, 'stage_endpoints': endpoints(kind),
        'resources': measured_budget(parent), 'prior_report': completed['report'],
        'initialization': ('Exact resume of both update200 trajectories.' if kind == 'continuation' else
            'DeepMind spatial overlay; Mamba1 seed22; zero temporal/residual output heads; fresh optimizer/RNG/cursor/state; MGDA from first update.'),
        'policy': {'arms': list(ARMS), 'max_steps': final, 'feedback_mode': 'closed_loop',
            'objective_pair': 'mse_amse_excess', 'finite_diagnostics': 'warning',
            'automatic_retries': 0, 'automatic_extension': False, 'terminal_stage': 'report',
            'checkpoint_averaging': False, 'selection_allowed': False}})


def verify(experiment, *, frozen=False):
    manifest = unseal(experiment / 'manifest.json')
    if manifest['format'] != FORMAT or manifest['experiment'] != str(experiment.resolve()):
        raise ValueError('Campaign identity differs')
    if frozen and ROOT != Path(manifest['source_root']):
        raise ValueError('Use the frozen driver')
    if verify_source(manifest['source_root'])['source_digest'] != manifest['source_digest']:
        raise ValueError('Frozen source differs')
    for record in manifest['inputs']:
        check(record)
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    for data in manifest['evaluation_data']:
        verify_frozen_zarr_metadata(data['root'], data['metadata'])
    return manifest


def proof_path(experiment, arm, phase):
    return experiment / 'checks' / f'{arm}_phase{phase:02d}.json'


def train_proof(experiment, manifest, arm, phase=None):
    phase = len(manifest['stage_endpoints']) - 1 if phase is None else phase
    proof = unseal(proof_path(experiment, arm, phase))
    for key, value in {'passed': True, 'arm': arm, 'phase': phase, 'completed_steps': manifest['stage_endpoints'][phase],
                       'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest']}.items():
        if proof.get(key) != value:
            raise ValueError(f'Training proof differs: {key}')
    for name in ('checkpoint', 'metrics', 'history'):
        check(proof[name])
    return proof


def training_command(experiment, manifest, arm, phase, resume):
    command = [sys.executable, '-u', Path(manifest['source_root']) / 'scripts/training/train_v24_Ilya.py',
        '--config', experiment / 'configs' / f'{arm}.json', '--stop-after-steps',
        str(manifest['stage_endpoints'][phase])]
    if resume is not None:
        command += ['--resume', str(resume)]
    return command


def train(experiment, task, phase):
    manifest = verify(experiment, frozen=True)
    gpu_required()
    if task not in range(len(ARMS)) or phase not in range(len(manifest['stage_endpoints'])):
        raise ValueError('Invalid task or phase')
    arm = ARMS[task]
    run = experiment / 'runs' / arm
    if proof_path(experiment, arm, phase).exists():
        raise FileExistsError('Completed stage must not be repeated')
    if (run / 'failed_update.json').exists():
        raise ValueError('A failed run requires diagnosis, not automatic retry')
    if phase:
        previous = train_proof(experiment, manifest, arm, phase - 1)
        resume = previous['checkpoint']['path']
    elif manifest['kind'] == 'continuation':
        initial = manifest['initial'][arm]
        for name, record in initial['history'].items():
            if file_sha256(run / name) != record['sha256']:
                raise ValueError('Initial continuation history differs')
        resume = initial['checkpoint']['path']
    else:
        if run.exists():
            raise FileExistsError('Fresh training output already exists')
        resume = None
    started = time.monotonic()
    call(training_command(experiment, manifest, arm, phase, resume))
    stop = manifest['stage_endpoints'][phase]
    records, checkpoint = pilot.validate_run(experiment, arm, expected_steps=stop)
    return seal(proof_path(experiment, arm, phase), {'passed': True, 'arm': arm, 'phase': phase,
        'completed_steps': stop, 'manifest_hash': manifest['record_hash'],
        'source_digest': manifest['source_digest'], 'checkpoint': pinned(checkpoint),
        'metrics': pinned(run / 'train_metrics.jsonl'), 'history': pinned(run / 'backtracking_trials.jsonl'),
        'elapsed_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        'warning_counts': {key: sum(bool(row.get(key)) for row in records)
                           for key in ('decrease_warning', 'repeatability_warning', 'stalled')}})


def evaluate(experiment, task, stage):
    import importlib
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from src.models.mamba.v24_Ilya.spectral_observer import SpectralObserver
    manifest = verify(experiment, frozen=True)
    gpu_required()
    for arm in ARMS:
        train_proof(experiment, manifest, arm)
    path = experiment / 'evaluation/manifest.json'
    evaluation_manifest = evaluator.load_manifest(path)
    item = evaluation_manifest['task_specs'][task]
    spec = evaluator.make_specification(path, task, stage)
    destination = evaluator.evaluation_directory(evaluation_manifest, item, stage)
    if destination.exists():
        raise FileExistsError('Existing evaluation requires diagnosis')
    write(destination / 'specification.json', spec)
    identity = {key: spec[key] for key in ('protocol_hash', 'model_id', 'checkpoint_sha256')}
    identity['baseline_checkpoint_sha256'] = spec['baseline']['checkpoint_sha256']
    observer = None
    if stage == 'spectral':
        module, function = evaluation_manifest['observer_factory'].split(':', 1)
        observer = evaluator.CompositeObserver([
            SpectralObserver(destination, identity, target_steps=40, export_fields=False),
            evaluator.GlobalFieldObserver(destination, identity, target_steps=40),
            getattr(importlib.import_module(module), function)(spec, destination, identity)])
    config = V24IlyaEvalConfig(ckpt=Path(spec['checkpoint']), out_json=destination / 'evaluation.unpublished.json',
        ckpt_in=Path(spec['baseline']['path']), stats_dir=Path(spec['stats_dir']),
        data_path=spec['data_path'], val_year=2022, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=8, warmup_steps=0, omit_rms_bias=True,
        initialization_manifest=Path(spec['initialization_manifest']['path']),
        model_id=spec['model_id'], **spec['forecast'], **spec['architecture'])
    result = evaluate_v24_Ilya(config, prediction_observer=observer)
    result.update(identity)
    evaluator.validate_result(result, spec)
    write(destination / 'evaluation.json', result)
    (destination / 'evaluation.unpublished.json').unlink(missing_ok=True)
    if stage == 'spectral':
        evaluator.load_spectral_result(destination, spec, result)


def report(experiment):
    import numpy as np
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    manifest = verify(experiment, frozen=True)
    training = {arm: train_proof(experiment, manifest, arm) for arm in ARMS}
    path = experiment / 'evaluation/manifest.json'
    evaluation = evaluator.load_manifest(path)
    summaries, features, canonical = [], [], None
    previous = {row['arm']: row for row in read(manifest['prior_report']['path'])['summary']}
    for index, item in enumerate(evaluation['task_specs']):
        _, _, exact, _ = evaluator.resolve_result(path, index, 'exact')
        directory, spec, result, _ = evaluator.resolve_result(path, index, 'spectral')
        rows = evaluator.feature_rows(evaluator.load_spectral_result(directory, spec, result))
        loss = exact['original_graphcast_loss']
        if index == 0:
            canonical = loss
        own = np.asarray(loss['baseline_per_step'])
        reference = np.asarray(canonical['baseline_per_step'])
        summary = {'id': item['id'], 'arm': item['family'], 'exact_gc_loss': loss['full_rollout'],
            'exact_gc_gain_pct': 100 * (1 - loss['full_rollout'] / canonical['baseline_rollout']),
            'loss_change_vs_best_mse_pct': 100 * (loss['full_rollout'] / canonical['full_rollout'] - 1),
            'own_baseline_gain_pct': loss['improvement_pct_rollout'],
            'baseline_max_abs_drift': float(np.max(np.abs(own - reference))),
            'baseline_relative_l2_drift': float(np.linalg.norm(own - reference) / np.linalg.norm(reference)),
            'power_error_pct_feature_day_mean': float(np.mean([r['mean_relative_power_error_pct'] for r in rows])),
            'coherence_feature_day_mean': float(np.mean([r['mean_coherence'] for r in rows]))}
        if manifest['kind'] == 'continuation' and item['family'] in ARMS:
            summary['loss_change_vs_step200_pct'] = 100 * (loss['full_rollout'] / previous[item['family']]['exact_gc_loss'] - 1)
        summaries.append(summary)
        features.extend({'id': item['id'], **row} for row in rows)
    out = experiment / 'evaluation/report'
    pilot.write_csv(out / 'summary.csv', summaries)
    pilot.write_csv(out / 'spectral_features.csv', features)
    write(out / 'report.json', {'status': 'complete', 'kind': manifest['kind'], 'summary': summaries,
        'training': training, 'max_steps': manifest['policy']['max_steps'],
        'cohort': {'initialization_times': list(pilot.DATES), 'leads': 40, 'role': 'development'},
        'reference': 'Parent task is the best trained MSE reference; plain GC is its frozen baseline execution.',
        'scientific_success_is_separate': True, 'automatic_extension': False,
        'metric_notes': 'Exact GC loss first; power/coherence are equal feature/day means. Finite drift is advisory.'})
    lines = ['# Full-BPTT MGDA initialization comparison', '', f"{manifest['kind']}, {manifest['policy']['max_steps']} updates; eight2022 starts ×40 leads.", '',
        '| Model | Exact GC loss | Gain vs plain GC (%) | Change vs best MSE (%) |', '|---|---:|---:|---:|']
    lines += [f"| {r['id']} | {r['exact_gc_loss']:.6f} | {r['exact_gc_gain_pct']:.3f} | {r['loss_change_vs_best_mse_pct']:.3f} |" for r in summaries]
    lines += ['', 'Power/coherence and finite baseline drift are retained in report.json. Single-seed development comparison; completion does not establish scientific improvement.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    return seal(experiment / 'complete.json', {'status': 'complete', 'terminal_stage': 'report',
        'manifest_hash': manifest['record_hash'], 'report': pinned(out / 'report.json')})


def sbatch(experiment, manifest, stage, *, memory, minutes, gpu=False, array=None, dependency=None, phase=0):
    if not 1 <= memory <= 512 or not 1 <= minutes <= 8640:
        raise ValueError('Resource request outside bounds')
    name = f'{stage}-{phase:02d}' if stage == 'train' else stage
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={manifest['workspace']}",
        '--kill-on-invalid-dep=yes', '--cpus-per-task=8' if gpu else '--cpus-per-task=2',
        f'--mem={memory}G', f'--time={minutes}', f"--job-name={manifest['kind']}-{name}",
        f'--output={experiment}/logs/{name}_%A_%a.out', f'--error={experiment}/logs/{name}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80', '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium' if minutes <= 4320 else 'gpu-long')]
    if array is not None:
        if '%' in array:
            raise ValueError('No array throttle')
        args.append(f'--array={array}')
    if dependency:
        dependency_kind = 'aftercorr' if stage == 'train' else 'afterok'
        args.append(f'--dependency={dependency_kind}:{dependency}')
    return args + [str(Path(manifest['source_root']) / WRAPPER), stage, str(experiment), str(phase)]


def submission_plan(experiment, manifest):
    nodes = []
    dependency = None
    for phase, stop in enumerate(manifest['stage_endpoints']):
        name = f'train_{phase:02d}'
        command = sbatch(experiment, manifest, 'train', **stage_budget(manifest, phase),
                         gpu=True, array='0-1', dependency=dependency, phase=phase)
        nodes.append({'id': name, 'depends_on': dependency, 'stop_after_steps': stop, 'command': command})
        dependency = f'<{name}>'
    for stage in ('exact', 'spectral'):
        nodes.append({'id': f'evaluate_{stage}', 'depends_on': dependency,
            'command': sbatch(experiment, manifest, f'evaluate-{stage}', memory=16, minutes=62,
                              gpu=True, array='1-2', dependency=dependency)})
    dependency = '<evaluate_exact>:<evaluate_spectral>'
    nodes.append({'id': 'report', 'depends_on': dependency,
        'command': sbatch(experiment, manifest, 'report', memory=12, minutes=30, dependency=dependency)})
    return nodes


def submit(experiment):
    manifest = verify(experiment)
    jobs = {}
    for node in submission_plan(experiment, manifest):
        command = [str(arg) for arg in node['command']]
        for name, job in jobs.items():
            command = [arg.replace(f'<{name}>', job) for arg in command]
        jobs[node['id']] = submit_once(experiment, node['id'], command)
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'dry-run', 'submit', 'train', 'evaluate-exact', 'evaluate-spectral', 'report'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path(PARENT))
    parser.add_argument('--kind', choices=('continuation', 'fresh'))
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--phase', type=int, default=0)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if args.stage == 'prepare':
        if args.kind is None:
            parser.error('prepare requires --kind')
        manifest = prepare(experiment, args.parent, ROOT, args.kind)
        result = submission_plan(experiment, manifest)
        write(experiment / 'submission_plan.json', result)
    elif args.stage == 'dry-run':
        result = submission_plan(experiment, verify(experiment))
    elif args.stage == 'train':
        result = train(experiment, args.task, args.phase)
    elif args.stage.startswith('evaluate-'):
        result = evaluate(experiment, args.task, args.stage.split('-', 1)[1])
    else:
        result = globals()[args.stage](experiment)
    if result is not None:
        import json
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
