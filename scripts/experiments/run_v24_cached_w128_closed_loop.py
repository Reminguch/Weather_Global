"""Frozen closed-loop K20 continuation of selected cached width128/512 runs."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, verify_source, source_environment
from scripts.experiments.run_v24_mgda import (
    call, check, file_sha256, pinned, read, seal, submit_once, unseal, write,
)

PARENTS = 'artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007'
REFERENCE = 'artifacts/checkpoints/v24_Ilya/res1_w128_di16_k20_finetune_20260918'
ARMS = ('di32', 'di64')
PARENT_STEPS = (8000, 14000)
STEPS = (500, 1000, 1500, 2000)
DRIVER = 'scripts/experiments/run_v24_cached_w128_closed_loop.py'
WRAPPER = 'scripts/experiments/v24_cached_w128_closed_loop.slurm'
W512_PARENT = 'artifacts/checkpoints/v24_Ilya/res1_cached_stepwise_pair_recovery_20261007/runs/cached'


def make_config(template, experiment, arm, *, smoke=False):
    config = copy.deepcopy(template)
    config.pop('derived', None)
    config.pop('config_source', None)
    config.pop('shared_init_sha256', None)
    config['sequence']['feedback_mode'] = 'closed_loop_sg'
    config['optimizer'].update(max_steps=3 if smoke else 2000,
        checkpoint_every=3 if smoke else 500, learning_rate=1e-5,
        learning_rate_schedule='constant', end_learning_rate=None,
        warmup_steps=0 if smoke else 200)
    config['validation'].update(enabled=not smoke, every_steps=500,
                                num_segments=8, final_num_segments=8)
    config['output'] = {'output_root': str(experiment / ('smoke' if smoke else 'runs')),
                        'run_name': arm}
    return config


def prepare(experiment, workspace, *, width512=False):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.runner import validate_input_paths
    from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata
    from scripts.experiments.run_v24_amse_split_mgda import coordinate_coverage
    if experiment.exists():
        raise FileExistsError(f'Use a new experiment directory: {experiment}')
    reference_root = workspace / REFERENCE
    reference = read(reference_root / 'reference_evaluation.json')
    eval_data = read(reference_root / 'manifest.json')['eval_data']
    metadata = frozen_zarr_metadata(eval_data)
    coverage = coordinate_coverage({'data_path': eval_data, 'stage': 'exact',
        'evaluation_year': 2022, 'chosen_idx': reference['chosen_idx'],
        'n_samples': 32, 'target_steps': 40})
    if reference['evaluation_status'] != 'complete' or len(reference['chosen_idx']) != 32:
        raise ValueError('Incomplete matched32 reference')
    entries, inputs, configs = [], [], []
    selected = [('w512_di64', 4000, workspace / W512_PARENT)] if width512 else [
        (arm, step, workspace / PARENTS / f'{arm}_carry' / 'runs/cached')
        for arm, step in zip(ARMS, PARENT_STEPS)]
    for arm, parent_step, parent in selected:
        cp_path = parent / f'checkpoints/checkpoint_step{parent_step:08d}.pkl'
        cp = load_v24_Ilya_training_checkpoint(cp_path)
        template = read(parent / 'run_config.json')
        template.pop('shared_init_sha256', None)
        architecture = template['architecture']
        if (cp.completed_step != parent_step or cp.resolved_training_config != template
                or architecture['temporal_d_inner'] != (64 if width512 else int(arm[2:]))
                or (architecture['residual_width'] or architecture['width']) != (512 if width512 else 128)
                or architecture['temporal_bc_groups'] != (2 if width512 else 1)
                or template['sequence'] != {'ar_tail_k': 20, 'bptt_steps': 24,
                    'feedback_mode': 'baseline', 'segment_steps': 96, 'temporal_state_policy': 'carry'}):
            raise ValueError('Parent checkpoint/configuration does not match the selected arm')
        leaves = [v for values in cp.residual_params.values() for v in values.values()]
        if not all(np.isfinite(np.asarray(v)).all() for v in leaves):
            raise ValueError('Nonfinite parent parameters')
        parent_record = pinned(cp_path)
        entry = {'arm': arm, 'parent_step': parent_step, 'parent': parent_record,
                 'baseline_fingerprint': cp.baseline_checkpoint_fingerprint,
                 'anchor_fingerprint': cp.anchor_manifest_fingerprint,
                 'parameter_count': sum(v.size for v in leaves), 'architecture': architecture}
        for smoke in (False, True):
            key = 'smoke_config' if smoke else 'config'
            path = experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json'
            write(path, make_config(template, experiment, arm, smoke=smoke))
            cfg = load_training_config(path)
            validate_input_paths(cfg)
            if cfg.ar_tail_k != 20 or cfg.bptt_steps != 24 or cfg.distributed.global_batch_size != 1:
                raise ValueError('Unexpected continuation shape')
            entry[key] = str(path)
            configs.append(pinned(path))
        entries.append(entry)
        inputs += [parent_record, pinned(parent / 'run_config.json')]
        del cp, leaves
    data = template['data']
    inputs += [pinned(data['baseline_checkpoint']), pinned(Path(data['prepared_root']) / 'metadata.json'),
               pinned(Path(data['anchor_manifest_root']) / 'metadata.json')]
    inputs += [pinned(p) for p in sorted((Path(data['anchor_manifest_root']) / 'anchors').glob('*.npy'))]
    inputs += [pinned(Path(data['stats_dir']) / n) for n in
               ('mean_by_level.nc', 'stddev_by_level.nc', 'diffs_stddev_by_level.nc')]
    inputs.append(pinned(reference_root / 'reference_evaluation.json'))
    for name in ('logs', 'checks', 'results', 'submissions', 'evaluation'):
        (experiment / name).mkdir(exist_ok=True)
    write(experiment / 'checks/coordinate_coverage.json', coverage)
    snapshot = freeze_source(workspace, experiment / 'source')
    seal(experiment / 'manifest.json', {
        'format': 'cached_closed_loop_k20_v2', 'workspace': str(workspace),
        'residual_width': 512 if width512 else 128,
        'source_root': str(experiment / 'source'), 'source_digest': snapshot['source_digest'],
        'entries': entries, 'configs': configs, 'inputs': inputs,
        'eval_data': eval_data, 'eval_metadata': metadata, 'chosen_idx': reference['chosen_idx'],
        'evaluation_steps': [0, *STEPS], 'primary_endpoint': 2000,
        'evaluation_protocol': '2022 development, matched32 cold_full, 40 six-hour leads, zero memory, corrected feedback',
        'stage_transfer': 'All parent residual weights; new Adam/RNG/cursor; explicitly zero recurrent state',
        'training_protocol': 'batch1 BPTT24 segment96 K20 closed_loop_sg all-step BF16 objective; carry memory within segments',
        'numerical_policy': 'Finite repeated-baseline drift advisory; identities, finite values and completeness blocking',
        'resources': ({'w512_di64': {'memory': 416, 'minutes': 600},
                       'eval': {'memory': 20, 'minutes': 120}} if width512 else
                     {'di32': {'memory': 352, 'minutes': 480},
                      'di64': {'memory': 384, 'minutes': 480},
                      'eval': {'memory': 16, 'minutes': 90}}),
        'resource_evidence': ({'online_same_architecture_job': '15295410',
            'elapsed': '13:04:04', 'updates': 4000, 'peak_host_gib': 304058992 / 1024**2,
            'eval_job': '15352692', 'two_evals_elapsed': '02:12:53',
            'eval_peak_host_gib': 9727.5 / 1024,
            'rationale': '416 GiB is 1.43x same-architecture online peak; 10h is 1.53x half of observed 4k elapsed, allowing smoke and closed-loop overhead. 20 GiB/2h per exact32 evaluation allows the wider di64/BCG2 residual.'}
            if width512 else {'online_di16_job': '14117544', 'elapsed': '04:25:13',
            'peak_host_gib': 255160964 / 1024**2, 'updates': 2000,
            'eval_job': '14117546', 'five_evals_elapsed': '05:12:39',
            'eval_peak_host_gib': 10205248 / 1024**2,
            'rationale': 'Same online width128/batch1/K20 pipeline; 1.45x/1.58x host-memory headroom for di32/di64; 8h includes independent smoke, validation and wider-state overhead. Each exact32 evaluation receives 1.44x historical per-artifact elapsed.'})})
    verify(experiment)
    print(experiment)


def verify(experiment, *, runtime=False):
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    m = unseal(experiment / 'manifest.json')
    if verify_source(m['source_root'])['source_digest'] != m['source_digest']:
        raise ValueError('Frozen source digest differs')
    if runtime and ROOT.resolve() != Path(m['source_root']).resolve():
        raise ValueError('Execute work from the frozen source')
    for record in m['inputs'] + m['configs']:
        check(record)
    verify_frozen_zarr_metadata(m['eval_data'], m['eval_metadata'])
    return m


def train_command(entry, *, smoke=False):
    return [sys.executable, '-u', ROOT / 'scripts/training/train_v24_Ilya.py',
            '--config', entry['smoke_config' if smoke else 'config'],
            '--init-from', entry['parent']['path'], '--zero-state-on-init']


def checkpoint_path(experiment, arm, step, *, smoke=False):
    return experiment / ('smoke' if smoke else 'runs') / arm / 'checkpoints' / f'checkpoint_step{step:08d}.pkl'


def check_training(experiment, entry, *, smoke=False):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    cfg = load_training_config(Path(entry['smoke_config' if smoke else 'config']))
    target = 3 if smoke else 2000
    reports = []
    for step in ((3,) if smoke else STEPS):
        path = checkpoint_path(experiment, entry['arm'], step, smoke=smoke)
        cp = load_v24_Ilya_training_checkpoint(path)
        if cp.completed_step != step or cp.resolved_training_config != cfg.to_dict():
            raise ValueError('Native checkpoint step or configuration mismatch')
        meta = cp.parameter_overlay_metadata
        if meta.get('init_from_sha256') != entry['parent']['sha256'] or meta.get('zero_state_on_init') is not True:
            raise ValueError('Wrong warm-start identity or initial state policy')
        if cp.baseline_checkpoint_fingerprint != entry['baseline_fingerprint'] or cp.anchor_manifest_fingerprint != entry['anchor_fingerprint']:
            raise ValueError('Baseline/anchor identity changed')
        leaves = [v for values in cp.residual_params.values() for v in values.values()]
        if not all(np.isfinite(np.asarray(v)).all() for v in leaves):
            raise ValueError('Nonfinite trained parameters')
        reports.append(pinned(path))
    rows = [json.loads(line) for line in (cfg.run_dir / 'train_metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1, target + 1)):
        raise ValueError('Missing or duplicated training steps')
    if not all(math.isfinite(r[k]) for r in rows for k in ('loss', 'gradient_norm')):
        raise ValueError('Nonfinite training')
    if not smoke:
        vals = [json.loads(line) for line in (cfg.run_dir / 'validation_metrics.jsonl').read_text().splitlines()]
        fixed = [r for r in vals if r['role'] == 'fixed_checkpoint']
        if {r['step'] for r in fixed} != set(STEPS) or not all(math.isfinite(r['loss']) for r in vals):
            raise ValueError('Missing/nonfinite validation')
        if len({r['subset_fingerprint'] for r in vals}) != 1:
            raise ValueError('Validation subset changed')
    return {'passed': True, 'completed_step': target, 'checkpoints': reports,
            'median_steady_seconds': float(np.median([r['step_seconds'] for r in rows[1:]]))}


def eval_command(m, entry, checkpoint, output, *, smoke=False):
    cfg = read(entry['config'])
    data, arch = cfg['data'], cfg['architecture']
    args = [sys.executable, '-u', ROOT / 'scripts/analyze_models/eval_v24_Ilya.py',
        '--ckpt', checkpoint, '--ckpt-in', data['baseline_checkpoint'],
        '--data-path', m['eval_data'], '--stats-dir', data['stats_dir'],
        '--resolution', str(arch['resolution']), '--mesh-size', str(arch['mesh_size']),
        '--width', str(arch['width']),
        '--residual-initialization', arch['residual_initialization'],
        '--baseline-msg-steps', str(arch['baseline_msg_steps']), '--residual-msg-steps', str(arch['residual_msg_steps']),
        '--val-year', '2022', '--train-start-year', '2015', '--train-end-year', '2021',
        '--input-duration', '12h', '--temporal-location', arch['temporal_location'],
        '--temporal-d-inner', str(arch['temporal_d_inner']), '--temporal-d-state', str(arch['temporal_d_state']),
        '--temporal-d-conv', str(arch['temporal_d_conv']), '--temporal-dt-rank', str(arch['temporal_dt_rank']),
        '--temporal-init-scheme', arch['temporal_init_scheme'],
        '--temporal-layers', str(arch['temporal_layers']), '--temporal-bc-groups', str(arch['temporal_bc_groups']), '--temporal-stateful',
        '--eval-mode', 'cold_full', '--residual-state-init', 'zero', '--residual-alpha', '1',
        '--target-steps', '2' if smoke else '40', '--warmup-steps', '24',
        '--n-samples', '1' if smoke else '32', '--seed', '0', '--omit-rms-bias', '--out-json', output]
    if arch['residual_width'] is not None:
        args += ['--residual-width', str(arch['residual_width'])]
    return args


def validate_evaluation(result, m, checkpoint, *, smoke=False):
    import numpy as np
    if Path(result['ckpt']).resolve() != Path(checkpoint).resolve():
        raise ValueError('Evaluated checkpoint identity mismatch')
    baseline_paths = {read(e['config'])['data']['baseline_checkpoint'] for e in m['entries']}
    if result['ckpt_in'] not in baseline_paths or len(baseline_paths) != 1:
        raise ValueError('Evaluated baseline identity mismatch')
    if (result['evaluation_status'] != 'complete'
            or result['evaluated_samples'] != (1 if smoke else 32)
            or result['target_steps'] != (2 if smoke else 40)
            or result['eval_mode'] != 'cold_full' or result['eval_feedback'] != 'full'
            or result['residual_state_init'] != 'zero'):
        raise ValueError('Evaluation protocol/completeness mismatch')
    if not smoke and result['chosen_idx'] != m['chosen_idx']:
        raise ValueError('Evaluation origins differ from matched32')
    g = result['original_graphcast_loss']
    b, f = np.array(g['baseline_per_step']), np.array(g['full_per_step'])
    if b.shape != (2 if smoke else 40,) or f.shape != b.shape or not np.isfinite([b, f]).all() or not (b > 0).all():
        raise ValueError('Invalid exact loss arrays')
    if not np.isclose(g['baseline_rollout'], b.mean()) or not np.isclose(g['full_rollout'], f.mean()):
        raise ValueError('Exact loss aggregation mismatch')
    if not np.isclose(g['improvement_pct_rollout'], 100 * (1 - f.sum() / b.sum())):
        raise ValueError('Exact improvement aggregation mismatch')
    return {'passed': True, 'checkpoint': pinned(checkpoint)}


def train(experiment, m, entry):
    if (experiment / 'runs' / entry['arm']).exists() or (experiment / 'smoke' / entry['arm']).exists():
        raise FileExistsError('Existing run: use explicit recovery instead of overwriting')
    call(train_command(entry, smoke=True))
    proof = check_training(experiment, entry, smoke=True)
    cp = checkpoint_path(experiment, entry['arm'], 3, smoke=True)
    out = experiment / 'checks' / f"{entry['arm']}_smoke_eval.json"
    call(eval_command(m, entry, cp, out, smoke=True))
    proof['evaluation'] = validate_evaluation(read(out), m, cp, smoke=True)
    write(experiment / 'checks' / f"{entry['arm']}_preflight.json", proof)
    call(train_command(entry))
    write(experiment / 'checks' / f"{entry['arm']}_train.json", check_training(experiment, entry))


def evaluate(experiment, m, task):
    if not 0 <= task < len(m['entries']) * 5:
        raise ValueError('Evaluation task outside the manifest')
    entry = m['entries'][task // 5]
    step = (0, *STEPS)[task % 5]
    cp = Path(entry['parent']['path']) if step == 0 else checkpoint_path(experiment, entry['arm'], step)
    if step:
        proof = read(experiment / 'checks' / f"{entry['arm']}_train.json")
        if proof.get('passed') is not True:
            raise ValueError('Training completion gate did not pass')
        check(next(r for r in proof['checkpoints'] if r['path'] == str(cp)))
    out = experiment / 'evaluation' / f"{entry['arm']}_step{step:05d}.json"
    if out.exists():
        raise FileExistsError(out)
    call(eval_command(m, entry, cp, out))
    proof = validate_evaluation(read(out), m, cp)
    proof['output'] = pinned(out)
    write(out.with_suffix('.proof.json'), proof)


def summarize(experiment, m):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (experiment / 'results').mkdir(exist_ok=True)
    results = {}
    arms = [entry['arm'] for entry in m['entries']]
    for arm in arms:
        for step in (0, *STEPS):
            path = experiment / 'evaluation' / f'{arm}_step{step:05d}.json'
            proof = read(path.with_suffix('.proof.json'))
            if proof.get('passed') is not True:
                raise ValueError('Missing evaluation proof')
            check(proof['output']); check(proof['checkpoint'])
            result = read(path)
            validate_evaluation(result, m, proof['checkpoint']['path'])
            results[arm, step] = result
    canonical = results[arms[0], 0]['original_graphcast_loss']
    b = np.array(canonical['baseline_per_step'])
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for arm, color in zip(arms, ('#1f77b4', '#df7b16')):
        initial = results[arm, 0]['original_graphcast_loss']
        for step in (0, *STEPS):
            g = results[arm, step]['original_graphcast_loss']
            f, own_b = np.array(g['full_per_step']), np.array(g['baseline_per_step'])
            rows.append({'arm': arm, 'continuation_step': step,
                'baseline_loss_canonical': float(b.mean()), 'model_loss': float(f.mean()),
                'loss_reduction_pct_canonical': float(100 * (1 - f.sum() / b.sum())),
                'day10_loss_reduction_pct_canonical': float(100 * (1 - f[-1] / b[-1])),
                'loss_reduction_pct_own_baseline': g['improvement_pct_rollout'],
                'day10_loss_reduction_pct_own_baseline': float(100 * (1 - f[-1] / own_b[-1])),
                'loss_reduction_pct_vs_parent': float(100 * (1 - f.mean() / initial['full_rollout'])),
                'pooled_baseline_drift_pct': float(100 * (own_b.mean() / b.mean() - 1)),
                'max_lead_baseline_drift_pct': float(100 * np.max(np.abs(own_b / b - 1))),
                'source': str(experiment / 'evaluation' / f'{arm}_step{step:05d}.json')})
            if step in (0, 2000):
                label = f'{arm} ' + ('parent' if step == 0 else '+2k closed loop')
                axes[0].plot(np.arange(1, 41) / 4, f, color=color, ls='--' if step == 0 else '-', label=label)
                axes[1].plot(np.arange(1, 41) / 4, 100 * (1 - f / b), color=color, ls='--' if step == 0 else '-', label=label)
    axes[0].plot(np.arange(1, 41) / 4, b, color='black', lw=1, label='Frozen GC')
    axes[0].set_ylabel('Exact original GraphCast loss')
    axes[1].set_ylabel('Exact loss reduction vs canonical GC (%)')
    for ax in axes:
        ax.set_xlabel('Forecast lead (days)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"Cached width{m.get('residual_width', 128)} to closed-loop K20 · matched32 development evaluation")
    images = Path(m['workspace']) / 'plots/analyze_models/images/resolution_eval' / experiment.name
    images.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(images / f'closed_loop_comparison.{ext}', dpi=180)
    with (experiment / 'results/summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(experiment / 'results/summary.json', {'status': 'complete', 'rows': rows,
        'primary_endpoint': 2000, 'cohort': '2022 development matched32',
        'canonical_baseline': f'{arms[0]} parent execution', 'numerical_drift': 'advisory, reported per artifact',
        'interpretation': 'Checkpoint selection and scores use development data; no independent-test or causal feedback-only claim. Intermediate checkpoints are diagnostics; no further training is released.'})


def sbatch(experiment, m, stage, *, task=None, array=None, dependency=None, arm=None):
    gpu = stage != 'summarize'
    budget = m['resources'][arm if stage == 'train' else 'eval'] if gpu else {'memory': 8, 'minutes': 20}
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={m['workspace']}",
        '--cpus-per-task=8' if gpu else '--cpus-per-task=2', f"--mem={budget['memory']}G",
        f"--time={budget['minutes']}", f'--job-name=w{m.get("residual_width", 128)}-cl20-{stage}-{arm or "all"}',
        f'--output={experiment}/logs/{stage}_%A_%a.out', f'--error={experiment}/logs/{stage}_%A_%a.err',
        '--kill-on-invalid-dep=yes']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80',
                 '--qos=gpu-medium' if budget['minutes'] > 480 else '--qos=gpu-short']
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    if array is not None:
        args.append(f'--array={array}')
    args += [str(Path(m['source_root']) / WRAPPER), stage, str(experiment), m['workspace']]
    if task is not None:
        args.append(str(task))
    return args


def submit(experiment, m, *, dry_run=False):
    jobs, commands = {}, {}
    def add(name, stage, **kwargs):
        args = sbatch(experiment, m, stage, **kwargs)
        commands[name] = args
        jobs[name] = f'<{name}>' if dry_run else submit_once(experiment, name, args)
    arms = [entry['arm'] for entry in m['entries']]
    add('parents', 'eval', array=','.join(str(i * 5) for i in range(len(arms))))
    for task, arm in enumerate(arms):
        add(f'train_{arm}', 'train', task=task, arm=arm)
        add(f'eval_{arm}', 'eval', array=f'{task * 5 + 1}-{task * 5 + 4}', dependency=jobs[f'train_{arm}'])
    add('report', 'summarize', task=0, dependency=':'.join(jobs[k] for k in ['parents', *[f'eval_{arm}' for arm in arms]]))
    write(experiment / ('submission_dry_run.json' if dry_run else 'jobs.json'), {'jobs': jobs, 'commands': commands})
    print(json.dumps(jobs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'submit', 'train', 'eval', 'summarize'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--width512', action='store_true', help='Prepare the cached width512 di64/BCG2 update-4000 continuation')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == 'prepare':
        prepare(experiment, args.workspace.resolve(), width512=args.width512); return
    m = verify(experiment, runtime=args.stage in ('train', 'eval', 'summarize'))
    if args.stage == 'submit':
        submit(experiment, m, dry_run=args.dry_run); return
    if args.stage == 'verify':
        print('Frozen source, checkpoints, configurations and data metadata verified.'); return
    os.environ.update(source_environment(m['source_root'], m['workspace']))
    if args.stage in ('train', 'eval'):
        import jax
        if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != 'gpu':
            raise RuntimeError('GPU stage requires one allocated GPU')
    try:
        if args.stage == 'train':
            if not 0 <= args.task < len(m['entries']):
                raise ValueError('Invalid training task')
            train(experiment, m, m['entries'][args.task])
        elif args.stage == 'eval':
            evaluate(experiment, m, args.task)
        else:
            summarize(experiment, m)
    except Exception as error:
        write(experiment / 'failures' / f'{args.stage}_{args.task}.json',
              {'stage': args.stage, 'task': args.task, 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
