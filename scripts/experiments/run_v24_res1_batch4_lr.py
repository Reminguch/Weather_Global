#!/usr/bin/env python3
"""Prepare/run the documented six-arm batch-4 temporal-LR experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'
BASE_CONFIG = ROOT / 'configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904/mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k.json'
REFERENCE = ROOT / 'artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json'
ARMS = [('joint', 1., 1.), ('slow_mamba', 1., .3), ('frozen_spatial', 0., 1.)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def prepare():
    if (EXPERIMENT / 'manifest.json').exists():
        raise RuntimeError('Experiment already prepared; refusing to overwrite its manifest')
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg['distributed'].update(mode='single', num_devices=1, per_device_batch_size=4)
    cfg['validation']['enabled'] = False
    cfg['optimizer'].update(max_steps=2000, checkpoint_every=250, warmup_steps=50)
    cfg['output']['output_root'] = str(EXPERIMENT / 'runs')
    entries = []
    for di in (16, 32):
        seed = copy.deepcopy(cfg)
        seed['architecture']['temporal_d_inner'] = di
        seed['output'].update(output_root=str(EXPERIMENT / 'seeds'), run_name=f'di{di}_shared')
        seed['optimizer'].update(max_steps=500, checkpoint_every=250,
                                 learning_rate_schedule='constant', end_learning_rate=None)
        write_json(EXPERIMENT / f'configs/seed_di{di}.json', seed)
        start = EXPERIMENT / f'seeds/di{di}_shared/checkpoints/checkpoint_step00000500.pkl'
        for arm, spatial, mamba in ARMS:
            item = copy.deepcopy(cfg)
            name = f'di{di}_{arm}'
            item['architecture']['temporal_d_inner'] = di
            item['optimizer'].update(spatial_lr_multiplier=spatial, mamba_lr_multiplier=mamba)
            item['output']['run_name'] = name
            path = EXPERIMENT / f'configs/{name}.json'
            write_json(path, item)
            entries.append(dict(task=len(entries), name=name, di=di, condition=arm,
                                config=str(path), init_from=str(start),
                                spatial_lr_multiplier=spatial, mamba_lr_multiplier=mamba))
    for directory in ('logs', 'source', 'checks', 'results'):
        (EXPERIMENT / directory).mkdir(parents=True, exist_ok=True)
    write_json(EXPERIMENT / 'manifest.json', dict(
        experiment='res1_batch4_temporal_lr_20260908', batch_size=4,
        seed_updates=500, branch_updates=2000, seed=22, entries=entries,
        evaluation_reference=str(REFERENCE),
        stage_order=['smoke (2 widths)', 'shared warm-up (2 widths)', 'train (6 arms)', 'exact evaluation (6 arms)'],
    ))
    print(EXPERIMENT)


def train(config, start=None, resume=None):
    # Force each independent experiment stage to initialize a fresh process.
    args = ['scripts/training/train_v24_Ilya.py', '--config', config]
    if start is not None:
        args += ['--init-from', start]
    if resume is not None:
        args += ['--resume', resume]
    run(*args)


def prepare_branch_start(entry):
    """Run an independent, reproducibly configured warm-up in this allocation."""
    if 'warmup_config' not in entry:
        return
    train(entry['warmup_config'])
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_checkpoint
    import numpy as np
    params = load_v24_Ilya_checkpoint(entry['init_from']).residual_params
    digest = hashlib.sha256()
    for module in sorted(params):
        for name in sorted(params[module]):
            value = np.asarray(params[module][name])
            digest.update(f'{module}/{name}:{value.dtype}:{value.shape}\n'.encode())
            digest.update(value.tobytes())
    write_json(EXPERIMENT / f"checks/start_{entry['name']}.json", dict(
        checkpoint=entry['init_from'], residual_params_sha256=digest.hexdigest()))


def smoke(di):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_checkpoint
    from src.models.mamba.v24_Ilya.training.endpoint_step import is_mamba_parameter
    cfg = json.loads((EXPERIMENT / f'configs/seed_di{di}.json').read_text())
    cfg['output'].update(output_root=str(EXPERIMENT / 'smoke'), run_name=f'di{di}_joint')
    cfg['optimizer'].update(max_steps=3, checkpoint_every=1, warmup_steps=0)
    path = EXPERIMENT / f'configs/smoke_di{di}_joint.json'
    write_json(path, cfg)
    train(path)
    start = EXPERIMENT / f'smoke/di{di}_joint/checkpoints/checkpoint_step00000003.pkl'
    cfg['output']['run_name'] = f'di{di}_frozen'
    cfg['optimizer'].update(max_steps=2, spatial_lr_multiplier=0., mamba_lr_multiplier=1.)
    path = EXPERIMENT / f'configs/smoke_di{di}_frozen.json'
    write_json(path, cfg)
    train(path, start)
    before = load_v24_Ilya_checkpoint(start).residual_params
    after_path = EXPERIMENT / f'smoke/di{di}_frozen/checkpoints/checkpoint_step00000002.pkl'
    after = load_v24_Ilya_checkpoint(after_path).residual_params
    changed_mamba = 0
    for module, values in before.items():
        for name, value in values.items():
            equal = np.array_equal(value, after[module][name])
            if is_mamba_parameter(module):
                changed_mamba += int(not equal)
            elif not equal:
                raise AssertionError(f'Frozen parameter changed: {module}/{name}')
    assert changed_mamba > 0, 'Mamba parameters did not learn through frozen spatial layers'
    cfg['optimizer']['max_steps'] = 3
    path = EXPERIMENT / f'configs/smoke_di{di}_frozen_resume.json'
    write_json(path, cfg)
    train(path, resume=after_path)
    records = [json.loads(line) for line in (EXPERIMENT / f'smoke/di{di}_joint/train_metrics.jsonl').read_text().splitlines()]
    assert all(item['batch_size'] == 4 and np.isfinite(item['loss']) for item in records)
    write_json(EXPERIMENT / f'checks/smoke_di{di}.json', dict(
        passed=True, actual_batch_size=4, changed_mamba_tensors=changed_mamba,
        frozen_spatial_exact=True, exact_resume_completed=True,
        step_seconds=[item['step_seconds'] for item in records],
        device_memory=[item.get('device_memory') for item in records],
    ))


def evaluate(entry):
    directory = EXPERIMENT / 'runs' / entry['name']
    cfg = json.loads(Path(entry['config']).read_text())
    for tag, checkpoint in [('shared_start', Path(entry['init_from'])),
                            ('step2000', directory / 'checkpoints/checkpoint_step00002000.pkl'),
                            ('swa_step00500-02000', directory / 'swa/swa_step00500-02000.pkl')]:
        output = directory / 'eval/cold_full_zero_exact_gc/matched32' / f'{tag}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite evaluation {output}')
        run('scripts/analyze_models/eval_v24_Ilya.py',
            '--ckpt', checkpoint, '--ckpt-in', cfg['data']['baseline_checkpoint'],
            '--data-path', '/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr',
            '--stats-dir', cfg['data']['stats_dir'], '--resolution', '1.0', '--mesh-size', '5', '--width', '512',
            '--baseline-msg-steps', '16', '--residual-msg-steps', '2',
            '--val-year', '2022', '--train-start-year', '2015', '--train-end-year', '2021',
            '--input-duration', '12h', '--temporal-location', 'mesh_processor_interleaved',
            '--temporal-d-inner', entry['di'], '--temporal-d-state', '16', '--temporal-d-conv', '4',
            '--temporal-dt-rank', 'auto', '--temporal-init-scheme', 'mamba1',
            '--temporal-layers', '2', '--temporal-bc-groups', '1', '--temporal-stateful',
            '--eval-mode', 'cold_full', '--residual-state-init', 'zero', '--residual-alpha', '1',
            '--target-steps', '40', '--warmup-steps', '24', '--n-samples', '32', '--seed', '0',
            '--out-json', output)
        value = json.loads(output.read_text())
        assert value['chosen_idx'] == json.loads(REFERENCE.read_text())['chosen_idx']
        assert value['evaluated_samples'] == 32 and len(value['original_graphcast_loss']['full_per_step']) == 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'seed', 'train', 'eval', 'summarize'])
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    manifest = json.loads((EXPERIMENT / 'manifest.json').read_text())
    if args.stage == 'summarize':
        summarize(manifest)
        return
    if args.stage in ('smoke', 'seed'):
        di = (16, 32)[args.task]
        if args.stage == 'smoke':
            smoke(di)
        else:
            assert json.loads((EXPERIMENT / f'checks/smoke_di{di}.json').read_text())['passed']
            train(EXPERIMENT / f'configs/seed_di{di}.json')
    else:
        entry = manifest['entries'][args.task]
        if args.stage == 'train':
            prepare_branch_start(entry)
            train(entry['config'], entry['init_from'])
            directory = EXPERIMENT / 'runs' / entry['name']
            steps = (500, 1000, 1500, 2000)
            run('scripts/training/build_v24_Ilya_swa.py', '--inputs',
                *[directory / f'checkpoints/checkpoint_step{step:08d}.pkl' for step in steps],
                '--source-steps', *steps, '--output', directory / 'swa/swa_step00500-02000.pkl')
        else:
            evaluate(entry)


def summarize(manifest):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    rows = []
    starts_equal = {}
    for di in (16, 32):
        hashes = [json.loads((EXPERIMENT / f"checks/start_{e['name']}.json").read_text())['residual_params_sha256']
                  for e in manifest['entries'] if e['di'] == di and 'warmup_config' in e]
        starts_equal[di] = len(set(hashes)) <= 1
    write_json(EXPERIMENT / 'results/starting_weights_comparison.json', starts_equal)
    for entry in manifest['entries']:
        directory = EXPERIMENT / 'runs' / entry['name'] / 'eval/cold_full_zero_exact_gc/matched32'
        initial = json.loads((directory / 'shared_start.json').read_text())['original_graphcast_loss']
        for tag in ('step2000', 'swa_step00500-02000'):
            path = directory / f'{tag}.json'
            result = json.loads(path.read_text())['original_graphcast_loss']
            rows.append(dict(name=entry['name'], artifact=tag, di=entry['di'],
                             identical_starting_weights_within_width=starts_equal[entry['di']],
                             gc_loss_reduction_pct=result['improvement_pct_rollout'],
                             initial_gc_loss_reduction_pct=initial['improvement_pct_rollout'],
                             gain_over_start_pp=result['improvement_pct_rollout']-initial['improvement_pct_rollout'],
                             day10_gc_loss_reduction_pct=result['improvement_pct_per_step'][39], source=str(path)))
            if tag.startswith('swa'):
                axis = axes[0 if entry['di'] == 16 else 1]
                axis.plot(np.arange(1, 41)/4, result['improvement_pct_per_step'], label=entry['condition'])
                if entry['condition'] == 'joint' or not starts_equal[entry['di']]:
                    label = 'shared start' if starts_equal[entry['di']] else f"{entry['condition']} start"
                    axis.plot(np.arange(1, 41)/4, initial['improvement_pct_per_step'], ls=':', label=label)
    for di, axis in zip((16, 32), axes):
        axis.set_title(f'di{di}, batch 4, BCG1')
        axis.set_xlabel('Forecast lead (days)')
        axis.axhline(0, color='black', lw=.6)
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel('Exact original GraphCast loss reduction (%)')
    figure.tight_layout()
    figure.savefig(EXPERIMENT / 'results/comparison.png', dpi=180)
    plt.close(figure)
    with (EXPERIMENT / 'results/summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
