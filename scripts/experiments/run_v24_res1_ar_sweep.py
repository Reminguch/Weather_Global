#!/usr/bin/env python3
"""Frozen five-horizon sweep of the selected batch-4, fast-Mamba recipe."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import run_v24_res1_batch4_lr as base
from scripts.experiments.frozen_runtime import file_sha256, freeze_source

WORKSPACE = Path(os.environ.get('WEATHER_WORKSPACE', ROOT)).resolve()
HORIZONS = (1, 4, 8, 12, 16)
SWA_STEPS = (500, 1000, 1500, 2000)
WINNER = 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'


def read(path):
    return json.loads(Path(path).read_text())


def configs_for(experiment, k):
    """Retain the historical K convention: K1 is teacher forcing (tail=0)."""
    result = {}
    for stage, template in [('warmup', 'warmup_di16_fast_mamba'),
                            ('train', 'di16_fast_mamba')]:
        cfg = read(WORKSPACE / WINNER / 'configs' / f'{template}.json')
        for key in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
            cfg['data'][key] = str((WORKSPACE / cfg['data'][key]).resolve())
        cfg['sequence']['ar_tail_k'] = 0 if k == 1 else k
        cfg['output'] = dict(output_root=str(experiment / ('warmup' if stage == 'warmup' else 'runs')),
                             run_name=f'k{k:02d}')
        result[stage] = cfg
    # A short independent GPU check exercises the exact horizon and optimizer.
    smoke = copy.deepcopy(result['train'])
    smoke['optimizer'].update(max_steps=3, checkpoint_every=3, warmup_steps=0,
                              learning_rate_schedule='constant', end_learning_rate=None)
    smoke['output']['output_root'] = str(experiment / 'smoke')
    result['smoke'] = smoke
    return result


def prepare(experiment):
    from src.data_operations.loaders.zarr_metadata import frozen_zarr_metadata
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.runner import validate_input_paths
    if experiment.exists():
        raise FileExistsError(f'Use a new experiment directory: {experiment}')
    metadata = frozen_zarr_metadata(base.EVAL_DATA)
    reference = WORKSPACE / WINNER / 'runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json'
    if len(read(reference)['chosen_idx']) != 32:
        raise ValueError('Expected the matched32 reference')
    entries, config_hashes = [], {}
    for k in HORIZONS:
        paths = {}
        for stage, cfg in configs_for(experiment, k).items():
            path = experiment / 'configs' / f'k{k:02d}_{stage}.json'
            base.write_json(path, cfg)
            validate_input_paths(load_training_config(path))
            paths[stage] = str(path)
            config_hashes[str(path)] = file_sha256(path)
        entries.append(dict(task=len(entries), name=f'k{k:02d}', k=k, di=16,
                            ar_tail_k=0 if k == 1 else k,
                            config=paths['train'], warmup_config=paths['warmup'],
                            smoke_config=paths['smoke'],
                            init_from=str(experiment / f'warmup/k{k:02d}/checkpoints/checkpoint_step00000500.pkl')))
    cfg = read(entries[0]['config'])
    data = cfg['data']
    pinned = [Path(data['baseline_checkpoint']), reference,
              Path(data['prepared_root']) / 'metadata.json',
              Path(data['anchor_manifest_root']) / 'metadata.json']
    pinned += list((Path(data['anchor_manifest_root']) / 'anchors').glob('*.npy'))
    pinned += list(Path(data['stats_dir']).glob('*.nc'))
    pinned += [WORKSPACE / WINNER / 'configs' / f'{name}.json'
               for name in ('warmup_di16_fast_mamba', 'di16_fast_mamba')]
    for directory in ('checks', 'logs', 'results'):
        (experiment / directory).mkdir(parents=True, exist_ok=True)
    freeze_source(WORKSPACE, experiment / 'source')
    base.write_json(experiment / 'manifest.json', dict(
        workspace=str(WORKSPACE), source_root=str(experiment / 'source'),
        source_manifest_sha256=file_sha256(experiment / 'source/snapshot.json'),
        config_hashes=config_hashes, input_hashes={str(p): file_sha256(p) for p in pinned},
        eval_data=base.EVAL_DATA, eval_data_metadata=metadata,
        evaluation_reference=str(reference), entries=entries, seed=22,
        warmup_updates=500, branch_updates=2000, swa_steps=list(SWA_STEPS),
        initialization='Same baseline overlay and random seed; each K trained from initialization; no K20-trained parent',
        stage_transfer='Parameters only; new optimizer, zero recurrent state, reset RNG/cursor as in selected batch4 recipe',
        k_convention='Historical K scan: K1 -> ar_tail_k=0; otherwise ar_tail_k=K. Longest forecast chain is ar_tail_k+1.',
        training_budget='2500 optimizer updates per arm, batch4 x BPTT24; independent 3-update smoke excluded',
        primary_checkpoint='swa_step00500-02000',
        evaluation='32 matched 2022 origins, 40 six-hour leads, cold full feedback, zero initial recurrent state; own paired baseline per evaluation',
    ))
    print(experiment)


def check_checkpoint(config_path, step):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    cfg = load_training_config(Path(config_path))
    path = cfg.run_dir / f'checkpoints/checkpoint_step{step:08d}.pkl'
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != step or checkpoint.resolved_training_config != cfg.to_dict():
        raise ValueError(f'Checkpoint does not match the prepared stage: {path}')
    if not all(np.isfinite(np.asarray(v)).all() for p in checkpoint.residual_params.values() for v in p.values()):
        raise ValueError(f'Nonfinite parameters: {path}')
    records = [json.loads(line) for line in (cfg.run_dir / 'train_metrics.jsonl').read_text().splitlines()]
    if len(records) != step or not all(np.isfinite(r['loss']) and r['batch_size'] == 4 for r in records):
        raise ValueError(f'Incomplete or nonfinite training: {path}')
    return dict(checkpoint=str(path), sha256=file_sha256(path), completed_step=step,
                median_step_seconds=float(np.median([r['step_seconds'] for r in records])))


def train(entry, experiment):
    base.train(entry['smoke_config'])
    smoke = check_checkpoint(entry['smoke_config'], 3)
    base.write_json(experiment / f"checks/{entry['name']}_smoke.json", smoke)
    base.train(entry['warmup_config'])
    warmup = check_checkpoint(entry['warmup_config'], 500)
    base.write_json(experiment / f"checks/{entry['name']}_warmup.json", warmup)
    base.train(entry['config'], entry['init_from'])
    result = check_checkpoint(entry['config'], 2000)
    directory = experiment / 'runs' / entry['name']
    base.run('scripts/training/build_v24_Ilya_swa.py', '--inputs',
             *[directory / f'checkpoints/checkpoint_step{s:08d}.pkl' for s in SWA_STEPS],
             '--source-steps', *SWA_STEPS,
             '--output', directory / 'swa/swa_step00500-02000.pkl')
    base.write_json(experiment / f"checks/{entry['name']}_train.json", result)


def summarize(manifest, experiment):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for entry in manifest['entries']:
        for tag in ('step2000', 'swa_step00500-02000'):
            path = experiment / 'runs' / entry['name'] / 'eval/cold_full_zero_exact_gc/matched32' / f'{tag}.json'
            payload = read(path)
            if payload['chosen_idx'] != read(manifest['evaluation_reference'])['chosen_idx'] or payload['evaluated_samples'] != 32:
                raise ValueError(f'Unmatched evaluation: {path}')
            loss = payload['original_graphcast_loss']
            score = 100 * (1 - loss['full_rollout'] / loss['baseline_rollout'])
            if not np.isfinite(score) or not np.isclose(score, loss['improvement_pct_rollout']):
                raise ValueError(f'Invalid exact loss ratio: {path}')
            rows.append(dict(k=entry['k'], ar_tail_k=entry['ar_tail_k'], checkpoint=tag,
                             baseline_rollout=loss['baseline_rollout'], full_rollout=loss['full_rollout'],
                             exact_gc_loss_reduction_pct=score,
                             day10_gc_loss_reduction_pct=loss['improvement_pct_per_step'][39], source=str(path)))
            if tag.startswith('swa'):
                ax.plot(np.arange(1, 41) / 4, loss['improvement_pct_per_step'], label=f"K={entry['k']}")
    ax.axhline(0, color='black', linewidth=.6)
    ax.set(xlabel='Forecast lead (days)', ylabel='Exact GraphCast loss reduction (%)')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    for suffix in ('pdf', 'png'):
        fig.savefig(experiment / f'results/rollout_improvement.{suffix}', dpi=180)
    plt.close(fig)
    with (experiment / 'results/summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base.write_json(experiment / 'results/summary.json', rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'train', 'eval', 'summarize'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == 'prepare':
        prepare(experiment)
        return
    manifest = read(experiment / 'manifest.json')
    base.EXPERIMENT = experiment
    base.WORKSPACE = Path(manifest['workspace'])
    base.REFERENCE = Path(manifest['evaluation_reference'])
    base.EVAL_DATA = manifest['eval_data']
    base.BRANCH_UPDATES = manifest['branch_updates']
    base.SWA_STEPS = tuple(manifest['swa_steps'])
    base.require_runtime(manifest)
    for path, expected in manifest['input_hashes'].items():
        if file_sha256(path) != expected:
            raise ValueError(f'Prepared input changed: {path}')
    if args.stage == 'verify':
        print('Frozen source, configurations, datasets and reference verified')
    elif args.stage == 'summarize':
        summarize(manifest, experiment)
    else:
        if not 0 <= args.task < len(manifest['entries']):
            raise ValueError(f'Invalid array task: {args.task}')
        entry = manifest['entries'][args.task]
        if args.stage == 'train':
            train(entry, experiment)
        else:
            base.evaluate(entry)


if __name__ == '__main__':
    main()
