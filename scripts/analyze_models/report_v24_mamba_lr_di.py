"""Report exact 10-day GC losses for the batch-4 Mamba LR/inner-width study.

Reads existing evaluations only. Captures input bytes so live partial results
remain auditable, and keeps unmatched partial results out of completed rankings.
Run through scripts/graphcast_env.sh.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'
NARROW = ROOT / 'artifacts/checkpoints/v24_Ilya/res1_width128_mamba1_20260909'
WIDE = ROOT / ('artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/'
               'mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k/'
               'eval/cold_full_zero_exact_gc/matched_v22_reference32/swa_step02000-08000.json')
SWA = 'swa_step00500-02000'
DAYS = np.arange(1, 41) / 4
CONDITIONS = ['slow_mamba', 'joint', 'fast_mamba']
LABELS = {'slow_mamba': 'Slower Mamba (0.3x)', 'joint': 'Joint LR (1x)',
          'fast_mamba': 'Faster Mamba (3x)', 'slow_spatial': 'Slower spatial (0.3x)',
          'frozen_spatial': 'Frozen spatial'}
COLORS = {'slow_mamba': '#ba593b', 'joint': '#207a79', 'fast_mamba': '#7357a5'}
INK, MUTED = '#17354a', '#536673'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    slug = output.stem
    data_dir = ROOT / 'plots/analyze_models/data/resolution_eval' / slug
    image_dir = ROOT / 'plots/analyze_models/images/resolution_eval' / slug
    for directory in [output.parent, data_dir, image_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    sources, cache = [], {}

    def read(path):
        path = Path(path)
        if path in cache:
            return cache[path]
        raw = path.read_bytes()
        relative = path.relative_to(ROOT)
        snapshot = data_dir / 'input_snapshot' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        sources.append(dict(path=str(relative), snapshot=str(snapshot.relative_to(ROOT)),
                            sha256=hashlib.sha256(raw).hexdigest()))
        cache[path] = json.loads(raw)
        return cache[path]

    manifest = read(EXP / 'manifest.json')
    historical_ref = read(Path(manifest['evaluation_reference']))
    ref = read(EXP / 'runs/di16_joint/eval/cold_full_zero_exact_gc/matched32' / f'{SWA}.json')
    assert ref['chosen_idx'] == historical_ref['chosen_idx']
    assert ref['ckpt_in'] == historical_ref['ckpt_in']
    assert len(set(ref['chosen_idx'])) == 32 and ref['target_steps'] == 40
    checks = []

    def validate(d, path):
        for key in ['chosen_idx', 'target_steps', 'eval_mode', 'eval_feedback',
                    'residual_state_init', 'resolution',
                    'anchor_history_steps', 'anchor_index_semantics', 'warmup_steps']:
            assert d[key] == ref[key], (path, key)
        assert (ROOT / d['ckpt_in']).resolve() == (ROOT / ref['ckpt_in']).resolve(), path
        n = d['evaluated_samples']
        complete = d.get('evaluation_status', 'complete') == 'complete'
        assert 0 <= n <= 32
        if complete:
            assert n == 32 and d.get('metrics_complete', True), path
        assert d.get('completed_chosen_idx', ref['chosen_idx']) == ref['chosen_idx'][:n]
        if not n:
            return
        g = d['original_graphcast_loss']
        b, f = np.array(g['baseline_per_step']), np.array(g['full_per_step'])
        assert b.shape == f.shape == (40,) and np.isfinite([b, f]).all()
        assert (b > 0).all()
        assert np.allclose(g['improvement_pct_per_step'], 100 * (1 - f / b))
        assert np.isclose(g['baseline_rollout'], b.mean())
        assert np.isclose(g['full_rollout'], f.mean())
        assert np.isclose(g['improvement_pct_rollout'],
                          100 * (1 - g['full_rollout'] / g['baseline_rollout']))
        baseline_diff = None
        if complete:
            baseline_diff = float(np.max(np.abs(b / np.array(ref['original_graphcast_loss']['baseline_per_step']) - 1)))
            assert baseline_diff < .01, (path, baseline_diff)
        checks.append(dict(path=str(path.relative_to(ROOT)), samples=n, complete=complete,
                           max_baseline_relative_difference=baseline_diff))

    results, rows, starts = {}, [], {}
    for entry in manifest['entries']:
        name = entry['name']
        cfg = read(Path(entry['config']))
        assert cfg['architecture']['temporal_d_inner'] == entry['di']
        assert cfg['optimizer']['mamba_lr_multiplier'] == entry['mamba_lr_multiplier']
        assert cfg['optimizer']['spatial_lr_multiplier'] == entry['spatial_lr_multiplier']
        starts[name] = read(EXP / 'checks' / f'start_{name}.json')
        for tag in ['shared_start', 'step2000', SWA]:
            directory = EXP / 'runs' / name / 'eval/cold_full_zero_exact_gc/matched32'
            path = directory / f'{tag}.json'
            if not path.exists():
                path = directory / f'{tag}.partial.json'
            try:
                d = read(path)
            except FileNotFoundError:
                path = directory / f'{tag}.json'
                if not path.exists():
                    continue
                d = read(path)
            validate(d, path)
            results[name, tag] = d
            g = d['original_graphcast_loss']
            rows.append(dict(name=name, di=entry['di'], condition=entry['condition'],
                             artifact=tag, status=d['evaluation_status'], samples=d['evaluated_samples'],
                             spatial_lr_multiplier=entry['spatial_lr_multiplier'],
                             mamba_lr_multiplier=entry['mamba_lr_multiplier'],
                             baseline_rollout=g['baseline_rollout'], full_rollout=g['full_rollout'],
                             rollout_reduction_pct=g['improvement_pct_rollout'],
                             day10_reduction_pct=g['improvement_pct_per_step'][39] if d['evaluated_samples'] else None,
                             source=str(path.relative_to(ROOT))))

    wide = read(WIDE)
    narrow_dir = NARROW / 'runs/w128_fresh_mamba1_di16_bcg1_rank32_seed22/eval/cold_full_zero_exact_gc/matched32'
    narrow_swa = read(narrow_dir / 'swa_step02000-08000.json')
    narrow_final = read(narrow_dir / 'step10000.json')
    narrow_summary = read(NARROW / 'results/summary.json')
    for d, p in [(wide, WIDE), (narrow_swa, narrow_dir / 'swa_step02000-08000.json'),
                 (narrow_final, narrow_dir / 'step10000.json')]:
        validate(d, p)

    def complete(name, tag=SWA):
        d = results.get((name, tag))
        return d if d and d['evaluation_status'] == 'complete' else None

    def pct(d, day=None):
        if d is None:
            return None
        g = d['original_graphcast_loss']
        return g['improvement_pct_rollout'] if day is None else g['improvement_pct_per_step'][day * 4 - 1]

    def fmt(value):
        return f'{value:.2f}%' if value is not None else 'Pending'

    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (data_dir / 'per_lead_gc_loss.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name', 'artifact', 'status', 'samples', 'lead_days', 'baseline_gc_loss',
                         'model_gc_loss', 'gc_loss_reduction_pct'])
        for (name, tag), d in results.items():
            if not d['evaluated_samples']:
                continue
            g = d['original_graphcast_loss']
            for i, day in enumerate(DAYS):
                writer.writerow([name, tag, d['evaluation_status'], d['evaluated_samples'], day,
                                 g['baseline_per_step'][i], g['full_per_step'][i], g['improvement_pct_per_step'][i]])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42})
    page_number = 0

    def page(title, subtitle):
        nonlocal page_number
        page_number += 1
        fig = plt.figure(figsize=(11.7, 8.3), facecolor='white')
        fig.text(.06, .935, title, fontsize=23, weight='bold')
        fig.text(.06, .89, subtitle, fontsize=10, color=MUTED)
        fig.text(.06, .035, f'v24 Mamba LR / di study  |  snapshot {now}', fontsize=8, color=MUTED)
        fig.text(.94, .035, str(page_number), ha='right', fontsize=8, color=MUTED)
        return fig

    def table(fig, bounds, headers, cells, widths=None, fontsize=10):
        ax = fig.add_axes(bounds)
        ax.axis('off')
        t = ax.table(cellText=cells, colLabels=headers, colWidths=widths,
                     loc='center', cellLoc='left', bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(fontsize)
        for (i, _), cell in t.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#d4e9e6' if i == 0 else ('#f0f5f7' if i % 2 else '#f8fafb'))
            if i == 0:
                cell.set_text_props(weight='bold')
        return t

    def style(ax, ylabel, title):
        ax.set(xlim=(.25, 10), xlabel='Forecast lead (days)', ylabel=ylabel, title=title)
        ax.set_xticks([1, 2, 4, 6, 8, 10])
        ax.grid(alpha=.18)

    def save(pdf, fig):
        pdf.savefig(fig)
        fig.savefig(image_dir / f'page_{page_number:02d}.png', dpi=130)
        plt.close(fig)

    with PdfPages(output, metadata={'Title': 'v24 Mamba learning rate and inner width: 10-day GraphCast loss',
                                    'Author': 'Weather_global experiment analysis'}) as pdf:
        fig = page('Mamba learning rate and inner width',
                   'Exact original GraphCast loss  |  1 degree  |  batch 4  |  di16 and di32  |  10-day forecasts')
        fig.text(.06, .82, 'Completed results favor joint LR; faster-Mamba evaluation is still incomplete.'
                 if any(complete(f'di{di}_fast_mamba') is None for di in [16, 32])
                 else 'Completed slower, joint and faster Mamba comparisons', fontsize=15, weight='bold')
        cells = []
        for di in [16, 32]:
            for condition in CONDITIONS:
                name = f'di{di}_{condition}'
                d, final = complete(name), complete(name, 'step2000')
                delta = pct(d) - pct(complete(f'di{di}_joint')) if d else None
                cells.append([f'di{di}', LABELS[condition], fmt(pct(final)), fmt(pct(d)),
                              fmt(pct(d, 10)), f'{delta:+.2f} pp' if delta is not None else 'Pending'])
        table(fig, [.06, .45, .88, .31], ['Inner width', 'Mamba LR', 'Final rollout', 'SWA rollout',
                                          'SWA day 10', 'SWA vs joint'], cells,
              [.12, .25, .16, .16, .16, .15], fontsize=10)
        findings = []
        for di in [16, 32]:
            slow, joint = complete(f'di{di}_slow_mamba'), complete(f'di{di}_joint')
            findings.append(f'di{di}: slower Mamba {pct(slow):.2f}% vs joint {pct(joint):.2f}% '
                            f'rollout reduction ({pct(slow)-pct(joint):+.2f} percentage points).')
        findings += [f'Increasing di16 to di32: joint {pct(complete("di32_joint"))-pct(complete("di16_joint")):+.2f} pp; '
                     f'slower Mamba {pct(complete("di32_slow_mamba"))-pct(complete("di16_slow_mamba")):+.2f} pp (SWA).',
                     'Different warm-up weights and one training seed limit causal conclusions.',
                     '', 'Higher reduction is better. All headline scores require 32 completed forecast starts.',
                     'Rollout: aggregate all 40 six-hour losses before taking the ratio. Day 10: the final lead only.',
                     'Final = branch update 2,000. SWA = average of branch checkpoints 500, 1,000, 1,500, 2,000.']
        fig.text(.06, .405, '\n'.join(findings), va='top', fontsize=10.5, linespacing=1.65)
        save(pdf, fig)

        fig = page('Forecast loss across all 10 days',
                   'SWA checkpoints  |  complete 32-start evaluations  |  solid colors identify Mamba learning-rate multipliers')
        for col, di in enumerate([16, 32]):
            left = .075 + .48 * col
            ax_loss = fig.add_axes([left, .53, .38, .28])
            ax_imp = fig.add_axes([left, .16, .38, .28])
            baselines = []
            for condition in CONDITIONS:
                d = complete(f'di{di}_{condition}')
                if d is None:
                    continue
                g = d['original_graphcast_loss']
                baselines.append(g['baseline_per_step'])
                ax_loss.plot(DAYS, g['full_per_step'], color=COLORS[condition], lw=2, label=LABELS[condition])
                ax_imp.plot(DAYS, g['improvement_pct_per_step'], color=COLORS[condition], lw=2)
            b = complete(f'di{di}_joint')['original_graphcast_loss']['baseline_per_step']
            ax_loss.plot(DAYS, b, color='#333333', ls='--', lw=1.7, label='Frozen GraphCast')
            ax_loss.fill_between(DAYS, np.min(baselines, axis=0), np.max(baselines, axis=0), color='gray', alpha=.15)
            ax_imp.axhline(0, color='gray', lw=1)
            style(ax_loss, 'GC loss (normalized weighted MSE)', f'di{di}: absolute loss (lower is better)')
            style(ax_imp, 'GC loss reduction (%)', f'di{di}: improvement (higher is better)')
            ax_loss.set_ylim(bottom=0)
            ax_imp.set_ylim(-3, 28)
            ax_loss.legend(fontsize=8, frameon=False, loc='upper left')
        fig.text(.06, .08, 'Each improvement uses its own matched baseline. Gray bands show baseline variation across completed runs.',
                 fontsize=9, color=MUTED)
        save(pdf, fig)

        fig = page('How the result depends on di',
                   'di is the Mamba inner dimension; spatial width stays 512 and BC groups stay 1')
        ax = fig.add_axes([.08, .57, .4, .24])
        for condition in CONDITIONS:
            a, b = complete(f'di16_{condition}'), complete(f'di32_{condition}')
            if a is not None and b is not None:
                delta = np.array(b['original_graphcast_loss']['improvement_pct_per_step']) - a['original_graphcast_loss']['improvement_pct_per_step']
                ax.plot(DAYS, delta, color=COLORS[condition], lw=2, label=LABELS[condition])
        style(ax, 'di32 minus di16 (percentage points)', 'SWA improvement difference by lead')
        ax.axhline(0, color='gray', lw=1)
        ax.legend(fontsize=8, frameon=False)
        daily = []
        for di in [16, 32]:
            for condition in CONDITIONS:
                d = complete(f'di{di}_{condition}')
                daily.append([f'di{di} {condition.replace("_mamba", "")}', *[fmt(pct(d, day)) for day in [1, 5, 10]]])
        table(fig, [.55, .565, .39, .27], ['SWA', 'Day 1', 'Day 5', 'Day 10'], daily,
              [.4, .2, .2, .2], fontsize=9)
        gain_rows = []
        for di in [16, 32]:
            for condition in ['joint', 'slow_mamba']:
                name = f'di{di}_{condition}'
                start, d = complete(name, 'shared_start'), complete(name)
                gain_rows.append([name, fmt(pct(start)), fmt(pct(d)), f'{pct(d)-pct(start):+.2f} pp'])
        table(fig, [.06, .275, .88, .205], ['Arm', 'Own warm-up', 'SWA rollout', 'Gain over own warm-up'],
              gain_rows, [.34, .22, .22, .22], fontsize=10)
        fig.text(.06, .225,
                 'The slower-Mamba rollout score increases more with di (+1.11 pp) than the joint score (+0.62 pp).\n'
                 'The gap to joint narrows from 0.97 pp at di16 to 0.48 pp at di32.\n'
                 'These are descriptive differences: all ten warm-up parameter hashes differ. In particular, di16 slower\n'
                 'Mamba starts weaker but gains more during continuation. A shared-checkpoint comparison is needed\n'
                 'to attribute the final-score difference to learning rate alone.', fontsize=10, va='top', linespacing=1.5)
        save(pdf, fig)

        fig = page('Faster Mamba: evaluation progress',
                   'Peak Mamba LR 3e-4; spatial LR 1e-4  |  final branch checkpoint shown here; SWA reported separately')
        statuses = []
        for col, di in enumerate([16, 32]):
            name = f'di{di}_fast_mamba'
            d = results.get((name, 'step2000'))
            n = d['evaluated_samples'] if d else 0
            status = d['evaluation_status'] if d else 'not started'
            g = d['original_graphcast_loss'] if n else None
            ax_loss = fig.add_axes([.075 + .48 * col, .53, .38, .25])
            ax_imp = fig.add_axes([.075 + .48 * col, .19, .38, .25])
            if g:
                ax_loss.plot(DAYS, g['baseline_per_step'], ls='--', color='#333333', label='Frozen GC on this subset')
                ax_loss.plot(DAYS, g['full_per_step'], color=COLORS['fast_mamba'], lw=2, label='Faster Mamba')
                ax_imp.plot(DAYS, g['improvement_pct_per_step'], color=COLORS['fast_mamba'], lw=2)
                ax_loss.legend(fontsize=8, frameon=False)
            ax_imp.axhline(0, color='gray', lw=1)
            style(ax_loss, 'GC loss (normalized weighted MSE)', f'di{di}: {status.upper()} - {n}/32 starts')
            style(ax_imp, 'GC loss reduction (%)', f'Rollout {fmt(pct(d)) if n else "pending"}; day 10 {fmt(pct(d, 10)) if n else "pending"}')
            swa = results.get((name, SWA))
            statuses.append(f'di{di} SWA: {swa["evaluation_status"] + " " + str(swa["evaluated_samples"]) + "/32" if swa else "not yet available"}')
        fig.text(.06, .83, 'PROVISIONAL: partial curves use different subsets; do not compare their scores across di or with full32.'
                 if any(complete(f'di{di}_fast_mamba', 'step2000') is None for di in [16, 32])
                 else 'Both final checkpoints have completed the matched 32-start evaluation.',
                 fontsize=10, color='#a3422f', weight='bold')
        fig.text(.06, .105, '  |  '.join(statuses) + '\n'
                 'Snapshots retain the exact input bytes and completed start IDs. Missing per-start losses prevent a matched-subset comparison.',
                 fontsize=9, color=MUTED, linespacing=1.6)
        save(pdf, fig)

        fig = page('Smaller residual GraphCast: cost and skill',
                   'Separate batch-1 experiment  |  10,000 updates  |  di16 / BCG1  |  same 32 starts and 40 forecast leads')
        small_items = [('Width512 pretrained / SWA 2k-8k', wide, '#207a79'),
                       ('Width128 fresh / SWA 2k-8k', narrow_swa, '#ba593b'),
                       ('Width128 fresh / final 10k', narrow_final, '#7357a5')]
        axes = [fig.add_axes([.075, .39, .38, .39]), fig.add_axes([.555, .39, .38, .39])]
        for label, d, color in small_items:
            g = d['original_graphcast_loss']
            axes[0].plot(DAYS, g['full_per_step'], label=label, color=color, lw=2)
            axes[1].plot(DAYS, g['improvement_pct_per_step'], color=color, lw=2)
        axes[0].plot(DAYS, wide['original_graphcast_loss']['baseline_per_step'], color='#333333', ls='--', label='Frozen GraphCast')
        axes[1].axhline(0, color='gray', lw=1)
        style(axes[0], 'GC loss (normalized weighted MSE)', 'Absolute forecast loss')
        style(axes[1], 'GC loss reduction (%)', 'Improvement over GraphCast')
        axes[0].legend(fontsize=8, frameon=False)
        params = [r['residual_parameters'] for r in narrow_summary['rows']]
        table(fig, [.06, .17, .88, .15], ['Residual branch / artifact', 'Parameters', 'Rollout reduction', 'Day-10 reduction'],
              [[label, f'{params[0] if i == 0 else params[1]:,}', fmt(pct(d)), fmt(pct(d, 10))]
               for i, (label, d, _) in enumerate(small_items)], [.43, .17, .2, .2], fontsize=10)
        fig.text(.06, .115, 'Width128 has 92.94% fewer residual parameters, retaining 22.28% of the reference SWA benefit.\n'
                 'Width and pretrained-versus-fresh initialization both change; this does not isolate width. Baseline width stays 512.',
                 fontsize=10, linespacing=1.6)
        save(pdf, fig)

        fig = page('All arms, methods and evidence',
                   'Completed-only score table  |  exact normalized, latitude-, pressure- and variable-weighted GraphCast MSE')
        all_cells = []
        for entry in manifest['entries']:
            name = entry['name']
            d, f = complete(name), complete(name, 'step2000')
            all_cells.append([name, fmt(pct(complete(name, 'shared_start'))), fmt(pct(f)), fmt(pct(d)), fmt(pct(d, 10))])
        table(fig, [.06, .43, .88, .39], ['Arm', 'Warm-up rollout', 'Final rollout', 'SWA rollout', 'SWA day 10'],
              all_cells, [.34, .17, .17, .16, .16], fontsize=9)
        max_diff = max(c['max_baseline_relative_difference'] or 0 for c in checks)
        fig.text(.06, .39,
                 'METHOD\n'
                 'Loss reduction = 100 x (1 - model loss / matched frozen-baseline loss). Rollout losses aggregate\n'
                 'all starts and 40 six-hour leads before the ratio; percentages are never averaged across leads.\n'
                 'Training: 500 private warm-up + 2,000 branch updates; batch 4; seed 22; Mamba1; residual width512/MP2;\n'
                 'di16 or di32; BCG1; state size16; 2 temporal layers. Peak LR 1e-4 with Mamba factors 0.3 / 1 / 3,\n'
                 '50-update warm-up, cosine to 1e-5 before group scaling; AdamW (0.9, 0.98), weight decay 1e-4.\n'
                 'Evaluation: 2022, same 32 start IDs, cold full feedback, zero state, no history warm start.\n'
                 f'Checked completeness, IDs, metadata, loss ratios and arrays. Maximum matched baseline deviation: {100*max_diff:.3f}%.\n'
                 'Partial previews have fewer starts; no common per-start losses are saved for retrospective matching.\n\n'
                 'EVIDENCE AND REPRODUCTION\n'
                 f'Report inputs, hashes and checks: {output.with_suffix(".sources.json").name}\n'
                 f'Numerical summary: {output.with_suffix(".csv").name}\n'
                 'Input snapshots and per-lead CSV: plots/analyze_models/data/resolution_eval/' + slug + '/\n'
                 'Generator: scripts/analyze_models/report_v24_mamba_lr_di.py (GraphCast environment).\n'
                 'One seed, differing warm-up weights and a development evaluation set: no significance claim.',
                 fontsize=9, va='top', linespacing=1.45)
        save(pdf, fig)

    output.with_suffix('.sources.json').write_text(json.dumps(dict(
        generated_utc=now, report=str(output.relative_to(ROOT)), sources=sources,
        validation=checks, warmup_hashes={name: value['residual_params_sha256'] for name, value in starts.items()},
        partial_policy='Separate previews only; no ranking against completed evaluations or across unequal subsets.',
        per_lead_csv=str((data_dir / 'per_lead_gc_loss.csv').relative_to(ROOT)),
        figure_directory=str(image_dir.relative_to(ROOT))), indent=2) + '\n')
    print(json.dumps(dict(pdf=str(output), pages=page_number, input_files=len(sources),
                          validated_evaluations=len(checks), snapshot_utc=now), indent=2))


if __name__ == '__main__':
    main()
