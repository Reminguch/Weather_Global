"""Rebuild the v24 optimizer PDF from complete matched evaluation artifacts."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904'
REF = ROOT / ('artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/'
              'di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/reports/v24_optimizer_search_2026-09-10.pdf')
    parser.add_argument('--accounting', type=Path, required=True)
    args = parser.parse_args()
    ref = json.loads(REF.read_text())
    states = {r['JobID']: r['State'] for r in csv.DictReader(args.accounting.open(), delimiter='|')}
    sources = []

    def read_result(path):
        d = json.loads(path.read_text())
        assert d['evaluated_samples'] == 32 and d['target_steps'] == 40, path
        assert d.get('evaluation_status', 'complete') == 'complete', path
        assert d.get('metrics_complete', True), path
        assert d['chosen_idx'] == ref['chosen_idx'], path
        for key in ['eval_mode', 'eval_feedback', 'residual_state_init', 'resolution', 'ckpt_in']:
            assert d[key] == ref[key], (path, key)
        m = d['original_graphcast_loss']
        assert len(m['full_per_step']) == len(m['baseline_per_step']) == 40
        assert np.isfinite(m['improvement_pct_rollout'])
        assert np.isclose(m['improvement_pct_rollout'], 100 * (1-m['full_rollout']/m['baseline_rollout']))
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return d

    ref = read_result(REF)
    rows, payloads = [], {}
    config_root = ROOT / 'configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904'
    for shape, train_id, eval_id in [('di16', '13435248', '13435335'), ('di64', '13435250', '13435336')]:
        for idx, config in enumerate(sorted(config_root.glob(f'*_{shape}_*.json'))):
            c = json.loads(config.read_text())
            name = c['output']['run_name']
            a, o = c['architecture'], c['optimizer']
            row = dict(run=name, shape=shape, init=name.split('_')[0], lr=o['learning_rate'],
                       beta1=o['adam_beta1'], beta2=o['adam_beta2'],
                       train_job=f'{train_id}_{idx}', eval_job=f'{eval_id}_{idx}',
                       train_state=states.get(f'{train_id}_{idx}', 'not in snapshot'),
                       eval_state=states.get(f'{eval_id}_{idx}', 'not in snapshot'))
            for kind in ['checkpoint', 'swa']:
                row[kind] = None
                row[kind+'_tag'] = ''
            for path in sorted((EXP/name/'eval/cold_full_zero_exact_gc/matched_v22_reference32').glob('*.json')):
                if path.stem.startswith(('step', 'swa_step')) and not path.stem.endswith('partial'):
                    d = read_result(path)
                    kind = 'swa' if path.stem.startswith('swa') else 'checkpoint'
                    assert row[kind] is None, (name, kind)
                    row[kind] = d['original_graphcast_loss']['improvement_pct_rollout']
                    row[kind+'_tag'] = path.stem
                    payloads[(name, kind)] = d
            rows.append(row)
    assert len(rows) == 24
    complete = [r for r in rows if r['checkpoint'] is not None and r['swa'] is not None]
    ranked = sorted(complete, key=lambda r: max(r['checkpoint'], r['swa']), reverse=True)
    now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    args.output.with_suffix('.sources.json').write_text(json.dumps({'generated': now, 'sources': sources,
        'accounting': args.accounting.read_text()}, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

    def page(title, subtitle):
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, title, fontsize=22, weight='bold', color='#16324f')
        fig.text(.06, .9, subtitle, fontsize=10, color='#52616b')
        fig.text(.06, .035, f'v24 optimizer matrix | refreshed {now} | seed 22', fontsize=8, color='#52616b')
        return fig

    def label(r):
        return f"{r['init']} {r['shape']} | LR {r['lr']:.0e} | ({r['beta1']}, {r['beta2']})"

    # The table reports loss at day 10, separately from the rollout-average metric.
    for r in rows:
        for kind in ['checkpoint', 'swa']:
            d = payloads.get((r['run'], kind))
            r[kind+'_day10_gc_improvement_pct'] = d['original_graphcast_loss']['improvement_pct_per_step'][39] if d else None
            r[kind+'_day10_gc_loss'] = d['original_graphcast_loss']['full_per_step'][39] if d else None
            r[kind+'_day10_t2m_rmse'] = d['per_variable_per_step']['2m_temperature']['rmse_full'][39] if d else None
    with args.output.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = [('legacy', 'di16'), ('mamba1', 'di16'), ('legacy', 'di64'), ('mamba1', 'di64')]
    selected = []
    for init, shape in groups:
        candidates = [r for r in complete if r['init'] == init and r['shape'] == shape]
        if candidates:
            selected.append(min(candidates, key=lambda r: r['swa_day10_gc_loss']))
    with PdfPages(args.output) as pdf:
        fig = page('Mamba1 di64 | completed evaluation',
                   'Exact original GraphCast loss reduction | 32 matched anchors | 40 six-hour leads')
        di64 = max((r for r in complete if r['init'] == 'mamba1' and r['shape'] == 'di64'),
                   key=lambda r: r['swa'])
        di16 = max((r for r in complete if r['init'] == 'mamba1' and r['shape'] == 'di16'),
                   key=lambda r: r['swa'])
        fig.text(.06, .80, f"{di64['swa']:.2f}% full-rollout loss reduction", fontsize=25,
                 weight='bold', color='#228877')
        fig.text(.06, .745, label(di64) + ' | BCG4 | ' + di64['swa_tag'], fontsize=11)
        comparison = [
            ['Model / checkpoint', '10-day rollout reduction', 'Day-10 reduction'],
            ['Mamba1 di64 / selected SWA', f"{di64['swa']:.2f}%", f"{di64['swa_day10_gc_improvement_pct']:.2f}%"],
            [f"Mamba1 di64 / {di64['checkpoint_tag']}", f"{di64['checkpoint']:.2f}%", f"{di64['checkpoint_day10_gc_improvement_pct']:.2f}%"],
            ['Mamba1 di16 / best rollout SWA', f"{di16['swa']:.2f}%", f"{di16['swa_day10_gc_improvement_pct']:.2f}%"],
        ]
        ax = fig.add_axes([.06, .47, .88, .22]); ax.axis('off')
        table = ax.table(cellText=comparison[1:], colLabels=comparison[0],
                         colWidths=[.44, .30, .26], cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(11)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#cee6e1' if i == 0 else '#f1f6f8')
            if i == 0: cell.set_text_props(weight='bold', color='#16324f')
        m = payloads[(di64['run'], 'swa')]['original_graphcast_loss']
        leads = '   |   '.join(f"Day {i//4}: {m['improvement_pct_per_step'][i-1]:.2f}%" for i in [4,8,16,28,40])
        lines = [
            'di64 SWA loss reduction by forecast lead', leads, '',
            f"di16 comparator: {label(di16)} | BCG1 | {di16['swa_tag']}",
            f"di64 minus di16 rollout reduction: {di64['swa']-di16['swa']:+.2f} percentage points.",
            ('Optimizer settings match; BCG count differs, so this does not isolate inner dimension.'
             if all(di64[k] == di16[k] for k in ['lr', 'beta1', 'beta2']) else
             'Optimizer settings and BCG count differ; this does not isolate the effect of inner dimension.'), '',
            'Rollout reduction = 100 x (1 - full_rollout / baseline_rollout), from saved exact losses.',
            'Losses are aggregated over samples and leads before the ratio; day-10 values are separate.',
            f"Completed checkpoint + SWA evaluations: {len(complete)}/24 configurations.",
            'Results use one training seed and an evaluation set used for model selection.',
        ]
        fig.text(.06, .415, '\n'.join(lines), va='top', fontsize=10, linespacing=1.55)
        pdf.savefig(fig); plt.close(fig)

        fig = page('Day-10 GraphCast loss improvement',
                   'SWA optimizer matrix | exact loss reduction versus matched frozen GraphCast | higher is better')
        fig.text(.06, .847, 'SWA: 2k-8k updates.  * Selected 4k-8k SWA instead.', fontsize=11, color='#16324f')
        fig.text(.06, .812, '1 degree  /  spatial width 512  /  all-24-step loss  /  10k updates  /  seed 22  /  32 anchors',
                 fontsize=9, color='#52616b')
        # Six optimizer settings by four model setups, with physically square cells.
        ax = fig.add_axes([.26, .205, .63, .53])
        settings = [(lr, b1, b2) for lr in [1e-4, 5e-5]
                    for b1, b2 in [(.8, .98), (.9, .98), (.9, .999)]]
        matrix = np.full((4, 6), np.nan)
        entries = {}
        for i, (init, shape) in enumerate(groups):
            for j, (lr, b1, b2) in enumerate(settings):
                r = next(r for r in rows if (r['init'], r['shape'], r['lr'], r['beta1'], r['beta2'])
                         == (init, shape, lr, b1, b2))
                entries[i, j] = r
                if r['swa_day10_gc_improvement_pct'] is not None:
                    matrix[i, j] = r['swa_day10_gc_improvement_pct']
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('report_teal', ['#f1f6f8', '#cee6e1', '#86bcb5'])
        cmap.set_bad('#f0f1f3')
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=30, aspect='equal')
        ax.set_xticks(range(6), [f'({b1}, {b2})' for _, b1, b2 in settings], fontsize=10)
        ax.xaxis.tick_top()
        ax.tick_params(axis='both', which='both', length=0, pad=12)
        ax.set_yticks(range(4), ['Legacy  |  di16 / BCG1', 'Mamba1  |  di16 / BCG1',
                               'Legacy  |  di64 / BCG4', 'Mamba1  |  di64 / BCG4'], fontsize=11)
        ax.text(1, 1.08, 'Peak LR  1e-4', transform=ax.get_xaxis_transform(), ha='center', fontsize=12, weight='bold', color='#16324f')
        ax.text(4, 1.08, 'Peak LR  5e-5', transform=ax.get_xaxis_transform(), ha='center', fontsize=12, weight='bold', color='#16324f')
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=3)
        ax.axvline(2.5, color='white', linewidth=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (i, j), r in entries.items():
            value = matrix[i,j]
            suffix = '*' if r['swa_tag'] and r['swa_tag'] != 'swa_step02000-08000' else ''
            ax.text(j, i, f'{value:.2f}%{suffix}' if np.isfinite(value) else '--',
                    ha='center', va='center', fontsize=14, weight='bold',
                    color='#16324f')
        ref_pct = ref['original_graphcast_loss']['improvement_pct_per_step'][39]
        fig.text(.26, .165, f'V22 reference SWA: {ref_pct:.2f}%    |    Frozen GraphCast: 0%    |    -- awaiting evaluation',
                 fontsize=10, color='#16324f')
        fig.text(.06, .10,
                 'Columns show Adam (beta1, beta2). Each cell is 100 x (1 - model loss / matched baseline loss) at day 10.\n'
                 'Setup denotes residual-Mamba initialization. This is day-10 improvement, not the rollout-average percentage.',
                 fontsize=9, color='#52616b')
        pdf.savefig(fig); plt.close(fig)

        fig = page('10-day rollout | GraphCast loss and 2 m temperature',
                   'Lowest day-10 GC-loss SWA in each initialization/di group; same 32 anchors. Lower is better.')
        axes = [fig.add_axes([.075,.37,.40,.45]), fig.add_axes([.565,.37,.40,.45])]
        days = np.arange(1,41)/4
        colors = ['#4477aa', '#228877', '#cc6677', '#aa8833']
        for r, color in zip(selected, colors):
            d = payloads[(r['run'], 'swa')]
            axes[0].plot(days, d['original_graphcast_loss']['full_per_step'], color=color, lw=2, label=label(r))
            axes[1].plot(days, d['per_variable_per_step']['2m_temperature']['rmse_full'], color=color, lw=2)
        axes[0].plot(days, ref['original_graphcast_loss']['full_per_step'], color='#555555', ls='--', lw=2, label='V22 reference SWA (2k-8k)')
        axes[1].plot(days, ref['per_variable_per_step']['2m_temperature']['rmse_full'], color='#555555', ls='--', lw=2)
        axes[0].plot(days, ref['original_graphcast_loss']['baseline_per_step'], color='black', ls=':', lw=2, label='Frozen GraphCast (reference evaluation)')
        axes[1].plot(days, ref['per_variable_per_step']['2m_temperature']['rmse_baseline'], color='black', ls=':', lw=2)
        for ax, title, ylabel in zip(axes, ['Exact GraphCast loss', '2 m temperature'],
                                    ['Normalized weighted MSE', 'RMSE (K)']):
            ax.set_title(title, fontsize=13, weight='bold'); ax.set_xlabel('Forecast lead (days)')
            ax.set_ylabel(ylabel); ax.set_xlim(.25,10); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.065,.30), ncol=2, fontsize=8.5, frameon=False)
        detail = '; '.join(f"{r['init']} {r['shape']}: {r['swa_tag'].replace('swa_step','')}" for r in selected)
        fig.text(.06,.16, 'SWA windows: '+detail, fontsize=8)
        fig.text(.06,.105,
                 'GC loss aggregates all forecast variables with the original GraphCast weights. Temperature RMSE is in kelvin.\n'
                 'Group winners are selected on these evaluation results; comparisons are descriptive (one training seed).', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        fig = page('10-day rollout | improvement over GraphCast',
                   'Same SWA models as the absolute-value plots; percentage reduction at each lead. Higher is better.')
        axes = [fig.add_axes([.075,.37,.40,.45]), fig.add_axes([.565,.37,.40,.45])]
        for r, color in zip(selected, colors):
            d = payloads[(r['run'], 'swa')]
            axes[0].plot(days, d['original_graphcast_loss']['improvement_pct_per_step'],
                         color=color, lw=2, label=label(r))
            axes[1].plot(days, d['per_variable_per_step']['2m_temperature']['improvement_pct_rmse'],
                         color=color, lw=2)
        axes[0].plot(days, ref['original_graphcast_loss']['improvement_pct_per_step'],
                     color='#555555', ls='--', lw=2, label='V22 reference SWA (2k-8k)')
        axes[1].plot(days, ref['per_variable_per_step']['2m_temperature']['improvement_pct_rmse'],
                     color='#555555', ls='--', lw=2)
        for ax, title, ylabel in zip(axes, ['Exact GraphCast loss', '2 m temperature RMSE'],
                                    ['GC loss reduction (%)', 'Temperature RMSE reduction (%)']):
            ax.axhline(0, color='black', ls=':', lw=1, label='Frozen GraphCast (0%)')
            ax.set_title(title, fontsize=13, weight='bold'); ax.set_xlabel('Forecast lead (days)')
            ax.set_ylabel(ylabel); ax.set_xlim(.25,10); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.065,.30),
                   ncol=2, fontsize=8.5, frameon=False)
        fig.text(.06,.16, 'SWA windows: '+detail, fontsize=8)
        fig.text(.06,.105,
                 'At each lead: 100 x (1 - model / baseline), using each evaluation\'s matched frozen baseline.\n'
                 'Left: exact GraphCast loss reduction. Right: temperature RMSE reduction. Negative values mean degradation.\n'
                 'These are per-lead percentages; averaging them does not give the exact rollout-loss reduction.', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        fig = page('Evaluation setup and remaining configurations', 'Details behind the table and rollout curves')
        baselines = [d['original_graphcast_loss']['baseline_per_step'][39] for d in payloads.values()]
        lines = [
            'Training',
            'Frozen GraphCast Small; residual spatial width 512; two stateful temporal layers.',
            'di16 uses BCG1; di64 uses BCG4. Legacy Haiku and Mamba1 denote initialization schemes.',
            'Closed-loop stop-gradient feedback; segment 96; BPTT 24; AR tail 20; all-24-step loss.',
            'AdamW: 10k updates, 200-update warmup, cosine decay to 0.1 x peak LR; seed 22.',
            '', 'Evaluation and selection',
            '2022 evaluation: 32 matched anchors; cold full-feedback rollout; zero residual state.',
            '40 six-hour leads = 10 days. The matrix reports day-10 GC loss reduction (%).',
            'Each run screens checkpoints 2k/4k/6k/8k/10k and SWA 2k-8k/4k-8k/6k-10k on 8 anchors.',
            'The best checkpoint and SWA by screen rollout-loss reduction are evaluated on 32 anchors.',
            'The curves select the lowest day-10-loss matched SWA within each initialization/di group.',
            f'Day-10 frozen-baseline GC loss varies across matrix evaluations: {min(baselines):.4f} to {max(baselines):.4f}.',
            'The plotted frozen baseline comes from the V22 reference evaluation.',
            '', f'Unfinished configurations ({24-len(complete)}/24)',
        ]
        for r in rows:
            if r not in complete:
                lines.append(f"{label(r)}: training {r['train_state']}; evaluation {r['eval_state']}")
        lines += ['', 'Validation and provenance',
                  'Checked completed sample counts, matching anchors, baseline checkpoint, evaluation mode,',
                  'residual state and resolution; exact GC rollout ratios agree with the saved losses.',
                  'Companion CSV includes day-10 GC loss, day-10 temperature RMSE and artifact identities.',
                  'Companion sources JSON stores input hashes and the Slurm snapshot. No jobs were launched.']
        fig.text(.06,.84,'\n'.join(lines), va='top', fontsize=10, linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)
    print(json.dumps({'pdf': str(args.output), 'complete_runs': len(complete),
                      'curve_runs': [r['run'] for r in selected]}))


if __name__ == '__main__':
    main()
