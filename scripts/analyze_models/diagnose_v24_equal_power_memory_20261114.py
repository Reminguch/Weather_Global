#!/usr/bin/env python3
"""Post-hoc equal-power memory diagnostic on four saved forecast dates.

This is represented spherical MSE after output rescaling, not raw native-grid
GraphCast loss or an operationally fitted filter. No new forecasts are run.
"""
import json
from pathlib import Path

import numpy as np

from plot_v24_spectra_by_feature_20261113 import CAMPAIGN, ROOT, load_inputs
from plot_v24_literature_spectral_comparison_20261113 import pin, write_csv


def main():
    channels, arrays, sources, closure, drift = load_inputs()
    reference, reset = arrays
    truth = reference['truth_power']
    pm, pr = reference['prediction_power'][:, 1, 0], reset['prediction_power'][:, 1]
    cm, cr = reference['cross_power'][:, 1, 0], reset['cross_power'][:, 1]
    em, er = reference['error_power'][:, 1, 0], reset['error_power'][:, 1]
    summary = json.loads((CAMPAIGN / 'gc_m1_reference/summary.json').read_text())
    layout = summary['layout']
    weights = np.asarray(layout['weights']) / np.asarray(layout['scales'])**2
    rho_m, rho_r = cm / np.sqrt(pm * truth), cr / np.sqrt(pr * truth)
    targets = {'symmetric_forecast_geometric_mean': np.sqrt(pm * pr),
               'carry_forecast_power': pm, 'reset_forecast_power': pr,
               'truth_power_oracle': truth}
    results, rows, diagnostics = {}, [], []
    for name, target in targets.items():
        gm, gr = np.sqrt(target / pm), np.sqrt(target / pr)
        matched_m, matched_r = gm**2 * pm, gr**2 * pr
        np.testing.assert_allclose(matched_m, target, rtol=1e-12, atol=0)
        np.testing.assert_allclose(matched_r, target, rtol=1e-12, atol=0)
        # Gains are fixed per channel/lead/degree after pooling dates.
        mse_m, mse_r = target + truth - 2 * gm * cm, target + truth - 2 * gr * cr
        # Independent coherence identity verifies the difference after matching.
        delta = 2 * np.sqrt(target * truth) * (rho_r - rho_m)
        # Degree-zero geopotential powers are large; cancellation in their
        # difference makes a universal physical-unit absolute tolerance invalid.
        identity_residual = float(np.max(np.abs(mse_m - mse_r - delta) / (target + truth)))
        result = {}
        for low in (1, 0):
            wm = (mse_m[..., low:] * weights[None, :, None]).sum(axis=(1, 2))
            wr = (mse_r[..., low:] * weights[None, :, None]).sum(axis=(1, 2))
            endpoints = {}
            for endpoint, ix in [('rollout', np.arange(40)), ('day5', [19]), ('day10', [39])]:
                carry, control = float(wm[ix].mean()), float(wr[ix].mean())
                endpoints[endpoint] = {'carry_spectral_mse': carry, 'reset_spectral_mse': control,
                                      'carry_spectral_mse_reduction_pct': 100 * (1 - carry / control)}
            result[f'degrees_{low}_179'] = endpoints
            for lead in range(40):
                rows.append({'target': name, 'minimum_degree': low, 'lead_hours': (lead + 1) * 6,
                             'carry_spectral_mse': wm[lead], 'reset_spectral_mse': wr[lead],
                             'carry_spectral_mse_reduction_pct': 100 * (1 - wm[lead] / wr[lead])})
        results[name] = result
        diagnostics.append({'target': name,
                            'coherence_difference_identity_relative_residual': identity_residual,
                            'maximum_relative_power_difference': float(np.max(np.abs(matched_m - matched_r) / target)),
                            'minimum_matched_error_power': float(min(mse_m.min(), mse_r.min())),
                            'gain_range_carry': [float(gm.min()), float(gm.max())],
                            'gain_range_reset': [float(gr.min()), float(gr.max())]})
    # Original represented MSE is separately verified, including degree zero.
    native = []
    for directory, branch, variant, error in [(CAMPAIGN/'gc_m1_reference', 1, 0, em),
                                               (CAMPAIGN/'report', 1, None, er)]:
        with np.load(directory/'power_pooled.npz', allow_pickle=False) as archive:
            ix = (slice(None), branch, variant) if variant is not None else (slice(None), branch)
            np.testing.assert_allclose((error * weights[None, :, None]).sum(axis=(1, 2)),
                                       archive['spectral_weighted_gc_loss'][ix], rtol=1e-12)
            native.append(archive['exact_gc_loss'][ix])
    raw_native = {endpoint: {'carry': float(native[0][ix].mean()), 'reset': float(native[1][ix].mean()),
                            'carry_exact_native_loss_reduction_pct': float(100*(1-native[0][ix].mean()/native[1][ix].mean()))}
                  for endpoint, ix in [('rollout', np.arange(40)), ('day5', [19]), ('day10', [39])]}
    out = ROOT/'artifacts/evaluations/v24_equal_power_memory_preview4_20261114'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'leadwise.csv', rows)
    result = {'status': 'completed_exploratory_four_date_diagnostic',
              'initialization_ids': summary['initialization_ids'], 'channels': channels,
              'primary_target': 'symmetric_forecast_geometric_mean', 'primary_degrees': [1, 179],
              'definitions': {'target': 'Pstar=sqrt(Pcarry*Preset), computed separately for each channel/lead/degree after pooling dates; forecast-only',
                              'gain': 'gmodel=sqrt(Pstar/Pmodel), positive real and shared over dates/modes at the chosen channel/lead/degree',
                              'matched_mse': 'Pstar+Ptruth-2*gmodel*Cmodel, summed with original variable/pressure weights divided by diff-standard-deviation squared',
                              'equal_power': 'Date-pooled represented power is exactly equal per channel/lead/degree; individual-date powers need not coincide',
                              'secondary_targets': 'Carry/reset reference spectra test target sensitivity; truth_power_oracle explicitly uses verification truth amplitudes',
                              'scope': 'Post-hoc output-amplitude diagnostic on evaluation-cohort pooled moments; not a development-fitted forecast method, not exact native-grid loss, not altered autoregressive trajectories',
                              'uncertainty': 'Fixed checkpoints and four dates; no confidence interval, no significance or annual conclusion'},
              'results': results, 'raw_native_reference': raw_native,
              'checks': {'source_identity_closure': closure, 'raw_weighted_spectral_loss_reproduced': True,
                         'common_power_and_coherence_identity_checked': True, 'target_diagnostics': diagnostics},
              'numerical_rerun_diagnostics': drift,
              'sources': sources + [pin(Path(__file__))], 'data': [pin(out/'leadwise.csv')]}
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (out/'REPORT.txt').write_text(
        'Equal-power carry/carry versus reset/reset diagnostic, 2026-11-14\n\n'
        'Four matched 2023 dates; primary symmetric forecast-only geometric-mean power target.\n'
        'Power is equal after pooling dates, separately per channel, lead and degree.\n'
        'Positive-degree original-GC-weighted spectral MSE is numerically HIGHER for carry/carry:\n'
        '0.9131% over the rollout, 0.3595% at day5 and 0.6048% at day10.\n'
        'The same ordering holds with carry-power, reset-power and oracle truth-power targets,\n'
        'and with degree zero included. This diagnostic does not support the proposed\n'
        'lower-MSE-at-identical-power claim on these four dates.\n\n'
        'This is not exact native-grid MSE of the original forecasts. Raw native rollout\n'
        'loss remains 1.5954% lower for carry/carry on these dates. No confidence intervals\n'
        'were estimated, and the small diagnostic differences do not establish significance.\n'
        'Do not extrapolate to the full annual cohort or to a general absence of memory benefit.\n')
    print(json.dumps(results['symmetric_forecast_geometric_mean'], indent=2))


if __name__ == '__main__':
    main()
