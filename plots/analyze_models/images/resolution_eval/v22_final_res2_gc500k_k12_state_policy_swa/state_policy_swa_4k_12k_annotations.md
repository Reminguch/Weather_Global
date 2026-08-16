# v22 res2 state-policy SWA comparison annotations

Figure: `state_policy_swa_4k_12k_allvars_and_2m.png`
Point data: `plots/analyze_models/data/resolution_eval/v22_final_res2_gc500k_k12_state_policy_swa/state_policy_swa_4k_12k.csv`

## Results

All-variable values are equal-variable RMSE reductions relative to the GC500k baseline. The whole-rollout value is the mean over all 40 leads.

| Training / evaluation | Whole rollout | Day 4 | Day 10 | 2 m RMSE day 4 | 2 m RMSE day 10 |
|---|---:|---:|---:|---:|---:|
| Carry / carry | 16.13% | 19.43% | 16.67% | 1.440 K | 2.744 K |
| Reset / reset | 14.81% | 18.70% | 13.60% | 1.467 K | 2.816 K |
| Carry / forced reset | 9.64% | 13.02% | 7.44% | 1.526 K | 2.936 K |

## Comparisons

- Carry evaluation benefit on the same carry-trained SWA: +6.49 pp whole, +6.41 pp day 4, +9.22 pp day 10.
- Carry/carry versus reset/reset: +1.32 pp whole, +0.73 pp day 4, +3.07 pp day 10.
- Reset-trained/reset-evaluated versus forcing reset on the carry SWA: +5.17 pp whole, +5.68 pp day 4, +6.16 pp day 10.

## Protocol and curve definitions

- v22 res2; SWA of checkpoints 4k, 6k, 8k, 10k, and 12k.
- Vanilla GC500k baseline; closed-full rollout; 32 matched cold anchors; 40 six-hour leads; zero initial temporal state.
- `Carry / carry`: stateful-trained SWA evaluated while carrying recurrent state.
- `Carry / forced reset`: the same stateful-trained SWA, but recurrent state is zeroed before every evaluation step (`rs_reset_every_step=true`).
- `Reset / reset`: reset-every-anchor-trained SWA evaluated with recurrent state zeroed before every step (`rs_reset_every_step=true`).
- Day 4 and day 10 correspond to leads 16 and 40, respectively.

## Source evaluations

- Carry / carry: `artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/eval/cold_full_zero/swa_step04000-12000.json`
- Carry / forced reset: `artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/di16_closed_sg_stateful_20k/eval/cold_full_reset_every_step_zero/swa_step04000-12000.json`
- Reset / reset: `artifacts/checkpoints/v22_final/res2_gc500k_k12_full_mamba_state_ablation_20k/di16_closed_sg_reset_every_anchor_20k/eval/cold_full_reset_every_step_zero/swa_step04000-12000.json`

## Comparability caveat

The carry-trained SWA is from the earlier v22 stateful execution, whereas the reset-trained SWA is from the newer explicit state-policy run. Their evaluation protocol is matched, but the training executions straddle the state-policy implementation rewrite. Interpret the carry/carry versus reset/reset gap as a run-level comparison, not a perfectly isolated training-policy ablation.
