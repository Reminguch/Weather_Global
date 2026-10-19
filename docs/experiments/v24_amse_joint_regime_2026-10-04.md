# AMSE joint RMSE and spectral-power campaign

The user authorized the full staged experiment on 2026-10-04. The scientific
target is simultaneous deterministic global RMSE and spectral-power-error
reduction, with Original GC and a trained MSE model scored separately as
references. Forecast fields are never averaged or blended.

Campaign root:
`artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004`.
Driver: `scripts/experiments/run_v24_amse_regime.py`.
Every submitted stage uses the campaign's complete frozen source snapshot,
pinned configurations, baseline/statistics/dataset metadata, and initialization
selection evidence. Existing experiment directories are not modified.

## Prespecified stages

1. **Existing trajectory evaluation.** Score native steps 500, 1000, 1500 and
   2000 for the MSE, Full AMSE and AMSE λ=0.1 trajectories, plus their relevant
   SWA checkpoints. Reuse compatible 2022 exact32 evaluations. Recompute the
   eight-start 2023 spectra to obtain per-initialization physical RMSE and
   spherical diagnostics alongside the existing zonal diagnostics.
2. **Gentle continuation pilot.** Select mature MSE initialization using only
   the saved 32 matched 2022 starts, comparing both campaigns' native step2000
   and prespecified SWA. Branch into exact grid MSE, spectral λ=0, AMSE λ=0.25,
   and AMSE λ=0.5. Each arm resets optimizer, RNG, cursor and recurrent state;
   uses the same batch4 data stream/seed22; trains 500 updates; and saves every
   100 updates. Spatial/Mamba peak learning rates are 1e-5/3e-5, with cosine
   decay to spatial 1e-6. Positive λ rises linearly from zero over 100 completed
   updates. A zero-based index100 is the first update at target λ.
3. **Conditional differentiable feedback ablation.** If the best AMSE recipe
   does not qualify, or loses more than0.001 mean coherence against mature MSE,
   compare stopped and differentiable physical feedback from the same selected
   checkpoint. Both follow 2→4→8-step windows, 100 updates per window, batch4,
   with identical resets and learning rates. The original GC parameters remain
   frozen; derivatives through its weather inputs are enabled in the new arm.
   Separate actual1° GPU preflights for each window must pass before release.
4. **Targeted objective ablations.** From the best development candidate,
   compare 500-update uniform continuation, precipitation protected with
   spectral MSE (λ multiplier0), and a fixed spherical-degree λ taper. The
   latter uses global spherical coherence from the eight development starts:
   multiplier0.2 at coherence≤0.3, increasing linearly to1 at≥0.7; the global
   mean retains spectral MSE. All other GraphCast variable/level normalization
   and weights are preserved. Score updates100/300/500.
5. **Locked larger-budget confirmation.** Select one positive-λ recipe from
   development evidence before reading confirmation results. Train that recipe
   and a matched exact-grid-MSE control from the common mature initialization,
   with seeds22/23/24 and 2000 updates each. Short-window recipes first use
   matched 100-update windows2 and4, then2000 updates at8. This tests the chosen
   objective/horizon at a larger budget; it does not reproduce the selected
   short-run checkpoint's complete warm-start history. Evaluate the fixed final
   checkpoint on64 predeclared starts. Never select a checkpoint or modify a
   recipe using those confirmation scores.

If no candidate qualifies, the bounded confirmation is explicitly an
exploratory test of the best observed AMSE recipe, not a claim of improvement.
Successful Slurm completion is distinct from scientific success.

## Evaluation and selection

The exploratory sample remains the existing eight 2023 starts. All primary
metrics use those same starts: exact original GraphCast loss, cosine-weighted
global physical RMSE, and midlatitude zonal power/coherence. The 32 matched
2022 cases are a separately labeled diagnostic. Nine primitive fields are
scored at days1/3/7/10: T2m, U10, V10, MSLP, TP6, T850, Q700, U850 and V850.
Derived wind-speed power is available, but is not mislabeled as global
wind-speed RMSE.

Qualification requires lower aggregate exact GC loss, lower mean feature/lead
RMSE ratio, and lower mean relative power error than Original GC, plus no more
than2% RMSE degradation in **any** of those nine fields at any selected lead.
Each feature/lead's joint win against GC and MSE is retained explicitly.
Aggregate qualification never means that every feature improves. Candidate
selection uses the Pareto frontier and joint-win counts; the controller filters
to positive-λ AMSE recipes before choosing a follow-up.

Global spherical diagnostics use the same area-weighted QR T179 transform as
the training objective. Degree0 is retained in files; anomaly-spectrum summaries
exclude it explicitly. The report separates continuous fitted spherical MSE,
discrete projected grid MSE, and unresolved grid error. Synthesis of error
fields is performed after subtracting truth to avoid large-offset pressure
cancellation. A projection closure check must pass.

The64 confirmation dates contain16 per meteorological season. Each is more
than10 days from the exploratory initializations and has a complete10-day
forecast within2023. They are excluded from this campaign's selection; other
2023 studies may already have inspected some dates. Forecast windows overlap
within confirmation, so uncertainty uses paired14-day calendar blocks within
seasons. Adjacent block boundaries can still share forecast weather; these
intervals are approximate, not independent-case significance tests. The three
seeds are separate deterministic forecasts. Summaries average their metrics,
never their fields, and retain individual-seed results and matched MSE controls.

## Execution, safeguards and resources

The four original-shape32-update preflights request256GiB and90min. Historical
64-update preflights14380046_0/1 peaked at151.8/155.4GiB and completed in47min.
The new2/4/8-step input-gradient preflights request320GiB/90min because their
full-resolution backward memory is unmeasured. Their failure does not prevent
the original stopped-feedback pilot from running.

Completed2000-update training jobs14147422/24 and14397797/99 took15h39–15h50;
their largest host-memory peak was234.22GiB. Training requests use the larger
of that history and the new measured preflight peak, multiplied by1.35 and
rounded to16GiB. Time uses measured p90 update duration, compilation allowance,
and a1.35 margin. Typical500-update requests are approximately320GiB/6h;
actual requests are recorded after preflight. GPU memory is separate: every
GPU task requests an80GB device and records telemetry.

Screening evaluates both exact32 and spectral8 at each declared checkpoint;
this deliberately retains cross-year diagnostics at additional evaluation cost.
Each screening evaluation requests16GiB/150min. Confirmation64 evaluations
request16GiB/8h, subject to the actual per-stage evidence. All arrays are
unthrottled; Slurm determines concurrency. There are no automatic retries,
unbounded searches, or modifications to other live jobs. Conditional phases
are limited to two feedback arms, three objective ablations, and one confirmed
recipe with three seeds and three matched MSE controls.

CPU tests cover dynamic λ forward/reverse gradients, configuration compatibility,
weather-input gradients (including a small real GraphCast network), frozen
baseline parameters, projected-error closure, matched evaluation provenance,
confirmation separation, and duplicate-submission guards. GPU smoke outcomes
and job IDs live under `checks/`, `submissions/`, and `logs/` in the campaign.

## Outputs and monitoring

- `status.json`: latest campaign stage and final completion evidence.
- `initialization_selection.json`:2022-only initialization choice and all inputs.
- `confirmation_dates.json`: fixed64-start protocol, written before training.
- `<phase>/evaluation/report/joint_selection.json`: machine-readable scores,
  feature/lead comparisons, uncertainty and conditional decisions.
- `<phase>/evaluation/report/README.md`: readable stage report.
- `plots/analyze_models/images/resolution_eval/res1_amse_joint_regime_20261004/<phase>/`:
  joint RMSE/power scatter and feature heatmaps, PNG/PDF plus plotted tables.

Submitted on2026-10-04; Slurm confirmed the following requests and dependencies:

| Stage | Job | Initial state |
|---|---:|---|
| Existing15-checkpoint evaluation |15004676_0–14| Pending, priority |
| Existing-trajectory report |15004677| Pending, afterok15004676 |
| Four32-update ordinary preflights |15004678_0–3| Pending, priority |
| Three differentiable-feedback preflights |15004679_4–6| Pending, priority |
| Gated pilot release |15004680| Pending, afterok15004678 |

The source digest is
`273630baa1c325f5f26deee5b6e4315bb6c01bf122f23f6a9eed3cab1bec2308`.
The selected initialization is the September24 MSE SWA checkpoint, with2022
exact32 loss9.12019047. All64 confirmation dates' two input times and40 target
times were verified against the actual dataset time coordinates.

Validation completed before submission:79 core AMSE/feedback/spherical/evaluation
tests, followed by50 campaign/evaluation integration tests (these sets overlap);
89 existing endpoint/configuration regressions also passed. Python compilation,
shell syntax, and whitespace checks passed. The new physical-feedback GPU
runtime/memory checks remain pending; no new scientific gain is claimed yet.

Monitor with:

```bash
squeue -j 15004676,15004677,15004678,15004679,15004680
cat artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004/status.json
```

Later job IDs are added automatically under the campaign's `submissions/`.
