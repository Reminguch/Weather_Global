# NeuralGCM residual experiments: test review, remaining problems, and revision plan

**Audit date: October 7, 2026.** This is a standalone review of saved experiment artifacts for discussion with Ilya. It updates the October 3 diagnosis with completed native K20 and lower-learning-rate runs. The weather-space architecture is still a proposal. This audit recomputes saved metrics on CPU; it does not rerun GPU training or claim that the proposed fixes have been implemented.

## 1. Findings that affect the next decision

The early experiments had confirmed physical-instability mechanisms and a validation error-handling defect. Subsequent constraints and stricter weather objectives addressed those specific issues, but K2 training continued to degrade five-day forecasts. Sliding-window weighting and a lower-learning-rate K2 control did not eliminate that degradation.

The newer **native K20** experiment changes the evidence. Training completed 2,000 optimizer updates. Its validation-selected checkpoint at update 1,100 improves all seven variable aggregates at 120 hours over pressure levels >30 hPa on 16 separate evaluation origins. Improvements range from **2.17% to 4.58%**. The latest available independent evaluation, update 1,620, also improves all seven aggregates, by **1.07% to 3.82%**. These results come from the existing native-state architecture, without implementing the proposed weather-space residual.

This is promising but incomplete. Some intermediate K20 checkpoints degrade severely, improvements do not hold at every lead or pressure level, and the final update-2,000 independent evaluation is missing. The evaluation watcher exited after a plotting-file error. Native K20 should remain a baseline for further work; the present evidence does not establish that native corrections are inherently ineffective or that moving the correction after the decoder will solve the remaining problems.

## 2. Experiment lineage and what each test established

| Experiment/test | Setup and observed result | Supported conclusion |
| --- | --- | --- |
| Early cached K1 pretraining, September 23–24 | Unconstrained native corrections; several runs failed during 20-step validation after epoch 5, before completing pretraining | The initial interface could produce unstable closed-loop states; validation failure also terminated the worker |
| Intervention tests on early checkpoints | Isolated pressure and divergence corrections; removed specific increment components | Direct pressure increments caused a second-step error jump; injected degree-zero divergence caused pressure drift in the tested cases |
| September 24 five-term K1/K2 runs | Four width128/256 runs reached 1,000 updates; objective fell about 96–98%, while common weather diagnostics worsened | A large reduction in the aggregate objective did not establish useful weather skill |
| September 24 K2→K20 transfer | Started from an already degraded K2 checkpoint; stopped after update 100 with a nonfinite loss/gradient exception | This specific transfer failed; it is different from the fresh K20 run reviewed below |
| Original adaptive-loss K2 campaign | 2,000 updates with upper/native objectives retained; all seven >30 hPa aggregates worsened at five days | Variable-group adaptive weighting alone did not solve long-range degradation |
| Strict-mask K2, fixed calibration and adaptive weighting | Both completed 2,000 updates; ≤30 hPa and native objectives removed; forecasts finite, but all seven five-day aggregates still worse | Removing these targets reduced some degradation but did not establish five-day gains |
| Strict-mask K2, lower LR | Peak LR 1e-4 and warmup 200; completed 2,000 updates; seven five-day aggregates still worse | This schedule change alone did not solve K2's long-range problem |
| Fresh native K20, adaptive weighting | Completed 2,000 updates; selected best at 1,100 and latest independently evaluated snapshot at 1,620 show seven positive >30 hPa five-day aggregates | Longer-horizon native training can produce useful checkpoints, with substantial checkpoint variability |
| GC reference audit | Recomputed existing GC K2/K22 evaluator files; reviewed v24 source and width128 report | The NGCM implementation did not reproduce the GC training conditions; this was a source/result audit, not a new GC training run |
| Proposed weather-space residual | Design document only | No implementation, smoke-test, or efficacy claim is available yet |

The historical intervention evidence and early failures are documented in [diagnostic0929.md](../../../diagnostic0929.md). For example, a pressure-only correction raised the following 12-hour loss from approximately 17,205 to 139,796 in an isolated test. Removing the newly injected degree-zero divergence eliminated the tested pressure drift. These interventions support specific mechanisms; they do not diagnose every later optimization failure.

The September 24 transfer had a five-day objective already 238.7 times the baseline before fine-tuning and 3,113.1 times the baseline after 100 updates. The exception combined nonfinite loss and gradient checks, so it does not identify the first failing variable or operation. These are objective ratios from that historical experiment, not physical RMSE ratios from the current campaign.

## 3. Actual native K20 setup and differences from K2

The backbone is frozen deterministic NeuralGCM at 2.8° resolution, including its encoder, learned physics, and decoder. The residual uses a fresh width128 Graph–Mamba branch, `d_inner=16`, mesh4, two processor steps and four recurrent layers, FP32, zero output head, and seed 22. Training uses 2015–2021 data, validation uses 2022, and 2023 remains reserved for the final test.

Each forecast origin is encoded once. The residual reads the current native state and known features, adds a constrained increment after each six-hour advance, and then decodes the corrected state. The next physical step consumes that corrected state with stop-gradient. Memory gradients span the episode. Future truth is used as a target, not as weather feedback. SST/sea-ice forcing uses the origin's lag24h snapshot, persisted through the episode.

| Setting | Strict-mask adaptive K2 | Fresh native K20 |
| --- | --- | --- |
| Supervised forecasts per origin | 6h and 12h | Every six hours through 120h |
| Memory BPTT | 2 steps | 20 steps |
| Memory reset | At each new origin | At each new origin |
| Batch size | 2 origins | 2 origins |
| Supervised forecasts per update | 4 | 40 |
| Updates / supervised forecast count | 2,000 / 8,000 | 2,000 / 80,000 |
| Initialization | Fresh zero head | Fresh zero head; no K2 transfer |
| Objective/controller source | Strict-mask data-only loss and EMA balancer | Byte-identical core objective/controller/trainer source |
| Selection origins | 8 validation origins | The same 8 origins |
| Selection forecast range | 6/12h | 6–120h |
| Selection cadence | Every 20 updates | Every 100 updates |
| Independent evaluation | 16 other origins | The same 16 other origins |

Both use Adam with beta1=0.9, beta2=0.95, epsilon=1e-6, without gradient clipping or weight decay. Peak LR is 2e-3 and warmup lasts 2,000 updates; decay starts around schedule step 15,000. The actual rates at updates 1,620 and 2,000 are approximately 0.001619 and 0.001999. The entire 2k budget remains in warmup.

The lower-LR K2 control changes peak LR to 1e-4 and warmup to 200, retaining the other Adam settings and the original delayed-decay definition. It therefore remains at 1e-4 after warmup during this 2k budget. It is not the GC AdamW/cosine preset.

**This is not a single-factor K comparison.** Longer valid windows change the eligible-origin list and its order. K20 uses newly sampled calibration origins and different fixed calibration coefficients. Its checkpoint-selection horizon changes, and each update uses ten times as many supervised forecasts. The representation, objective formula, and adaptive-control algorithm are unchanged, but the realized experiment is not otherwise identical. Configurations and completion records are archived under [native K20](evidence/native_k20/training/config.json) and [K2 lower LR](evidence/k2_low_lr/training/config.json).

## 4. Exact objective used by the current K20 experiment

For the seven decoded weather fields, the optimized scalar is

```text
L_train = Σ_v c_v · w_v · G_v
G_v = 20 M_v + 0.1 S_v + 2 B_v
```

- `M_v` is the normalized spherical-harmonic coefficient squared error, divided by sphere area and averaged over batch, lead, and selected levels. It includes the existing spectral filter.
- `S_v` compares spectral amplitudes: take the square root of summed coefficient squares over zonal wavenumber, compare prediction and target, and retain total wavenumbers 0–42.
- `B_v` first averages the difference of absolute modal coefficients over batch and time, then takes its squared norm and averages over levels. It is a coupled batch/time modal-amplitude bias term.

Only T, Z, u, v, q, cloud ice, and cloud liquid water at **p > 30 hPa** enter these terms. The eight excluded levels are sliced out before arithmetic. The 29 retained levels have equal reduction weights; there is no GC-style `p/mean(p)` pressure weighting. Native accuracy and native spectrum losses are disabled, although the model still applies corrections in native state. All 37 levels remain in the physical model and evaluation.

The normalization scale comes from **24-hour weather differences at 60 training timestamps**. Specific humidity is normalized per level; the other fields use scales pooled across levels. Modal values are multiplied by amplitude/scale before the loss, with amplitudes 2 for geopotential, 0.66 for specific humidity, 0.05 for each cloud field, and 1 otherwise. These factors act before squaring. Native input/correction normalization is a separate artifact: the output increment scale is based on encoded next truth minus the advanced native baseline, not these 24-hour loss statistics.

Errors are multiplied before squaring by lead-dependent factors, with `t` in hours:

```text
accuracy and bias: a(t) = (1 + t/24)^(-1/2)
spectrum:          s(t) = (1 + (t/40)^4)^(-1/2)
```

Thus every six-hour target participates, but later targets are downweighted. At 120h, the accuracy squared-error factor is 1/6 and the spectrum factor is 1/82. The existing fixed spectral filter is retained for K20; this is not a reproduction of the original NGCM fitted long-lead filter.

Fixed calibration uses the zero-residual rollout on 16 training origins, grouped into eight batches:

```text
c_v = prior_v / max(mean_calibration_group_loss_v, floor)
floor = max(1e-8, 0.01 × median(positive calibration group means))
```

The unnormalized priorities are 1 for each of T/Z/u/v/q and 0.05 for each cloud field, then normalized across the seven groups. Because calibration divides by each group's initial magnitude, amplitude constants alone do not describe the final relative contribution. The K20 calibration floor is 0.0049196 and no group mean is below it. Its actual fixed coefficients, in the above field order, are approximately `[0.417827, 0.291892, 0.312044, 0.398566, 0.261552, 1.483084, 1.657503]`.

The controller uses 100-update and 400-update EMAs of calibrated group losses and probes per-group parameter-gradient norms every 20 updates. At probe times, provisional weights are proportional to the square root of the relative fast/slow loss ratio, divided by the smoothed gradient norm. Projection enforces mean one for active groups, bounds 0.25–4, and a maximum 20% change per adjustment. Weights are external, stopped coefficients applied to the next batch. They do not use validation scores.

At update 1,620, the five main weather groups all have dynamic multiplier 0.25, while cloud ice and liquid have multipliers 1.952 and 3.798. At update 2,000 the cloud multipliers are 1.75 and 4.0. These observations show controller saturation. They do not establish that clouds dominate gradients or actual Adam updates, because the calibrated losses and gradient magnitudes differ.

One multiplier covers every retained level and lead for a field. It cannot independently correct a vertical imbalance or remove the existing long-lead downweighting. Large upper-level contributions to an unweighted physical RMSE summary also do not imply that upper-level gradients dominate the calibrated training objective.

Authoritative saved code: [weather objective](evidence/numerical_source/experimental/weather_only/objective.py), [base terms and temporal weights](evidence/numerical_source/src/models/neuralgcm_residual/paper_loss.py), [calibration](evidence/numerical_source/experimental/adaptive_loss/objective.py), [controller](evidence/numerical_source/experimental/adaptive_loss/controller.py), and [native forward/backward loop](evidence/numerical_source/src/models/neuralgcm_residual/trajectory_training.py).

## 5. Physical forecast results

All percentages below are `100 × (1 - model RMSE / frozen baseline RMSE)`. Positive means lower error. The evaluator first aggregates area-weighted squared errors across origins, then takes square roots; pressure-band aggregation likewise averages squared per-level RMSE before taking a square root. No adaptive training weights enter these scores.

The audit verifies identical 16 evaluation origins, pressure coordinates, and frozen-baseline RMSE arrays across the compared native experiments. These origins were not used for checkpoint selection in these runs, but some had been examined in earlier pilots. They are not an untouched final test set. All runs use one seed, and aggregate JSON does not provide the per-origin errors needed for paired uncertainty intervals.

### 5.1 Five-day forecasts on the 29 levels with p > 30 hPa

| Variable | K2 fixed calibration, 2000 | K2 adaptive, 2000 | K2 lower LR, 2000 | K20 selected best, 1100 | K20 latest evaluated, 1620 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temperature | -15.67% | -11.28% | -9.58% | +2.17% | +1.07% |
| Geopotential | -13.70% | -7.43% | -5.44% | +4.25% | +3.82% |
| Zonal wind | -5.14% | -3.63% | -3.67% | +3.57% | +2.88% |
| Meridional wind | -2.85% | -2.00% | -1.90% | +4.58% | +3.74% |
| Specific humidity | -5.34% | -4.84% | -4.46% | +2.80% | +1.51% |
| Cloud ice | -2.71% | -1.98% | -1.06% | +3.62% | +3.51% |
| Cloud liquid water | -1.27% | -1.04% | -0.68% | +2.71% | +3.23% |

The K20 best checkpoint was selected at update 1,100 using the eight selection origins and a physical score over all 6–120h leads. It was not chosen by maximizing this 16-origin, 120h table. The rule includes the zero-residual baseline, minimizes the mean of seven field RMSE ratios, and requires the worst aggregate field ratio to be at most 1.05. This guard is an average over selected levels/leads, not a bound on every individual forecast.

The K2 best checkpoint at update 1,840 also remains worse at five days: T/Z/u/v/q improvements are -4.86/-3.00/-1.68/-1.07/-3.23%, and the cloud fields are -0.83/-0.23%. Thus K2 degradation is not solely a final-checkpoint artifact. For the earlier adaptive scheme retaining upper/native objectives, update-2,000 temperature and geopotential improvements were -20.70% and -22.48%; strict masking reduced those regressions to -11.28% and -7.43%. Both masking and removal of native objectives changed together.

### 5.2 Short-lead and full-atmosphere limits

At 12 hours, using the same 16 origins and >30 hPa aggregate:

| Variable | K20 best, 1100 | K20 snapshot, 1620 |
| --- | ---: | ---: |
| Temperature | -0.05% | -0.11% |
| Geopotential | +1.58% | +0.65% |
| Zonal wind | -0.10% | -0.20% |
| Meridional wind | -0.08% | -0.30% |
| Specific humidity | +0.00% | -0.17% |
| Cloud ice | -0.05% | -0.27% |
| Cloud liquid water | -0.02% | -0.08% |

At 120 hours, using all 37 levels:

| Variable | K20 best, 1100 | K20 snapshot, 1620 |
| --- | ---: | ---: |
| Temperature | +3.90% | +1.43% |
| Geopotential | +6.76% | +6.91% |
| Zonal wind | -4.33% | -8.01% |
| Meridional wind | +1.90% | -0.27% |
| Specific humidity | +2.80% | +1.51% |
| Cloud ice | +3.60% | +3.50% |
| Cloud liquid water | +2.74% | +3.27% |

Consequently, seven positive >30 hPa five-day aggregates do not mean seven improvements at every lead or across the full atmosphere. The latest evaluated checkpoint still worsens full-column zonal wind by 8.01% and meridional wind by 0.27%. Turning off upper-level supervision does not freeze upper-level predictions. Rounded values near zero, such as humidity at 12h, are not evidence of a meaningful improvement. Where counts are shown elsewhere, ±0.1% is a display tolerance, not a significance test.

### 5.3 Checkpoint variability must remain visible

Every saved independent K20 evaluation is listed below, at 120h and p > 30 hPa:

| K20 update | Temperature | Geopotential | Zonal wind | Meridional wind | Specific humidity | Cloud ice | Cloud liquid water |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | +0.23% | +0.48% | +0.21% | +0.23% | +0.07% | +0.10% | +0.16% |
| 220 | -62.95% | -56.22% | -39.09% | -23.86% | -19.49% | -10.58% | -10.38% |
| 420 | -3.00% | +0.06% | -0.58% | -0.12% | -2.15% | +0.56% | -0.38% |
| 620 | -2.06% | +0.58% | -0.02% | +0.27% | -0.51% | +2.17% | +1.45% |
| 800 | -0.68% | +2.71% | +0.79% | +2.06% | -0.47% | +1.69% | +1.38% |
| 820 | -173.22% | -180.46% | -98.50% | -52.93% | -55.14% | -21.10% | -25.72% |
| 1000 | +0.63% | +2.76% | +3.09% | +4.42% | +0.60% | +2.62% | +0.74% |
| 1020 | +1.63% | +4.24% | +3.41% | +4.50% | +2.59% | +3.43% | +1.89% |
| 1100 | +2.17% | +4.25% | +3.57% | +4.58% | +2.80% | +3.62% | +2.71% |
| 1220 | +0.76% | +2.71% | +1.84% | +3.58% | +2.91% | +3.80% | +2.99% |
| 1420 | -9.04% | -6.23% | +0.90% | +3.22% | +1.00% | +3.39% | +2.54% |
| 1620 | +1.07% | +3.82% | +2.88% | +3.74% | +1.51% | +3.51% | +3.23% |

All these evaluations are finite. The severe update-820 regressions therefore reflect poor forecast skill rather than NaNs. Finite arithmetic and a short successful smoke test cannot rule out these training excursions. No single cause of the excursion has yet been isolated, and a few favorable checkpoints cannot establish reliable convergence.

## 6. Numerical and recovery tests already completed

These are archived GPU test reports, not new executions during this audit. They use independent pilot statistics, separate from production artifacts. H200 smoke routing was `ailab / gpu-test` with allocations of at most one hour.

| Test | Job | Recorded checks and result |
| --- | --- | --- |
| Strict-mask K2 numerical pilot | 14891022 | Passed: actual parameter change, frozen backbone, excluded-output invariance, exactly zero upper/native cotangents, dynamic JIT coefficients; fixed-gradient comparison error 0 |
| Strict-mask K2 resume | 14891023 | Passed: six uninterrupted updates versus interruption after update 3 and independent-process resume; exact state/controller history and recovery of inconsistent derived logs |
| Native K20 numerical pilot | 14937571 | Passed: genuine 20-step trajectory/tape, parameter update, mask and frozen-backbone checks; relative fixed-gradient comparison error 3.09e-7 |
| Native K20 resume | 14937572 | Passed: separate-process continuation, exact training/controller state and recovery checks |
| Lower-LR K2 schedule/resume | 14938454 | Passed under the actual 1e-4/200 schedule; inherited numerical kernel previously tested; six-update continuous/interrupted comparison |

The recovery comparisons cover parameters, Adam state, RNG, and controller state/history. They do not establish exact equality across arbitrary hardware, and native episodes reset memory at their origin. They therefore do not test the proposed 96-step cross-chunk memory carry. Six-update tests also cannot certify 2,000-update forecast skill or long-run optimization stability. Original reports are in [evidence/smoke](evidence/smoke).

The earlier GC audit found that historical K2 results mainly helped short leads, while K22 improved all eleven task variables at five and ten days in the saved evaluations. Those checkpoints had a different training history and budget from the current NGCM runs. The maintained v24 reference has 24-step BPTT and memory carry over 96-step segments; that configuration must not be retroactively assigned to every historical GC checkpoint. See the [GC/NGCM source and result audit](../gc_vs_ngcm_20261003/diagnostic.md).

## 7. Confirmed defects, unresolved mechanisms, and required changes

| Finding | Evidence status | Required response |
| --- | --- | --- |
| Unconstrained pressure/global-mean divergence corrections destabilized early rollouts | Confirmed by historical interventions in tested cases | Retain the implemented native increment constraints and regression tests; do not claim complete conservation or stability guarantees |
| Early validation exception terminated pretraining | Confirmed historical software defect; handling subsequently revised | Keep forecast failure separate from worker failure and checkpoint eligibility; retain failed-origin records |
| Aggregate objective improvement masked worse weather forecasts | Confirmed measurements | Use physical per-variable/per-level/lead evaluation for selection and reporting; keep separate full-column and target-band summaries |
| Field-group balancing has reached its bounds | Confirmed weights; causal effect on skill unresolved | Compare fixed versus adaptive objectives within K20 and log saturation, weighted group losses, gradient norms and directions |
| Entire high-LR 2k budget is warmup | Confirmed schedule mismatch with the intended bounded experiment | Test a schedule that warms up early and decays within the budget; keep other optimizer settings fixed for attribution |
| Native K20 has large intermediate regressions | Confirmed independent evaluations | Diagnose the affected updates and evaluate a K20 schedule control before treating the run as stable |
| K20 evaluation monitoring terminated before final coverage | Confirmed traceback and missing evaluation artifacts | Make rendering nonfatal to collection/scheduling, make plot publication robust, then evaluate final/best checkpoints |
| Decoder location, one-frame input, and short memory caused K2's poor skill | Differences confirmed; causal attribution unproven | Run controlled representation/input/memory ablations; retain the successful native K20 checkpoint as a baseline |
| Re-encoding in the proposed weather-space loop changes the baseline | Interface risk, not yet measured here | Compare original native rollout, zero-residual re-encoding wrapper, and trained weather residual before claiming improvement |

Both the maintained GC reference and the native NGCM implementation stop weather-feedback gradients while retaining memory gradients. The claim that GC succeeds because it differentiates through the entire weather rollout, whereas NGCM does not, is not supported by the inspected code. Differences in the physical interfaces still require controlled tests.

### The newly identified monitoring defect

The [watcher traceback](evidence/monitoring/watcher_error.log) records a `FileNotFoundError` while replacing a plot's shared temporary filename. The [watcher](evidence/monitoring/watch.py) calls the report subprocess with `check=True`, so this plotting exception propagates and exits the collector/scheduler. The [reporter](evidence/monitoring/report.py) uses a deterministic `.tmp.png` path and an advisory file lock. A concurrent writer or filesystem interaction is a plausible trigger, but the precise temporary-file disappearance has not been reproduced; the existing lock means concurrency must not be asserted as a proven root cause.

Training later reached 2,000 updates, as its [DONE record](evidence/native_k20/training/DONE.json) confirms. The watcher status remained stale and independent evaluation stopped at snapshot 1,620. No independent results for updates 1,820 or 2,000 are present in the audited directory. Internal eight-origin validation at update 2,000 exists, but it is not a substitute for the separate 16-origin evaluation. The recorded best selection score is 0.980246 at update 1,100, whereas the final score is 1.004162, on that internal aggregate metric.

The needed engineering change is to keep capture and evaluation scheduling alive after a rendering-only failure, record and retry that failure, use a unique temporary file in the destination directory with atomic replacement, and preserve a single-writer publication contract. A completion check must reconcile the final and selected-best checkpoints with validated result artifacts even if the watcher exits. Scientific-data validation errors must remain visible and must not be silently converted into successful results. Fault-injection tests should cover a renderer exception, a missing temporary file, and competing report attempts.

These repairs and the missing final evaluation are **proposed work, not completed fixes in this report**.

## 8. Revision plan and acceptance criteria

### Priority 0: complete and harden the existing evidence

Preserve the native K20 best checkpoint at update 1,100 and final checkpoint at update 2,000 with hashes. Repair the evaluation/reporting lifecycle described above and evaluate the final checkpoint on the same 16 origins. Reuse existing validated results rather than silently resubmitting them. Recover any additional snapshot only if its actual checkpoint exists; do not reconstruct missing historical checkpoints from a training log.

Report 6/12/24/72/120h, all seven fields and 37 levels, with >30 hPa and common-GC-level summaries. Completion means every required final/best result is present and validated, or explicitly recorded as a failed evaluation. Add per-origin errors in future evaluations so paired uncertainty intervals can be computed, then use additional origins/seeds for replication. Keep 2023 untouched until the evaluation protocol is fixed.

### Priority 1: stabilize native K20 with a controlled schedule experiment

The lower-LR K2 result does not answer whether native K20 would benefit from a different schedule. Run a native K20 schedule control with peak 1e-4, warmup 200, and decay to 1e-5 within 2,000 updates as a proposed starting preset. Retain the existing Adam beta/epsilon settings, correction constraints, objective, field schema, origin order, calibration artifact, and seed. This tests a joint schedule change, not the peak LR alone. Test gradient clipping separately before combining changes into a new default.

Log scheduled LR, actual parameter-update norm, per-field physical errors, and every eligible checkpoint. Require improvement to persist on the predefined selection/evaluation protocol and across replication, rather than accepting a lower training loss or one favorable snapshot. Report update count, supervised forecast count, and compute consumption; equal updates do not imply equal budgets across K2 and K20.

### Priority 2: separate normalization and adaptive-weighting questions

First compare fixed calibrated weighting against the existing adaptive controller within native K20 under otherwise identical settings. The current data do not establish whether adaptation improves K20. Record bound saturation and calibrated per-group gradients; measure gradient directions before claiming gradient conflict.

Then test per-variable/per-level normalization in the existing native architecture. Keep correction-output scales distinct from loss scales. A six-hour difference-statistics preset is proposed, with explicit handling of zero-variance cloud channels. Change loss normalization separately from adding pressure weights or removing spectrum/bias terms. These steps identify which objective changes help instead of assigning all gains to a bundled redesign.

Adaptive coefficients currently share a mean-one budget across all seven fields. If controlled tests confirm that this allocation is unhelpful, compare a controller with explicitly separated main-weather/cloud budgets or fixed cloud multipliers. These are hypotheses to evaluate; small cloud gradients and large cloud multipliers alone do not prove a harmful allocation.

### Priority 3: implement the GC-aligned weather-space architecture as a separate comparison

The [architecture README](../../../experimental/weather_space_residual/README.md) proposes direct residual addition after decoding, two weather-frame inputs, 24-step memory BPTT, and memory carry within 96-step segments. The weather inputs restart from a truth prefix at each 24-step chunk; the 96-step segment is not an uninterrupted free weather rollout. Native K20 has neither this two-frame interface nor this cross-chunk carry.

Start with the frozen re-encoding wrapper before training any residual. Evaluate three distinct systems:

1. Native NGCM, encoded once and advanced continuously.
2. A zero-residual weather wrapper that re-encodes every six hours.
3. The same wrapper with the learned weather-space residual.

The proposed initial engineering gate remains at most 1% degradation for any >30 hPa variable aggregate at 12h or 120h for the zero-residual wrapper relative to native NGCM. This is a proposed threshold, not a measured result or significance test. If the wrapper fails, investigate the interface before full training. Zero-head equality must hold against the wrapper, while multi-step equivalence to native NGCM must be measured rather than assumed.

Next validate two-frame causality, the output-level mask, current-step parameter gradients, weather stop-gradient, and 24-step memory gradients. Exercise a complete 96-step segment, its reset boundary, and exact recovery across chunk boundaries with independent pilot statistics. The GC `ar_tail_k=20` convention produces a 126h endpoint with a 24-step window and four truth-prefix calls; use explicit lead-hour presets to distinguish it from native K20's 120h endpoint.

Only after these checks should the new architecture run a full training comparison against native K20. Keep objective and schedule matched for architecture attribution. The README's bundled GC-style loss/AdamW preset is an additional system comparison, not proof that decoder placement alone caused a change. Two-frame, memory-gradient-length, and carry ablations require retraining under matched sequence and supervision conditions.

All future GPU smoke tests use `ailab / gpu-test` on H200, at most one hour per job, with pilot artifacts separate from production. Code, data, checkpoints, caches, and reports remain on scratch. This report does not launch those experiments.

## 9. Reproduction, source records, and limits

Run the CPU-only audit from the repository root:

```bash
python3 docs/experiments/ngcm_test_review_20261007/reproduce.py
```

It verifies **67 archived evidence files**, reads **32 saved evaluator checkpoints**, checks identical baseline arrays and origins, and regenerates **3,360 field/lead/band comparisons** in [comparison.csv](comparison.csv). It also verifies shared numerical-source hashes and the archived smoke-test pass records. [audit.json](audit.json) records the checks and all input hashes; [artifact_manifest.json](artifact_manifest.json) maps copied evidence to its original scratch path and SHA256. Local absolute paths inside historical JSON are provenance only; the script resolves copied baselines within the repository.

[Generated tables](tables.md) include every independently evaluated native K20 snapshot and the lower-LR control's data in the CSV. Earlier strict-mask and original-adaptive results are read from their existing published artifacts rather than duplicated. No checkpoints, environments, or training datasets are included in this report bundle.

The current conclusions are limited to saved finite forecasts from one seed and a small, previously examined validation set. Smoke-test success, decreasing adaptive loss, positive band averages, and architectural similarity to GC answer different questions. The immediate evidence supports retaining native K20, repairing evaluation coverage, and testing stabilization and architecture changes separately.
