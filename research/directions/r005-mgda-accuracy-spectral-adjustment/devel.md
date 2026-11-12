# Development — MSE / AMSE-adjustment MGDA

Updated: 2026-11-12 (America/New_York)

## Portable handoff for Lianghong — 2026-11-12

The user requested exact runnable code for another cluster and explicitly chose
**both studies**: five fresh M1-recipe GC–Mamba lambda arms and four fresh
standard-loss full-GC K8/K12/K16/K20 arms. Use the
[portable README](../../../docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md),
not the older Della horizon launcher that reused K20.

Package: `artifacts/handoffs/lianghong_weather_20261112/`. The code archive copies
every computational file from the submitted GC–Mamba source
`a6a1c4e7ae895eb788227924146b3da84e5542bb10b7f5e294ebbe15943abd7f`
byte-for-byte and adds portable training/comparison/smoothing adapters. A separate
weights archive contains public GC and historical M1 raw/SWA. Canonical anchors,
statistics, cohort manifests, source hashes and a runtime lock are included.
The 238.35-GiB prepared training store and evaluation datasets are external;
their source locators/inventory and the scope of hash checks are explicit.

Adapters generate new local paths/manifests/snapshots. GC–Mamba reuses the existing
500-reset/main750-exact-resume implementation. Full GC preserves the old lambda-0
recipe and validates all 2,500 data-order replay entries after relocating only
source paths. Both have two allocations of 1,250 updates. Local wrappers preserve
the activated Python environment; no account/partition/job IDs are hardcoded.
No jobs were submitted, cancelled or changed for this handoff, and no remote
environment or GPU execution is claimed as tested.

The shared evaluator handles all 34 new development models plus M1 raw/SWA,
retains spectra and exact MSE, selects full-GC finalists only on 2022, and evaluates
the common 122-start 2023 cohort with paired date-block uncertainty. Portable
filter/strength adapters reuse existing computation for any finalist. Full-GC
versus GC–Mamba retains the documented recipe difference. Local truth revisions
and runtime changes must be recorded rather than relabeled as identical inputs.

All 31 packaged CPU tests passed. Both actual training campaigns prepared
successfully with relocated anchors/statistics/reference weights, validating ten
GC–Mamba stage configs, four full-GC configs and the full shuffled replay. Actual
M1 smoothing preparation also passed with the fixed 32/122 cohorts. The package
source digest is `8b363120bf494948770ecfa90c460391057600401edd4690eed23d3e70d01f1e`.
Evidence is recorded in the handoff's `validation/` directory. The existing Della production
chains below remain unchanged; Lianghong's future remote run/job IDs are unknown.

## GC–Mamba-only M1-recipe restart submitted — 2026-11-12

The user approved the reviewed restart **only for GC–Mamba**, explicitly keeping
full GC/DeepMind unchanged. The earlier ten-arm review remains historical in
[e012](experiments.md#e012) and the [protocol](../../../docs/experiments/v24_m1_matched_lambda_restart_2026-11-12.md).
No full-GC stopped-feedback adaptation was activated.

New campaign: `artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112/`.
Five arms at lambda0/.25/.5/.75/1 use fresh public spatial overlay and seed22.
Initial500: constant LR1e-4 after50-update ramp, Mamba multiplier1. Main2000:
cosine1e-4→1e-5 after50-update ramp, Mamba multiplier3. Parameters-only transfer
at500 resets optimizer/RNG/cursor/recurrent state. Historical contiguous segment
order replaces the shuffled manifest. Lambda is applied in both stages.

Native lambda0 retains `loss_kind=mse`; positive lambda uses
`native_mse_amse_excess`, adding the existing FP32 spectral excess to the native
BF16 residual-MSE calculation. Weather gradients stop, memory gradients remain.
Step-zero checkpoints, metadata and resumed state are recorded explicitly.

Submitted jobs: first-part array **17406861_0–4** performs initial500+main750;
second-part array **17406862_0–4** uses `aftercorr:17406861` and exactly resumes
main750→2000. There is no reset at total1,250. History **17406863_0–4** evaluates
total500/1000/1500/2000; finish **17406864** follows the second part and launches
development fitting/annual smoothing through `run_v24_m1_matched_downstream.py`.
Training requests one80GB GPU,8CPUs,384GiB and30hours per part; history20GiB/4h,
finish8GiB/30min. Initial states are submitted/pending, with no new results.

Manifest hash:
`f75dbbdec1e1af4147e822bfe8a639e4775589d1abebdf26fbb1b13b29ce739c`.
Frozen training source:
`a6a1c4e7ae895eb788227924146b3da84e5542bb10b7f5e294ebbe15943abd7f`.
Validation passed116 native-loss/reset/resume tests,8 campaign tests and18 mixed
comparison/SWA/annual-report tests. All ten initial/main configs validated.

Cancelled only the old GC–Mamba production chain17062358/17062359/17062360 and
obsolete joint final chain17353172/17353173/17353174. Protected full-GC
17049163/17049164/17049165 and diagnostic17404694/17404732 remain unchanged.
All completed old checkpoints, evaluations and frozen sources are preserved.

Mixed final campaign: `artifacts/evaluations/v24_mixed_final_comparison_20261112/`.
Materialization **17406897** waits for `17049163:17406862`;20-model forecast array
**17406898_2–21** and report **17406899** follow. It reuses M1 raw/SWA forecasts
and the original pilot computational source. GC–Mamba SWA uses main500/1000/1500/2000;
full GC uses1000/1500/2000/2500, matching total-update positions. Future weights
are hashed only after actual completion. The plan hash is
`f6d00da5679d73c3285435ba0d71a6dddd800c956b19e00b35a8da1d05ff4f10`;
its separate frozen evaluator digest is
`8a5f9c9a8b02995e333de37a68d626ea5aaf940294b9c8a20ea21e7fa1935150`.

Annual smoothing uses actual main-step2000 checkpoints, never renamed step2500
files. A dependency-based join controller waits for actual submission records
from new GC–Mamba and unchanged full GC, then writes the combined annual report
under `artifacts/evaluations/v24_m1matched_gc_lambda_annual_20261112/`. These
downstream jobs are launched by finish/release hooks; their future IDs are not
invented here. Comparison metadata preserves the differing training recipes.
Inspect completion records before claiming results; finite warnings remain advisory.

## Matched intermediate checkpoints submitted — 2026-11-12

Campaign `artifacts/evaluations/v24_matched_budget_comparison_20261112/` uses
evaluation17404694_2–5 and dependent report17404732. New raw forecasts are M1
main500/750 (total1000/1250), November GC–Mamba lambda0 step1000 and full-GC
lambda0 step1000. Reuse both November1250 forecasts and M1 final raw/SWA, giving
eight model rows plus public GC. November1500 does not exist yet. Existing
training/final2500 submissions were unchanged at this diagnostic's launch;
the later GC–Mamba-only replacement is recorded above. The diagnostic and
four-horizon hold remain unchanged.

`prepare_v24_matched_budget_comparison.py` pins actual serialized checkpoint
steps, family, baseline and resolved config, then inherits the completed pilot
computation. Manifest hash
`2ee6bb4c9e351ea6f03c24e2393bdb76b14079d6d29e1523d99177c9f14aed1f`;
source digest `04e82e3820ecb955e7ecd5a21fed1b91c58a252a6b443a36ef58c7d03b595d3d`.
Four focused preparation tests and15 report/reuse tests passed. Only
orchestration/report files are overlaid. Reporter now emits the full-GC lambda
learning-rate caveat only when both zero and positive weights are included.
Resources: four tasks at16GiB/62min/one80GB GPU; report4GiB/15min. Inspect
`evaluation/*/complete.json` and the final report before claiming results.
[Protocol, setup audit and existing500-update evidence](../../../docs/experiments/v24_matched_budget_comparison_2026-11-12.md).

## Historical final2500 raw/SWA submission — 2026-11-11

Superseded on November12: jobs17353172/17353173/17353174 were cancelled and
replaced by the mixed final chain above. Preserve this plan and its completed
M1-only outputs as historical evidence; do not reactivate it against new weights.

The user requested2500 rather than the interim1250 pilot, with both MSE and
power spectra and with/without SWA. Existing ten scalar-lambda continuations
remain queued: fullGC17049163_0–4 and GC–Mamba17062358_0–4. New CPU
materialization17353172 depends on both arrays;20-model evaluation17353173
(indices2–21) and report17353174 follow. No training restart or extra updates.
M1 raw/SWA forecasts are reused; the separate four-horizon training hold stays.

Campaign: `artifacts/evaluations/v24_final_checkpoint_comparison_20261111/`.
`prepare_v24_final_checkpoint_comparison.py` seals current identities in
`plan.json`, then validates final2500 proofs and pins actual future weights
before creating `manifest.json`. SWA uses equal weights at1000/1500/2000/2500,
zero recurrent state and the existing checked helper for both model families.
Frozen evaluator inherits the completed Nov10 pilot; only orchestration and
reporting are overlaid. Source digest:
`90ccacb212e0ba60c7e9e428f94cda3480c8086e371c58120a79546a6471df52`.
Eight driver tests and15 paired-report/reuse tests passed; all3 Slurm test-only
requests passed. Resources: CPU8GiB/30min; GPU16GiB/62min; report4GiB/15min.

[Protocol and outputs](../../../docs/experiments/v24_final_checkpoint_comparison_2026-11-11.md).
The separate completed M1-only report reuses original4date/40lead forecasts;
the ten fresh2500 results remain pending. Reporter uses paired raw/SWA colors
and line styles, pooled exact MSE, physical spherical powers and explicit
within-run changes. Finite noise, worsening and stalls remain advisory.
Do not refresh this snapshot or substitute midpoint weights for final ones.

## Completed raw-checkpoint pilot — inspected2026-11-11

Both arrays17249273/17250938 and reports17249274/17250939 completed.
Sealed records/hashes and4date/40lead coverage verified; M1 reference outputs
are reused identically. Results and figures are in both campaign-named
resolution_eval data/images directories; [protocol/report summary](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md)
and e011 completion record give the interpretation. M1 SWA improves both MSE
and aggregate power error versus raw. Fresh1250 models remain worse on MSE;
historical Octoberλ0.25 improves both versus raw M1, but not versus SWA.
Do not generalize the older8-date zonal spectral conclusion to these spherical
scores. Finite warnings remain advisory; no new jobs or runtime changes.
Observed14m44s–15m46s/task, maximum8.75GiB RSS, total5.064GPUh; CPU reports
49/52s. Existing16GiB/62min allocation is adequate for this4date shape.

## Submitted raw M1 / lambda evaluation pilot — 2026-11-10

Follow-up requests both lambda sets. Historical extension17250938→17250939
adds eight raw checkpoints and reuses M1 tasks17249273_0/1. Campaign:
`artifacts/evaluations/v24_legacy_raw_checkpoint_pilot_20261110/`;
digest `5c714cfb5e73c34f5c43a32ce93c3cfcff5892a46b070c4810b01e27cb9061c5`.
`prepare_v24_legacy_raw_pilot.py` inherits the original frozen computation and
overlays only three orchestration/report files. The reporter verifies reused
parent source/cohort/checkpoint identity and splits legacy training histories.
15 driver/preparation plus6 report-reuse tests passed. Original jobs/snapshot
are unchanged. The completed cached non-SWA report is already at
`plots/analyze_models/data/resolution_eval/amse_lambda_raw_cached_20261110/report.md`;
its eight2023-start zonal figures are distinct from the four2022-start spherical
pilot. See the e011 scope clarification for results and interpretation.

Array17249273 evaluates M1 raw main-step2000, its historical SWA, and the ten
fresh scalarλ raw1250 checkpoints on four quarterly2022 starts ×40 leads.
CPU report17249274 follows automatically. Both initially pending. This is
authorized saved-checkpoint evaluation; the separate horizon training hold
remains. [Protocol](../../../docs/experiments/v24_raw_checkpoint_pilot_2026-11-10.md)
and [e011](experiments.md#e011) record checkpoint choices, existing M1 loss,
resource evidence and the GCFT λ0/positiveλ learning-rate difference.

Campaign: `artifacts/evaluations/v24_raw_checkpoint_pilot_20261110/`.
Frozen runtime digest:
`8203142838c3c60027df2f816a15ab11cb1da0cfb1e57f2e23ec2fe09d1b5324`.
`run_v24_raw_checkpoint_pilot.py` reuses the existing evaluator and streamed
spherical observer with model-specific metadata. `report_v24_raw_checkpoint_pilot.py`
writes exact MSE curves, power spectra/ratios, tradeoff and CSVs under the
matching resolution_eval data/images directories. All83 channels/40leads
remain in per-date/pooled NPZ. Twelve focused tests and synthetic reporting
passed. Do not refresh the snapshot or overwrite failed attempt outputs.

Monitor `squeue -j 17249273,17249274`; inspect `evaluation/*/complete.json`
before claiming completion. Four starts and intermediate1250 weights give
diagnostic results only; finite loss/power deterioration remains advisory.

## Submitted smoothing-strength sweep — 2026-11-09

User authorization covers this evaluation; the separate full-GC horizon
proposal remains on hold. Campaign:
`artifacts/evaluations/v24_smoothing_sweep_m1_20261109/`.
Evaluation array17213596 has8 unthrottled shards (16,16,15,15,15,15,15,15 starts),
each1×80GB GPU/4CPU/20GiB/180min. CPU report17213597 depends on the whole array.
Both were pending at the initial queue check. Check `squeue -j17213596,17213597`
and then the campaign's `checks/evaluate_*.json` and `report.json`; submission
is not completion evidence.

`run_v24_smoothing_sweep.py` extends the verified parent snapshot with five
explicit files. Parent forecasting code and independently development-fitted
filters are preserved. Source digest:
`22d780d97df6119954c3ca1f4349cf107ecdca26892ae271bd71f9f63780f5e9`.
The original worker gained only an optional observer-class argument.
`SmoothingSweepObserver` saves E0/q/Q for exact native-grid loss and physical
forecast/truth/delta powers and cross-moments for11 bases. `report_v24_smoothing_sweep.py`
pools date moments before ratios, reconstructs30 points/model, saves per-date
exact losses, and renders loss/power tradeoff and day10 strength curves.
It also records correlation/error-power summaries and repeated endpoint drift
against the previous annual report; all finite differences remain advisory.

All15 focused tests passed (observer4, report4, campaign2, existing campaign5).
Synthetic plot layouts were inspected; they are not evaluation results. No
new GPU qualification was needed: completed parent array17028677 supplies
resource evidence. See [protocol and resume commands](../../../docs/experiments/v24_smoothing_strength_sweep_2026-11-09.md)
and [e010](experiments.md#e010). Do not rerun preparation into the same directory
or alter its snapshot. Refine α from saved moments rather than rerunning
forecasts once the campaign completes.

## Prepared full-GC horizon comparison — 2026-11-09

The user requested K8/K12/K16/K20,2500 updates, exact/spectral evaluations
every500, and SWA. The [plan](../../../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md)
and [e009](experiments.md#e009) describe the fixed24-step window and the resulting
16/12/8/4 teacher prefixes. K20 reuses the existing `3e-7` trajectory at1250;
three new configurations share its fixed data order, seed and optimizer recipe.
The new campaign is prepared at
`artifacts/checkpoints/v24_Ilya/res1_full_gc_ar_k8_k12_k16_k20_20261109` with
source digest `77f264914960a3bb471a8588b3594ad9dbe836d6267da8b6f76432bfb27903d3`.
No new jobs are submitted. At the user's request the draft is on hold pending
the [reviewer-comparison audit](../../../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md#reviewer-comparison-audit--before-submission).
The sealed plan covers two training parts, development spectra, SWA and report,
but omits M1 and matched2023 evaluation. A revised implementation must not
modify the already frozen snapshot in place. Preserve the existing λ campaign.

The first500-update K20 score is now available: rollout10.337334 versus public
GC10.840716, a4.6434% reduction across32 development starts; day10 reduction
5.3090%. Those are intermediate development results. Focused tests and
`sbatch --test-only` passed; a real K20 checkpoint SWA was checked on CPU.

## Pending GCFT λ0 redirected to fresh LR3e-7 — 2026-11-07 evening

User authorized preserving pending training job17049162_0 while replacing its
500-update continuation with a fresh public-GC run. Active mapping:
`res1_fresh_gc_mamba_gc_lambda_20261107/selected_arms_full_gc.json`, task0 →
`gcft_s22_l000_lr3em7_fresh_20261107`, `resume_step=0`, peak LR3e-7, end LR3e-8,
warmup100, fixed2500-update schedule. Training IDs17049162_0→17049163_0 and their
resource requests/submission age are preserved; endpoints remain1250/2500.
The frozen source digest and original manifest/initial configs are unchanged.

Revision folder: [20261107_gcft_l000_fresh_lr3em7](../../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7).
It contains the original/new sealed selections, config/helper hashes, validation,
activation and scheduler evidence. Do not regenerate the old `select` stage:
the user-directed provisional production override supersedes its λ0 selection.
The four positive-λ arms remain1e-5; this is not a fixed-LR λ-only comparison.
Old500-update calibrations are retained as evidence, not resume parents.

Step500 development evaluation17062379 waits for17049162_0 and uses the frozen
evaluator through the revision's pinned helper. It writes the fresh run's
`calibration_evaluation/<new_id>/score.json`. History17049164_0 now requires
both17049163_0 and17062379, so the frozen λ0 score-reuse branch has genuine
matching evidence. Final controller17049165 and its smoothing dispatch read
the active mapping. Additional lower-LR pilots were discussed but not submitted
as part of this narrow authorization. Check the current queue rather than
treating its estimated start as a reservation.

Full-GC eight-update profiles and both500-update calibrations have completed,
demonstrating the one80GB configuration fits. LR1e-5/3e-5 exact dev32 loss
increases were13.8958%/52.2477%. Finite worsening remains advisory; the fresh
start is an explicit user revision. Focused dispatch/config/source checks passed.

## Initial fresh scalar λ / full-GC / smoothing launch — 2026-11-07

User explicitly authorized all three workstreams. The [plan](../../../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md)
and [e008](experiments.md#e008) are authoritative for this new comparison;
older MGDA campaigns below are separate and remain untouched.

- Training: `artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/`.
  Initial array17022621; full-GC controller17022622; Mamba controller17022623.
  Indices0/1/4 are full GC,2/3/5 are fresh GC–Mamba. All initially queued.
- Smoothing reference: `artifacts/evaluations/v24_smoothing_m1_20261107/`.
  Retained first development start17022624, controller17022625; initially queued.
  This fits/scores public GC and principal M1 in the same forecast pass.
- Frozen source: training-root`source/`, digest
  `ab39c404b062a55f2ee93883672442973fa4a2351699529ac11492a1a6b88475`.
  Do not update this snapshot from the workspace after submission.

`run_v24_scalar_lambda.py` implements independent per-family profiling,
500-update λ0 LR calibration, development selection, five2,500-update trajectories,
intermediate evaluation and per-family final smoothing dispatch. Eight-update
profiles are retained; positive λ0.5 profile reuse requires the selected LR to
match. `selected_arms_<family>.json` maps logical λ IDs to actual run directories;
read that mapping rather than assuming the final directory is the logical ID.
Production endpoints are1250/2500; the LR schedule remains2500 throughout.
No trained Mamba checkpoint initializes a fresh arm. LoRA need remains unknown
until full-GC GPU profiling finishes; no automated method/batch/horizon substitution.

`run_v24_smoothing.py` implements first-start measurement → remaining development
shards → frozen fit → first annual start → remaining annual shards → report.
The aggregate report waits for all ten new models and the principal M1 reference.
It computes paired exact-MSE block intervals; spectral coordinates remain
descriptive with per-date powers retained. Baseline, mean, projection and finite
optimizer warnings preserve measured values/reference bounds rather than stopping.

Checkpoint payloads carry explicit `model_family`; full-GC parameters occupy the
legacy `residual_params` storage slot but are interpreted as full predictions,
with an empty recurrent state. Evaluation defaults to family auto-detection.
New scalar config keys are `model_family`, objective`loss_kind=gc_mse_amse_excess`,
objective`spectral_excess_weight`, and data`train_order_manifest`.
The shared data fingerprint includes the frozen training-order manifest hash.
Native step-zero snapshots, optimizer/RNG/cursor/state and immutable per-stage
metric copies support reproducible continuation. Resume checks the live history
against the previous stage snapshot before appending.

Monitor:

```bash
squeue -j 17022621,17022622,17022623,17022624,17022625
```

Read each root's`submissions/` for automatically generated later job IDs,
`checks/` for completed stage/resource proofs, and`logs/` for actual GPU startup.
Future full-GC feasibility is recorded in `checks/profile_summary_full_gc.json`;
do not treat submission or CPU tests as evidence of an80GB fit. Each native
run writes`checkpoints/checkpoint_step00000000.pkl` before its first update.
No restart is needed while jobs remain valid/queued. If an actual execution
error occurs, diagnose it and retain checkpoints/proofs; never refresh a frozen
runtime or silently repeat an already completed stage.

## Authorized continuation and fresh training — 2026-11-02

The user approved both strategies after discussing initialization. The new
driver is `scripts/experiments/run_v24_mgda_initialization.py`; it creates
separate frozen campaigns and preserves the completed e004 artifacts.

- e005: `artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_continuation_two_parts_20261102/`.
  Extend both e004 MGDA and matched MSE-control checkpoints from200 to2000 total
  updates, retaining parameters, paired-RMS state, RNG, recurrent memory and
  cursor. LR remains1e-6. Two endpoints:1100/2000 (900 new updates each).
- e006: `artifacts/checkpoints/v24_Ilya/res1_mgda_full_bptt_fresh_two_parts_20261102/`.
  Train MGDA and matched MSE control for2500 updates from DeepMind spatial
  overlays plus the best setup's Mamba1 initialization, seed22 and zero output
  heads. No learned residual checkpoint or MSE-only pretraining is loaded.
  Both use paired RMS,100-update warm-up from1e-6 to1e-5, cosine decay to3e-6
  at update2000, then constant. Two endpoints:1250/2500 (1250 updates each).

Both keep the current exact-GC/AMSE-excess objectives, full weather BPTT24/K20,
segment96, batch4, architecture and BF16 compute. The old relative-power loss
is not substituted. These compare two training strategies; differing rates and
training histories prevent interpreting them as an initialization-only ablation.
Final raw checkpoints are the endpoints; no SWA, fixed-sum arm, further seed,
automatic retry or extension is included.

Shared frozen source digest:
`39f6cadbdf213f007497d796e483f9a61c772d04931606082426a2d7aa527d3b`.
The proven e004 runtime is preserved except the new driver/wrapper and allowing
output-location changes during exact resume. Copied checkpoints remain under
their new run's `checkpoints/` directory; original tensors and histories are
preserved by hashes.104 focused CPU cases passed, including a bitwise comparison
of a relocated resumed trajectory against uninterrupted training. Frozen
continuation preparation validated both real update200 checkpoint configs.

Resources use completed full-feedback jobs16402493_0/1: maximum steady p90
137.356s/update, historical host-RSS floor234.71GiB, startup allowance1664.78s.
Both parts request384GiB,8 CPUs and one80GB GPU. Each continuation part uses
2940min (49h); each fresh part4050min (67.5h), within gpu-medium's72h limit. `aftercorr`
dependencies keep MGDA and MSE training independent between matching array
indices. Final exact/spectral evaluations use16GiB/62min; reports12GiB/30min.
The existing verified best-MSE reference evaluations are reused. New final
forecasts use the same eight2022 origins and40 leads. Plain-GC and trained-MSE
references are labeled separately. Finite variance, drift, increases and stalls
remain warnings, with no resource/numerical qualification rerun.

The user requested only two parts to reduce repeated queue waits. All original
five/seven-stage jobs and their descendants were cancelled while pending at
16:00:13EST on November2; no training progress was discarded. Their snapshots,
configs and submission records remain preserved in the original roots.
e005 replacements: training arrays16673752→16673753; exact16673754,
spectral16673755, report16673756.
e006 replacements: training arrays16673757→16673758; exact16673759,
spectral16673760, report16673761. Monitor with `squeue -j 16673752,16673757`.
The nine focused campaign cases passed for the two-part schedule; the prior
numerical/resume evidence is reused. Both changed resource requests passed
Slurm's test-only check before actual submission.
Inspect each campaign's
`submissions/`, `checks/*_phase*.json` and `complete.json`. Actual execution and
forecast improvement remain pending until those artifacts establish them.

## Current implementation

The `mse_amse_excess` pair supports shared full weather-feedback BPTT with
`feedback_mode=closed_loop`, including24-step windows and a four-step teacher
prefix. Both gradients use one forward trajectory and independent sequential
reverse sweeps through recurrent memory, both overlapping weather inputs and
the frozen GraphCast input Jacobian. Scalar full-feedback, stopped-feedback,
`amse_split` and GC/power modes remain supported. The three direction modes
share the pair second-moment estimator; exact GC MSE remains the primary loss.

The October28 user policy is saved in [AGENTS.md](../../../AGENTS.md): finish
bounded finite training and judge skill in final evaluation. Both
`repeatability_policy=warn` and `decrease_policy=warn` are explicit in the new
campaign and are maintained defaults. Do not restore strict per-update loss
decrease, finite-direction, finite-stall or variance gates without user request.

The helper evaluates one candidate at the scheduled LR and commits it if finite,
including when losses increase or tie. Candidate end memory and one proposed
RMS update are committed together. `backtracking_trials.jsonl` retains the
historical filename but records one trial per advisory step, including raw
repeats, `measured_decrease`, `decrease_warning`, noise margins and above-noise
diagnostics. `descent_valid` and `stalled` remain truthful advisory flags.
The campaign validator/report permit these warnings and retain counts.
Explicit `error` policy exists only for historical strict behavior and tests.
Nonfinite gradients/moments/parameters/losses/state and integrity failures remain
blocking. Exact resume still trims histories to the checkpoint boundary.

<a id="full-feedback-campaign-ready--2026-10-28"></a>
## Full-feedback campaign completed — inspected 2026-11-02

Implementation and230 focused CPU cases are complete and passing; the
[validation record](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/checks/implementation_validation.json)
lists the tested files and their hashes. The frozen
[e004 campaign](experiments.md#e004) completed on October29 at
`artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/`.
Source digest:
`b71969ce37d1cf47b231d93719e24a21b01d01503717fe785a19baf6499cfd13`.
At preparation, all changed runtime files matched the workspace implementation. Only the required
historical prepared-array loader is restored; its provenance is recorded.
The user approved submission on October28. Observation16399850 completed at
19:59:51EDT after38m58s and release16399851 completed at20:01:11EDT. Training
array16402493 completed all three200-update arms on October29; MGDA retained
its first eight observed updates and resumed for192. Exact16402494 and
spectral16402495 evaluation arrays completed, followed by report16402496 at
04:41:00EDT on October29. Completion proofs and the
[final report](../../../artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_full_bptt24_20261028/evaluation/report/pilot_report.json)
were inspected on November2. The existing e003 snapshot is untouched.

The prepared contract is24/K20/96, batch4, original parent, seed22, LR1e-6 and
unchanged predictor precision. Both losses supervise every step equally.
The `observe` stage executed eight actual MGDA updates with
`max_steps=200 --stop-after-steps 8` and saved the step-eight checkpoint plus
immutable histories/configuration. Its allocation was one80GiB GPU,8 CPUs,
320GiB host RAM and240min. Observed timing and memory selected320GiB/690min for
production, retaining the historical234.71GiB long-run host-memory floor.
Both controls started fresh for200 updates. Telemetry, measured resources and
all200-update proofs are preserved; no horizon or effective-batch reduction
was needed.

Independent toy derivatives cover both objectives at24/prefix4, memory,
overlapping weather edges, frozen baseline weights and active input derivatives.
Boundary tests cover detached memory carry and segment reset; actual-runner
tests check one RNG/cursor/optimizer advance and exact resumed equality.
Candidate replay, all three directions, finite-increase/variance warnings,
nonfinite rejection, primary-MSE validation and feedback-mode resume mismatch
are covered. Campaign tests cover observation sealing, continuation after live
history growth, endpoint8/200 validation, tamper rejection and the dependency
graph. These are implementation checks, not forecast-skill or production GPU
capacity evidence.

On eight2022 origins and40 six-hour leads, MGDA exact GC loss is0.4275% higher
than the parent, while secondary mean spectral power error falls from44.5751%
to43.2507%. MSE control changes exact loss by−0.0499%; fixed sum by+0.2412%.
This is a spectral/accuracy tradeoff, not a demonstrated joint win. The small
single-seed development cohort supports no significance or generalization
claim. See [the result record](experiments.md#e004-completion-20261102).
Finite warnings remained advisory and every arm finished without zero updates.

No resubmission or automatic extension is needed or authorized. Preserve
`checks/`, `resources/production.json`, `submissions/` and the completed report;
do not refresh the frozen snapshot. The stopped-feedback preparation default
remains unchanged.

## Earlier advisory validation

The previous failure and cancelled evaluations remain documented in
[e002](experiments.md#2026-10-23--recovery-outcome-inspected). The new behavior
passes122 targeted cases after a fixture correction; all38 campaign cases
passed on rerun. Original failed-step loss records can now commit with an
explicit no-decrease warning. Scientific evaluation is still pending.

## Earlier e003 handoff — status recorded 2026-10-28

Current root: `artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_advisory_20261028/`.
Release16391359 was submitted October28 at16:11EDT and completed in6s.
MGDA16391367_0 is running; controls16391367_1/2 remain pending priority.
Exact evaluation16391368, spectral evaluation16391369 and report16391370
await dependencies.
Frozen source digest:
`e0751783bf520e26a555a23c9748d58aed5a2bc0dcd331930dfd8aabf580e700`.
Inspect its `submissions/`, `checks/`, `runs/*/train_metrics.jsonl` and
`evaluation/report/pilot_report.json`. Preserve earlier snapshots and failures.
No prior production checkpoint exists, so this campaign restarts from the
original prescribed parent rather than claiming to resume the46/1/31 updates.

No larger continuation is authorized by this pilot. Review exact GC loss
against the original parent and both controls, and power/coherence diagnostics,
before proposing the next scientific step.

Preparation uses `prepare --recovery-from <20261019-recovery> --reuse-qualification
--experiment <fresh-root>`. Only four declared runtime files are overlaid onto
the old frozen source. The prior successful qualification15877833 is pinned and
reused as historical resource evidence only. Submission starts a CPU release
job directly, which launches all three200-step arms and dependent evaluations;
there is no new GPU qualification job or claim of current numerical proof.
Training retains320GiB/390min per arm; run metadata records why.
Never refresh a submitted snapshot.
The historical evaluation store is Zarr v3 with `zarr.json`; restricted sandbox
access stalled on its external filesystem, while approved access worked. Future
submission-host verification needs the same external-filesystem/Slurm access.
The pilot snapshot and scientific data were not altered to work around this.

## Failure-analysis handoff

See the [October28 detailed diagnosis](experiments.md#diagnosis-20261028).
`scripts/analyze_models/diagnose_v24_mgda_failure.py` reads saved e002 losses and
metrics, writes a hashed JSON record and line-search figure, and runs a small
CPU toy to distinguish stopped-feedback AD from full-feedback AD. It does not
replay production data or modify jobs. Outputs live in
`artifacts/analysis/v24_Ilya/mgda_failure_20261028/`.

Two changes were identified outside e003: enable FP32 aggregation for GraphCast
mesh processing/decoding (both predictors currently BF16), and extend weather
feedback to both objectives. The second is now implemented for e004; precision
changes remain a separate follow-up. At diagnosis the full-weather backend was
scalar and limited to2/4/8 steps, and MGDA required stopped feedback. The local
MGDA coefficient audit agrees within9.11e-8 and shows no recorded zero update.
Keep finite diagnostics advisory; preserve the running source snapshot.

## Planned shared full-BPTT implementation — 2026-10-28

Status: implementation and e004 execution are complete. The final report
completed October29 and was inspected November2. The following preserves the
design rationale.
Before this change both objectives shared one24-step trajectory but stopped
the weather-feedback derivative; the new optional mode differentiates it.

### Recommended sequence contract

| Setting | First full-feedback comparison |
| --- | --- |
| BPTT window | 24 six-hour prediction steps; identical for both objectives |
| Autoregressive tail | `ar_tail_k=20`, retaining four teacher-input steps |
| Segment | 96 steps, four nonoverlapping 24-step update windows |
| Recurrent state | Carry values between windows; detach gradients at every window boundary; reset state at segment boundaries |
| Weather | Reinitialize from observed teacher frames at each window; preserve gradients through every generated weather edge within that window |
| Supervision | Both objectives at all 24 steps, equal weights 1/24; same batch, keys, forecast and state trajectory |
| Effective batch | Four independent lanes, as in the existing pilot |
| Update | One paired-RMS/MGDA update per window; one optimizer-state advance |
| Comparison | Same parent, seed22, LR1e-6, objective definitions and three200-step arms |

This is six days of sequence per gradient window and24 days of sequence per
segment, not a24-day free-running forecast. A96-step segment carries memory
across four weather restarts. `truth_prefix_steps=24-20=4` means five observed
input frames for four overlapping teacher windows. The fourth prediction
starts the generated weather chain; the remaining20 predictions use generated
inputs. Do not silently switch K20 to K23: K23 would instead make a fully
autoregressive24-prediction forecast from two initial observed frames and is a
separate training-distribution change. Both losses use the same actual graph
and truncation boundaries, including teacher-frame constants.

Choose24 initially to isolate the gradient-path change and preserve the memory
horizon, sample exposure and optimizer cadence of e003. A shorter8-step pilot
would also change the training distribution and produce12 updates per96-step
segment instead of four. Use short CPU toy unrolls for implementation tests,
not an automatic sequence of new2/4/8 training campaigns. Do not extend BPTT
to96 merely because memory values carry across a96-step segment.

### Implementation order

1. **Generalize the existing weather tape.** In
   `src/models/mamba/v24_Ilya/training/weather_feedback_step.py`, separate the
   forward/reverse engine from its scalar optimizer adapter. Support scalar
   or two-component FP32 losses, the24-step window and teacher prefixes1/4
   using the same scheduling rules as the existing stopped-feedback engine.
   At each step retain host weather/state boundaries; rematerialize one-step
   kernels in reverse. Do not compile the entire24-step weather graph into
   one training executable.
2. **Connect two losses to the same graph.** Use `full_field_objectives` for
   exact GC MSE plus spectral excess. Do not substitute scalar AMSE merely
   because the config says `loss_kind=amse`. From one reference trajectory,
   run two sequential reverse sweeps with loss seeds `(weight,0)` and
   `(0,weight)`. Each sweep owns separate parameter, recurrent-state and
   weather cotangents. Accumulate contributions into both overlapping input
   weather frames. Freeze GraphCast parameters, while including its input
   Jacobian. Initial observed frames, targets and prescribed forcings remain
   constants. Teacher-prefix predictions before the generated-feedback
   boundary must not receive weather cotangents from observed frames that
   happen to have the same time index. Recurrent-state derivatives still
   propagate through all24 steps.
3. **Reuse the MGDA update and commit policy.** In `training/mgda_step.py`,
   dispatch full-feedback training to the shared engine while retaining the
   existing raw losses, pair RMS metric, direction modes and one finite
   scheduled update. Candidate replay uses the same full-feedback forward
   semantics and identical starting memory/inputs/keys. Commit candidate
   memory and proposed optimizer state once. Keep finite increases, drift,
   stalls and variance advisory. The diagnostic repeats must not create extra
   optimizer updates or independently sampled objective trajectories.
4. **Wire configuration, validation and resume metadata.** Update
   `training/config.py` to permit the new MGDA `closed_loop` combination and
   explicitly validate prefix/window/segment compatibility. Update
   `training/endpoint_step.py` dispatch and memory accounting for two reverse
   sweeps and weather cotangents, and retain scalar/legacy behavior. Review
   forward-only calibration and validation dispatch: exact GC MSE must stay
   the primary logged loss rather than silently becoming scalar AMSE.
   Record feedback mode/backend in the resolved config/checkpoint and report.
   Treat changing feedback mode as a new warm-start treatment from declared
   parent weights, with fresh optimizer/RNG/cursor/state; exact resume remains
   reserved for the same full-feedback configuration.
5. **Integrate a fresh matched campaign.** Prepare new frozen metadata using
   the existing pilot conventions. MGDA, MSE-control and fixed-sum all use
   the same full-feedback graph and paired second moments. Keep current e003
   running to its final evaluation. Preserve parent/data/eight2022 origins
   and40 six-hour evaluation leads. Compare exact GC loss first, then spectral
   power/coherence. BF16-to-FP32 predictor/aggregation changes are separate
   from this gradient experiment so their effects can be distinguished.

### Focused validation

Extend existing fixtures rather than add another qualification ladder:

- Independent direct-AD comparison for both objectives on short deterministic
  FP32 toy rollouts, including two input-weather edges, a nonconstant baseline,
  recurrent state and both prefix1/prefix4. Confirm baseline parameter gradients
  remain zero while its input-gradient contribution is present; include one
  small24-step toy case and a finite-difference directional spot check.
- One boundary/commit test across adjacent windows and a segment reset: equal
  forward forecasts before changing parameters, detached history between
  windows, state carry/reset as declared, candidate state replay, and one
  optimizer/RNG/cursor advance for the two gradients. Check resumed same-mode
  metadata and rejection of a mismatched exact resume.
- Reuse warning/nonfinite and control-arm tests for the new dispatch: a finite
  worsening update completes, invalid arithmetic remains an error, and MSE/
  spectral values and diagnostic flags remain correctly labeled. Preserve
  scalar full-feedback and stopped-feedback regressions.

Deterministic toy derivative checks validate implementation. Production finite
repeat differences or loss increases remain warnings, not release conditions.

### Resource plan and completion

Existing scalar full-feedback width512/batch4 runs from
`res1_amse_joint_regime_20261004` completed100 updates at horizons2/4/8.
For8 steps, the saved completion proof records19.28s p90 and76.35GiB peak host
RSS. These provide real implementation evidence but not a budget proof for
24-step dual-loss reverse sweeps or the current parent. The two sequential
sweeps reuse one host tape and keep one sweep's weather cotangents active at a
time; GraphCast input backprop increases time and one-step GPU memory.

After implementation, measure a single short run at the intended24/K20/batch4
shape, covering two96-step segments (eight updates) and saving the step-eight
checkpoint. Count committed updates toward the200-update budget
and resume exactly; avoid discarding a series of qualifier runs. Record
steady latency, GPU peak and host RSS alongside the existing longer-run host
memory evidence before choosing final Slurm resources. No resource quantities
or jobs are approved/submitted by this planning note.

If batch4 cannot fit one-step input VJPs, first consider sequential microbatches
with effective batch4, averaging both objective gradients before a single RMS
and MGDA update. This is extra implementation scope only if measured necessary;
do not silently shorten BPTT for one objective or one control arm. Retain the
same gradient clipping and boundary semantics across all arms.

Implementation is complete when the new graph supports the declared24/K20/96
contract, the focused tests pass, metadata and exact resume describe it
correctly, and the bounded campaign is ready to run. Forecast improvement is
an experimental outcome, not an implementation acceptance condition.
