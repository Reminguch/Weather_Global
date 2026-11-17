# Matched reset/reset spectral addition to the literature-style figures

Date: 2026-11-13 (America/New_York).

**Current status, inspected November 15:** all 122 annual starts are complete,
including the original 92 and the 30-date extension. The validated annual
report is
[`annual122_report/summary.json`](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/summary.json).
The earlier dated sections describe historical preparation and interim states;
see the completion update at the end for the current handoff.

## Requested comparison

Add the separately reset-trained/reset-evaluated control to all seven figures
under `plots/analyze_models/images/resolution_eval/v24_literature_spectral_comparison_20261113/`.
The question is whether persistent memory changes the accuracy/amplitude/coherence
tradeoff, not whether Mamba outperforms another recurrent architecture. A better
tradeoff is a hypothesis until the matched spectral measurements are available.

Use `R_matched`, the SWA of main updates 500/1000/1500/2000 after the matched
500-update warm-up. Checkpoint:
`artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl`.
SHA256: `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`.
Do not substitute the historical unmatched `R_reset`, inference-reset `M1_reset`,
or a newer raw seed checkpoint.

## Existing and missing evidence

The completed `artifacts/evaluations/v24_matched_reset_2023_20261018/`
campaign contains exact loss for all 122 dates but no forecast fields or spectral
moments. Its saved losses can be restricted exactly to the current 92 dates.
The loss figure now includes this control; the other six figures await spectra.
Scalar MSE cannot supply amplitude, coherence, power spectra or represented
spectral RMSE.

The new campaign is `artifacts/evaluations/v24_matched_reset_spectra_20261113/`.
It evaluates the existing checkpoint with no training or filter fitting. Use
exactly the ordered 92 initialization IDs from the current optimal-filter
comparison, January 1 through October 1, 2023, and all forty six-hour leads.
This remains an incomplete annual cohort; the additional thirty dates are not
silently included in any model's average.

## Evaluation and comparison

Preserve the computational source from the completed optimal-filter campaign.
Overlay only the raw observer, campaign driver/wrapper and plotting script.
Explicitly set `reset_state_every_step=True`, zero initial memory, `cold_full`,
zero warm-up, residual alpha one and full corrected autoregressive feedback.
Require the evaluator to confirm its reset policy in the result.

The observer uses the same `OptimalSphericalFilter` and `spectral_moments` as
the existing GC/M1 figures: physical fields, 83 channels, native 181x360 grid,
degree 0 through 179. Save prediction, truth, cross and error powers per date
and lead for raw GC and raw reset/reset. Pool complete forecasts on the same
92 dates before forming amplitude ratios or coherence. Check layout, truth,
checkpoint/source identity and the spectral error identity. Record finite
forecast rerun differences; they are not rejection gates.

Save exact native-grid GraphCast loss separately from represented spectral
RMSE. Until spectral completion, the loss figure uses the archived reset
evaluation on the same dates. The completed report replaces it with the loss
from the spectral rerun, preserving archived-versus-rerun differences.

Only **raw** reset/reset is requested. Do not manufacture an optimal-reset
filter or apply the GC filter to it. In the two-column amplitude/coherence
figure, raw reset is an explicitly labelled reference in both columns.
Use purple consistently, retain the complementary blue/teal GC coherence
dashes, and retain all existing leads and variables. No arrows or markers
are added to the focused amplitude/coherence figure.

## Resources and completion

Eight unthrottled shards contain eleven or twelve dates each. Request one
A100-80GB PCIe GPU, four CPUs, 16 GiB host RAM and 62 minutes per shard.
Completed raw-moment calibration jobs `17430308_0..3` used eight dates each,
22:00–23:06 elapsed and approximately 9.6–9.8 GiB MaxRSS. Completed full-filter
jobs `17430310_0..5` used fifteen or sixteen dates, 37:15–39:14 and approximately
9.6–10.0 GiB. Raw-only observation omits synthesis/filtering; this request leaves
headroom without another qualification job. A CPU report follows successful
completion and regenerates all seven PNG/PDF pairs automatically.

Submitted November 13: evaluation array **17494525**, followed by report and
plot refresh **17494526** (`afterok:17494525`). The first queue check found
all eight forecast shards pending for priority and the report waiting on its
dependency. No new spectral result is available at this checkpoint.

Frozen source verification and CPU observer/configuration checks passed:
known native weighted losses, spectral closure, unchanged input fields,
complete ordered samples, all eight reset configurations and advisory finite
drift handling. Evidence is `checks/cpu_validation.json` within the campaign.
These are implementation checks, not completed GPU forecasts.

The campaign's submission records are the source of job IDs. Until the report
is complete, `RESET_STATUS.txt` beside the figures and `reset_control` in their
data summary identify which additions remain pending. Do not claim a spectral
tradeoff result from the loss-only addition.

## Implementation

- `scripts/experiments/run_v24_matched_reset_spectra.py`
- `scripts/experiments/v24_matched_reset_spectra.slurm`
- `src/analysis/raw_spectral_observer.py`
- `scripts/analyze_models/plot_v24_literature_spectral_comparison_20261113.py`

Related presentation plan:
[v24_spectral_presentation_plan_2026-11-13.txt](../reports/v24_spectral_presentation_plan_2026-11-13.txt).

## Faster preview requested later on November 13

The user requested a short GPU-test run and explicitly set its walltime to
**59 minutes**. Prepare a separate four-date preview on January 1, April 1,
July 3 and October 1, 2023. These calendar dates are fixed before examining the
reset spectra and are all present in the existing 92-date GC/M1 statistics.
Pool every comparator on exactly those four dates. Request `gpu-test`, one
`gpu80&pcie` GPU, four CPUs and 16 GiB; omit an explicit partition so the
cluster routes the job. Estimated execution is approximately 15–20 minutes
after starting, but the queue delay is unknown and the allocation is 59 minutes.

The preview is `artifacts/evaluations/v24_matched_reset_preview4_20261113/`.
Its seven figures go to the separate
`plots/analyze_models/images/resolution_eval/v24_literature_spectral_preview4_20261113/`
folder and are labelled preliminary. The plotting command requires explicit
`--preview` and a distinct output name, preventing replacement of the 92-date
figure set. Run forecast collection and rendering in the same short allocation
to avoid waiting for a second report job. The full 92-date campaign remains
scheduled; the preview does not establish an annual tradeoff conclusion.

Preview job **17495398** was submitted with `--qos=gpu-test --time=59`.
The scheduler confirmed partition `gputest`, QoS `gpu-test`, walltime `59:00`.
Its initial state was pending. Preparation verification is saved in
`artifacts/evaluations/v24_matched_reset_preview4_20261113/checks/preparation_verification.json`.
The campaign hash is
`ee2b37c9aa1880d8882ddae4b6f5d37cfe802a66c0f36e557f362de4f188d529`.

The preview subsequently completed successfully in **14 minutes 35 seconds**.
All seven PDFs and seven PNGs were generated, including reset/reset in purple;
their recorded hashes and plot-refresh proof match. At this inspection the
full 92-date array/report remain pending.

On these four dates, M1 has 1.5954% lower rollout loss than reset/reset but
higher mean absolute power-ratio error (67.4916% versus 65.9942%). At day 5,
M1 loss is 0.2824% higher (8.69141 versus 8.66693), with closely overlapping
coherence and slightly greater damping. At day 10, M1 loss is 4.3829% lower
but power error is higher (84.2451% versus 76.1326%). This is a tradeoff,
not demonstrated dominance over reset/reset. No spectral confidence intervals
or annual conclusions follow from this preview.

The rerun retains a finite numerical warning: reset pooled loss differs from
the archived four-date result by -0.02931 (about -0.29%); the maximum individual
relative difference is 1.40%. The small day-5 loss difference should therefore
not be emphasized. These values are preserved as warnings, not failure gates.

## November 14 — Accelerated submission of all 92 dates

At the user's request, the pending campaign was replaced with three 50-minute
GPU-test tasks and five regular-GPU tasks at 61 minutes:

| Job | Tasks | Partition / QoS | Walltime | Dates |
| --- | --- | --- | --- | --- |
| `17553195` | `0–2` | `gputest` / `gpu-test` | 50 minutes | 36 |
| `17553196` | `3–7` | `gpu` / `gpu-short` | 61 minutes | 56 |
| `17553202` | report | `cpu` / `test` | 30 minutes | all 92 |

Slurm denied an in-place queue change. Replacements were submitted held;
original array `17494525` and report `17494526` were canceled while still
pending, then the replacements were released. No completed work was lost.
Report `17553202` depends on `afterok:17553195:17553196` and refreshes the seven
literature figure pairs automatically. GPU-test permits three running and 25
submitted jobs per user; no other user GPU-test tasks existed at preparation.
Both arrays are unthrottled. Each shard retains one `gpu80&pcie` GPU, four
CPUs and 16 GiB RAM. Preview timing (523 seconds first date, approximately 84
seconds thereafter) supports 25–30 minutes per shard after starting.

The sealed campaign, frozen source, checkpoint, reset policy and full ordered
92-date January 1–October 1 cohort are unchanged, including forty leads and
83 channels. Scientific pins were verified. At final inspection all eight
tasks remained pending: GPU-test for unavailable nodes, regular GPU for
priority; the report waits on both arrays. Queue delay is unknown. No larger
reset spectra or new scientific conclusion is claimed.

Evidence: [plan](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_plan.json),
[job IDs](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_jobs.json),
and [release/dependency check](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/fast_20261114_activation.json).

## November 14 — Second GPU-test wave

After shards 00–02 completed and supplied the matched 36-date figures, the
user requested more pending work on GPU-test. New array `17560836_3–5` moves
34 dates to three 50-minute GPU-test tasks, within the per-user running limit
of three. Tasks `17553196_6–7` retain their regular-GPU queue age and 61-minute
limits for the final 22 dates. Each task still uses one `gpu80&pcie` GPU,
four CPUs and 16 GiB RAM.

The replacement array/report were submitted held, pending originals
`17553196_3–5` and report `17553202` were canceled, then replacements were
released. New report `17560837` depends on
`afterok:17560836:17553196_6:17553196_7`; completed shards 00–02 remain intact
and the report validates all eight shards. Initial replacement states were
pending, without holds. Source/campaign and scientific reference pins were
verified unchanged. No new spectra from this wave are claimed; completed
coverage remains 36/92 dates. Measured first-wave runtimes of 24m33s–24m55s
support roughly 25 minutes per new shard after starting; queue delay is unknown.

Evidence: [plan](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave2_20261114_plan.json),
[jobs](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave2_20261114_jobs.json),
and [activation](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave2_20261114_activation.json).
The final [queue check](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave2_20261114_queue.txt)
confirms all five remaining forecasts pending without holds and the report
waiting on the correct dependency.

## November 14 — Final GPU-test wave

Shards 00–05 completed with saved checks, bringing validated coverage to
70/92 dates. Under the existing acceleration request, the final 22 dates
now use GPU-test array `17572313_6–7`, each at 50 minutes with one
`gpu80&pcie` GPU, four CPUs and 16 GiB RAM. There were no other user GPU-test
jobs at preparation; two tasks remain within the three-running-job limit.
Recent comparable shards completed in 34–36 minutes; queue delay is unknown.

Replacements were submitted held; pending `17553196_6–7` and old report
`17560837` were canceled and confirmed absent before release. New report
`17572314` depends on `afterok:17572313` and checks all eight shards.
Initial replacement states are pending without holds. The completed 70 dates,
sealed campaign/source, checkpoint and scientific pins are unchanged; the
fixed 36-date manuscript artifacts remain intact. No final scientific result
is claimed before completion and pooling.

Evidence: [plan](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave3_20261114_plan.json),
[jobs](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave3_20261114_jobs.json),
and [activation](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave3_20261114_activation.json).
The final [queue check](../../artifacts/evaluations/v24_matched_reset_spectra_20261113/submissions/gputest_wave3_20261114_queue.txt)
shows task 6 configuring on `della-l02g13`, task 7 pending, and the report
waiting on the correct dependency; no new final-wave spectra are claimed.

## November 14 — Annual inventory correction and missing 30-date extension

The standard annual cohort contains 122 starts, January 1–December 30 every
third day. The original 92 reset spectral campaign deliberately matched the
completed portion of the current optimal-filter rerun, ending October 1; it
does not cover the remaining 30 starts on October 4–December 30.

The subsequent read-only audit confirms that raw GC/M1 spherical moments
already exist for **all 122** starts in both
`artifacts/evaluations/v24_smoothing_m1_20261107/evaluate/*/statistics/` and
`artifacts/evaluations/v24_smoothing_sweep_m1_20261109/evaluate/*/statistics/`.
Each campaign has 122 distinct date files with 83 channels, 40 leads and degrees
0–179; inspected metadata identifies the original M1 SWA `b224e69b…`.
The earlier statement that only 92 GC/M1 spectra were available was too broad:
92 is the completed count of the newer optimal-filter rerun. The old filtered
variants differ and require compatibility checks before reuse in that treatment.

No correct matched-reset tail spectra or saved fields were found in the
searched output roots. Historical annual reset results contain scalar losses.
The archived LH forecast cache uses a different seed 22 raw carry checkpoint
and only four surface variables; it is not the original matched-reset control.
The older full-year MGDA diagnostic is zonal rather than spherical.

The user now authorizes a separate tail campaign at
`artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/`:

- Exactly 30 missing starts, October 4–December 30, at all 40 six-hour leads.
- Nine GPU-test jobs: three with four dates and six with three dates.
- Walltime 50 minutes each; at most three running and 25 submitted GPU-test
  tasks per user, accounting for existing jobs when submitted.
- The same R1 SWA `db0e8032…`, zero initial memory, reset every step,
  `cold_full`, no warm-up, alpha one and unaltered autoregressive feedback.
- The existing 83-channel spherical observer and finite-drift warning policy.
- A validated annual report combining the original 92 reset dates with these 30,
  checking the exact 122-date union and preserving all prior interim artifacts.

The original 92-date campaign is now complete: tasks `17572313_6–7` finished
in 23m42s/23m28s, followed by report `17572314` in 1m09s. All eight shards passed
their saved checks; the inspected report summary records `status=complete`
and `count=92`. At tail preparation, GPU-test is empty and the verified user
limits are three running and 25 submitted tasks. Expected tail execution is
15–25 minutes per task after starting; queue delay remains unknown.

Submitted November 14 at 23:09:01 America/New_York:

| Job | Tasks | Partition / QoS | Walltime | Coverage |
| --- | --- | --- | --- | --- |
| `17574447` | `0–8` | `gputest` / `gpu-test` | 50 minutes | 30 tail dates |
| `17574448` | report | `cpu` / `test` | 30 minutes | 92+30 dates |

The CPU report depends on `afterok:17574447` and writes
`annual122_report/` in the tail campaign root. Every forecast task requests one
`gpu80&pcie` GPU, four CPUs and 16 GiB RAM. The report requests two CPUs and 12 GiB.
The initial queue snapshot shows all nine forecasts pending with reason `None`,
and the report pending on `Dependency`; no forecast task is held. The scheduler
enforces at most three running GPU-test tasks per user, and nine submitted tasks
fit within the 25-task submission limit verified before submission.

CPU preparation checks passed all nine forecast configurations, the exact
30+92=122-date union, inherited source verification and bit-for-bit reproduction
of all seven original 92 pooled arrays. Source digest:
`7a41a65f49c6b9c9b38da6757a09d0c45e520f2168328da435038bde6b887ba3`.
Campaign hash:
`d8fee6b03eab86b34a4ec88553f65365f957b2e432dd2651247e5d494ba8aeaa`.

Evidence: [preparation checks](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/checks/preparation_validation.json),
[forecast submission](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/evaluate.json),
[report submission](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/report.json),
and [initial queue](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/initial_queue.txt).
These establish submission and preparation, not completed tail forecasts or an
annual scientific result. The frozen 70-date v5 figures and archive are unchanged.

The later [verified queue](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/submissions/verified_queue.txt)
shows all nine forecasts still pending for `Priority`, and the report waiting
on its dependency. No tail task completion is claimed.

## November 15 — Verified annual completion and v5 handoff

The earlier pending snapshot is superseded. All nine tail tasks
`17574447_0–8` and report `17574448` finished `COMPLETED` with exit code `0:0`.
The report completed November 14 at 23:56:24 America/New_York. All nine saved
shard checks are complete and sum to 30 dates; the annual report combines
those with the original 92 into exactly 122 unique standard starts from
January 1 through December 30, 2023. All 40 six-hour leads through day 10,
83 channels and degrees 0–179 are present.

The November 15 audit checked the ordered annual cohort, the recorded digest
and all seven numeric pooled arrays; all arrays are finite. Evidence:
[annual summary](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/summary.json),
[completion check](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/checks/report.json),
and [pooled moments](../../artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114/annual122_report/power_pooled.npz).
The summary is `status=complete`, `scope=annual122`, `count=122`; the pool's
verified SHA256 is
`501a33216228a94e88b2f9739494fa1a661296d9c851ebdad1ed7137096e7a57`.
The latest optimal-filter GC/M1 campaign also now has all 122 date statistics.

Keep the saved finite-drift warning: pooled reset exact loss differs from its
archived evaluation by +0.000253116619395 (about +0.00244%); maximum local
relative difference is 3.29457% against a 0.1% advisory reference. These are
warnings, not failure conditions, and do not define a significance threshold.
The original reset checkpoint and protocol are unchanged.

The annual report has `plot_refresh=false`; completion did not itself replace
the manuscript figures. On November 15 the user authorized the annual v5
replacement, including its table, captions, numerical discussion and removal
of provisional wording. New replot output is kept under
`artifacts/evaluations/v24_matched_spectra_v5_replot_20261115`, portable figure
provenance under `Overleaf/archive/Graphs/v5_spectral_matched122_20261115`, and
build evidence under `artifacts/manuscripts/v5_spectral_matched122_20261115/build`.
The frozen 36/70-date archives and original seed/bootstrap evidence remain
separate. This manuscript work reuses saved annual evaluations; it launches
no forecasts or training.

The [completed annual replot](../../artifacts/evaluations/v24_matched_spectra_v5_replot_20261115/summary.json)
now supplies all five active v5 figures (6 and 10–13) and Table 1 exports on
the same 122 dates. M1 reduces exact loss relative to R1 by 2.8507% at day 5,
5.7560% over the rollout and 9.2566% at day 10, with power mismatch higher by
0.5030, 1.5053 and 7.9609 percentage points. M1 improves both aggregate metrics
relative to filtered GC; raw GC retains the lowest power mismatch. These are
fixed-checkpoint annual point estimates, not a same-spectrum superiority or
new uncertainty result. An
[independent 48-metric check](../../artifacts/evaluations/v24_matched_spectra_v5_replot_20261115/independent_numeric_validation.json)
reproduces the exported arithmetic to `1.4210854715202004e-14` maximum absolute
difference. The
[portable archive](../../Overleaf/archive/Graphs/v5_spectral_matched122_20261115/summary.json)
preserves numeric curves, plotting code, source pins and finite-drift warnings.
Both marked and clean v5 sources now reference the new annual figures and
updated table/contrasts, with provisional spectral wording removed. Before-edit
manuscript snapshots and the numeric replacement map remain in
`artifacts/manuscripts/v5_spectral_matched122_20261115/`.

The [final manuscript check](../../artifacts/manuscripts/v5_spectral_matched122_20261115/validation.json)
is complete: marked/clean v5 builds have 37/29 pages and zero LaTeX warnings,
with the clean main text still ten pages. Annual numbers and absence of
provisional wording are verified; inspected figure/table pages have no clipping
or overlap. The v96w response is synchronized and both response build variants
pass. Original collaborator comments and prior spectral archives are retained.
