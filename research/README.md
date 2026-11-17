# Research

Updated: 2026-11-15 (America/New_York)

November 15 [authorized manuscript illustration revision](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#v5-variable-spectral-presentation-20261115):
Section 4.6 retains combined 83-channel amplitude/coherence; Appendix C now
pairs day-5/day-10 spectra for T2m, Z500, U850 and Q100 with native-grid RMSE
reductions versus R1. Z500 replaces T400 in the manuscript illustrations;
the earlier exploratory selection remains archived. The scalar comparison
is retained as Table 6, with the new field RMSE comparison in Table 7.
The adverse Q100 result, all-channel tradeoff, original equal-degree metric,
and explicitly separate power-weighted exports remain visible. Both v5
sources and the v96w response are synchronized; existing forecasts were reused.
The [reference-line clarification](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#v5-spectral-reference-explanation-20261115)
adds the Subich rationale and interpretation in the main text: M1/R1 first
crossings coincide at degree 16 on day 5, but separate to 7/12 on day 10.
The appendix now also interprets the ranges below the reference for each
illustrated variable, relating them to coherence and native-grid RMSE.

November 15 [per-variable accuracy and spectral audit](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#v5-variable-spectral-audit-20261115):
all 83 channels now have matched native-grid RMSE, equal-degree and
truth-power-weighted spectral mismatch, coherence, and scale-crossing exports.
At day 10 versus R1, M1 improves RMSE in 81 channels: eight also improve
power-weighted mismatch, while 73 trade accuracy for larger mismatch; two
channels worsen in both. Equal-degree mismatch increases in all 83 channels,
so the new weighting must be named explicitly. Illustrative figures use T2m,
T400, U850 and Q100 to show lead-dependent improvements, tradeoffs and an
adverse example. This is exploratory analysis of existing fixed checkpoints;
that exploratory step added no forecasts, uncertainty estimates or manuscript
revisions. The subsequent authorized manuscript revision is linked above.

November 15 [completed annual spectra and v5 update](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#v5-spectra-matched122-20261115):
all 122 standard 2023 starts now have matched spherical spectra through day 10.
The nine reset tail tasks `17574447_0–8` and report `17574448` completed with
exit code zero; the annual report finished November 14 at 23:56:24 Eastern.
Its exact 92+30-date union, all seven finite pooled arrays and recorded digest
were verified November 15. The current optimal-filter GC/M1 campaign also
contains all 122 dates. Both v5 sources now replace the 70-date spectral
presentation with this annual cohort. The completed replot gives M1 exact-loss
reductions versus R1 of 2.85% at day 5, 5.76% over the rollout and 9.26% at
day 10, with power mismatch higher by 0.50/1.51/7.96 percentage points,
respectively. All five figures and numeric exports have a separate 122-date
archive. Both PDFs compile with zero LaTeX warnings; the clean main text
remains ten pages. The earlier dated entries below are historical snapshots.
Preserve finite rerun-drift warnings and the separate original bootstrap/seed
results.

November 14 [annual spectral audit and reset tail extension](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-reset-spectra-tail30-20261114):
the November 7 and 9 smoothing campaigns already contain raw GC/M1 spherical
moments for all 122 dates. The 92-date limit applies to the newer optimal-filter
rerun, not all saved GC/M1 spectra. No matching reset spectra for October 4–
December 30 were found. The user authorized those 30 dates as nine GPU-test
jobs of three or four dates, each at 50 minutes, within the per-user limits
of three running and 25 submitted tasks. Submitted array `17574447_0–8` uses
50 minutes per task; CPU report `17574448` follows `afterok:17574447`.
The initial check found all nine tasks pending and the report waiting on its
dependency. The original 92 reset dates are now complete:
`17572313_6–7` finished in 23m42s/23m28s and report `17572314` passed with all
92 dates. Annual pooling will reuse those 92 plus the new 30; the frozen 70-date
v5 artifacts remain preserved.

November 14 [v5 matched 70-date replot](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#v5-spectra-matched70-20261114):
the user chose to update the completed subset now. Figures 6 and 10–13,
Table 1 and both v5 sources use the same 70 starts, January 1–July 27.
M1 lowers exact loss versus R1 by 2.92% at day 5, 5.93% over the rollout and
9.70% at day 10; day-5 power mismatch is 0.54 percentage points higher.
These remain provisional fixed-checkpoint results, without new uncertainty.
The standard annual cohort is 122; 92 was the completed subset of the current
optimal-filter GC/M1 rerun. Older raw GC/M1 moments cover all 122 dates, as
established by the audit above. The 30 October–December reset spectra are not
included in the original reset campaign. No additional forecasts were submitted
for this replot.

November 14 [final GPU-test wave](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-reset-spectra-wave3-20261114):
70/92 dates are complete and validated. The final 22 dates now use released
GPU-test tasks `17572313_6–7` at 50 minutes, with report `17572314` after both.
These replace pending `17553196_6–7` and report `17560837`; completed outputs
and the fixed 36-date manuscript artifacts remain unchanged. The initial check
found task 6 configuring and task 7 pending. Both subsequently completed, followed
by the successful 92-date report; the annual 122-date extension is separate.

November 14 [Section 4.6 smoothing explanation](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#smoothing-section46-20261114):
the user replaces the brief Section4.5 paragraph with a dedicated Section4.6.
The main text now explains spectral relevance, cites GraphCast/AMSE, outlines
the output-filter procedure and shows day-5 spectra in Figure 7 plus the
four-model percentage summary in Table 2. Both metrics are explicitly labeled
as reductions relative to raw GraphCast. Appendix E retains detailed methods.
The fixed36 data and all figure bytes are unchanged; day5supports the design
with a small aggregate spectral cost, while day10has higher absolute MSE and
lower coherence. Space optimization is deferred by the user.

November 14 [36-date Appendix E revision](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#appendix-e-matched36-20261114):
`Overleaf/v4.tex` now uses the fixed January 1–April 16 cohort throughout
Appendix E, its table and all eight figures. The main text retains one short
red paragraph. Day-5 carry/carry improves MSE by 2.84% versus reset/reset with
0.50 percentage points higher aggregate power mismatch; the larger day-10
spectral cost remains explicit. All four comparators are retained. The 36-date
data, source pins and plotting exports are archived separately and must not
be overwritten when the larger campaign completes. Main 122-date results
and their uncertainty are unchanged.

November 14 [second GPU-test wave](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-reset-spectra-wave2-20261114):
completed reset spectra remain 36/92 dates. New `17560836_3–5` covers 34 dates
on GPU-test at 50 minutes, within the three-job limit; retained
`17553196_6–7` covers 22 dates on regular GPU at 61 minutes. Report `17560837`
follows all remaining tasks. Replacements are released; completed outputs,
frozen computation and the matched 36-date results below are unchanged.

November 14 [36-date matched spectral update](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-spectra-36-20261114):
newly completed reset shards 00–02 provide 36 dates (January 1–April 16).
All four models are re-pooled on this same cohort for losses, spectra,
amplitude/coherence and both day-5/day-10 feature atlases. Carry/carry has
2.84% lower day-5 MSE than reset/reset, with power mismatch 68.66% versus
68.16%; the rollout/day-10 MSE advantages are 6.01%/10.20%. These remain
interim January–April results. See `v24_literature_spectral_matched36_20261114`.

November 14 [day-10 per-feature spectra](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#feature-spectra-day10-20261114):
added all 57 day-10 PDF/PNG counterparts in the existing four-date `by_feature`
folder, with all four models and 83 channels. Day-5 artifacts are unchanged;
this reuses saved moments and launches no forecasts or training.

November 14 [accelerated matched-reset spectra](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-reset-spectra-fast-20261114):
all 92 fixed dates are scheduled: `17553195_0–2` on GPU-test for 50 minutes
(36 dates), `17553196_3–7` on regular GPU for 61 minutes (56 dates), and report
`17553202` after both. These released jobs replace pending `17494525/17494526`;
no completed work was lost. Frozen computation and cohort are unchanged.
The three GPU-test shards subsequently completed, providing the 36-date
interim comparison above. The full 92-date spectral campaign is incomplete.

November 14 [systematic reset/reset baseline in Appendix E](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#appendix-e-four-models-20261114):
all active appendix comparisons use raw GC, filtered GC, raw carry/carry and
raw reset/reset on the same four completed spectral dates. Filtered M1 is
excluded from active presentation. Larger 92-date results remain archived;
their confidence intervals are not transferred to the four-date diagnostic.
The principal 122-date memory result is unchanged. No new forecasts or training
were launched; future Appendix E extensions must preserve the four-model cohort.

November 14 [two-policy memory presentation](directions/r001-matched-memory-training/experiments.md#two-memory-policies-20261114):
the manuscript, Figure 6 and reviewer response now focus on carry/carry versus
separately trained reset/reset. The carry/reset diagnostic is omitted from
active presentation and retained in historical artifacts. Retained scores,
date confidence intervals, seed standard deviations and figure size are
unchanged; the one-reset-seed limitation remains explicit. No new runs or
uncertainty calculations were performed.

November 14 [pilot-wording correction](directions/r001-matched-memory-training/experiments.md#pilot-wording-cleanup-20261114):
the user reaffirms that the earlier 2023 pilot did not inform model selection.
Removed pilot-exposure framing and the related exclusion-sensitivity result
from active `v4.tex`; do not reintroduce them in the manuscript or response.
Internal provenance and all actual evaluation data and primary scores remain
unchanged. The earlier note to retain manuscript exposure disclosures is superseded.

November 14 [appendix presentation](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#smoothing-appendix-layout-20261114):
`v4.tex` now keeps one short smoothing paragraph in the main text and places
the detailed discussion and figures in Appendix E. New Figures 16–18 compare
original carry/carry and reset/reset loss, amplitude/coherence and physical
power at days 5 and 10 on the same four dates. No amplitude matching is applied;
the separate 92-date filtering comparison remains clearly identified. New
edits are red, earlier blue changes preserved, and numerical exports archived.

November 14 [equal-power memory diagnostic](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#equal-power-memory-20261114):
on the four-date preview, matching date-pooled power separately for each
channel/lead/degree gives carry/carry 0.9131% higher represented weighted
spectral MSE than reset/reset over the rollout. No uncertainty/significance
or full-year conclusion is inferred. This does not support the requested
lower-MSE-at-the-same-spectrum claim; raw native-grid memory-loss results
remain unchanged. No manuscript edit, publication or forecast was made for
this diagnostic.

November 14 [smoothing manuscript revision](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#smoothing-manuscript-20261114)
adds the 92-date filtering comparison to ICLR `v4.tex`, with new changes in red
and the four-date reset/reset spectral preview kept separate in Appendix E.
Raw M1 versus filtered GC has 4.2643% lower rollout loss, paired 95% interval
[3.3738,5.3438]%, and lower aggregate power mismatch (67.5267% vs78.3995%).
Raw GC retains better power matching. A memory benefit at identical spectral
fidelity is not claimed. New manuscript figures and archived numeric sources
are prepared; this step launches no forecasts or training.

**Current full-GC schedule:** all five lambda arms now start fresh at
**constant `3e-7` from the first update, with zero warmup and no decay**.
Training `17429091 → 17429092`, history `17429093` and completion `17429094`
replace the earlier cosine submissions, including the old lambda0 continuation.
No constant-rate scientific result exists yet. The saved cosine lambda0
checkpoints are historical evidence and cannot stand in for this new schedule.
The November `1e-5` full-GC lineages remain excluded from scientific comparisons.
GC–Mamba's M1-matched training/history `17406861/17406862/17406863` is unchanged.
See [e014](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e014)
and the [current protocol](../docs/experiments/v24_full_gc_constant_lr3em7_2026-11-12.md).
The rebuilt mixed raw/SWA chain is `17429215 → 17429216 → 17429217`;
annual/smoothing controller `17429218` follows the same training dependencies,
`17429092:17406862`.

November 13 [optimal-blurring evaluation for v4](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e015):
the 2022 fit is complete and 92/122 annual starts are evaluated (January 1–
October 1). Filtered M1 has 4.9850% lower rollout loss than filtered GC
(paired 95% interval 4.1243–5.9559%) on this subset. The remaining 30 October–
December starts in `17430310_6/7` are queued; automatic report `17430311` waits.
This is incomplete annual evidence. The [interim readout](../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/REPORT.md)
preserves canonical v4 loss reductions, separate spectral scores and measured
rerun drift. Existing training and constrained-smoothing campaigns are unchanged.
The [joint MSE/power comparison](../plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113/REPORT.md)
shows that filtering improves MSE but worsens power matching for both models.
Raw M1 beats filtered GC on both aggregate measures; versus raw GC it improves
MSE but worsens power matching. Full-year power report `17470053` follows
`17430311`. These are still 92-start point estimates, not annual conclusions.
See the [protocol](../docs/experiments/v24_optimal_smoothing_v4_2026-11-12.md).

November 13 [literature-style spectral figures](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#literature-spectral-20261113)
add AMSE-style weighted amplitude/coherence, GraphCast-style represented RMSE
and physical spectra, plus separate exact native-grid loss on those same 92
starts. The [presentation plan](../docs/reports/v24_spectral_presentation_plan_2026-11-13.txt)
keeps day 5 prominent and retains day 10 in the appendix. Original AMSE figures
are archived as references; their ensemble-mean RMSE is not our deterministic
loss. No training, inference or manuscript publication was performed.

The subsequent [matched reset/reset addition](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#matched-reset-spectra-20261113)
now includes archived reset loss on those same 92 dates. Its spectra were not
saved previously; raw spectral arrays **17553195/17553196** are queued, with
dependent report **17553202** to add the matched control to all seven figures
automatically. They replace pending November 13 jobs **17494525/17494526**.
A better memory-related spectral
tradeoff remains a hypothesis until those measurements complete.
Fast preview **17495398** completed in **14m35s** on `gpu-test` within its
requested59-minute limit. All seven figures now include reset/reset in the
separate four-date preview folder. M1 has1.60% lower rollout loss but worse
power matching; the preview does not demonstrate dominance over reset/reset.

The same preview now has [per-feature day-5 diagnostics](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#feature-spectra-20261113)
covering all 83 channels. T2m retains power relatively well; precipitation and
mid-atmosphere vertical velocity have strong deficits. Z500/T850 retain most
large-scale power while losing weak small-scale power, so both equal-degree
and ERA5-power-weighted mismatch rankings accompany the complete curves.

November 13 [training duration and AR horizon summary](../plots/analyze_models/data/resolution_eval/v24_training_steps_and_k_20261113/REPORT.md)
adds four reproducible figures from saved M1, GC–Mamba K-sweep and older MGDA
evaluations. MSE versus K is measured; joint MSE/power versus updates is measured
at fixed K20. Matched spectra across K remain missing. M1's raw checkpoint
trajectory is non-monotonic, and MGDA's improved power matching costs some MSE
relative to its parent. The cohorts and zonal/spherical metrics remain separate;
this synthesis adds no training or inference and no constant-rate full-GC result.

This index starts with the current matched memory-training comparison. Earlier
experiments remain in the existing records; this is not a historical migration.

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Does training with memory improve the selected configuration?](directions/r001-matched-memory-training/overview.md) | training | active | Nov13 e003: five fresh local carry SWAs give 14.1682% ± 0.4332 pp rollout and 21.0283% ± 0.7502 pp day-10 reduction (mean ± seed SD). Original matched-SWA 5.261 pp [4.698,5.862] contrast retained; local raw reset control has one seed. | Finish v4 build/publication separately. Carry-seed variation is measured; matched reset-seed variation remains unmeasured. Keep original/local and SWA/raw provenance distinct. |
| r002 | [How does the selected correction compare with higher-resolution GraphCast?](directions/r002-common-grid-forecast-comparison/overview.md) | evaluation | active | Nov13 e013: existing ENS-matched Figure 5/9 scores gain paired pointwise 95% date intervals on 91–108 starts per point; all twelve day-10 M1-vs-GC025 intervals have positive lower bounds. Point estimates unchanged. Separate 2018 annual completion remains unverified. | Finish v4 build/publication with date uncertainty distinguished from seed SD. Inspect annual array15870083, report15870088 and plot15870456 before claiming 2018 paper parity; no new status check performed. |
| r003 | [Fresh residual width comparison](directions/r003-fresh-residual-width/overview.md) | training | closed | Unexecuted proposal superseded: the user clarified that the question is the benefit of larger data. | Follow r004. |
| r004 | [Does more cached training data improve each residual width?](directions/r004-cached-training-data-scale/overview.md) | data | active | Oct28:20y replacement submitted; verification16392462 passed, prepare16392486 queued; new-arm startup16392467/68 waits on cache. Both7y controls reused; obsolete pending40y chain cancelled. No data-size result. | Inspect new2002–2022 preparation/cache progression, then two20k-open/2k-closed lineages and common2023 evaluation. |
| r005 | [Can MGDA improve spectral fidelity while preserving exact GC accuracy?](directions/r005-mgda-accuracy-spectral-adjustment/overview.md) | training | active | Nov15: matched annual spectra, five-figure replot and both v5 builds complete for all 122 starts. M1 exact loss is 2.85%/5.76%/9.26% lower than R1 at day 5/rollout/day 10, with 0.50/1.51/7.96 pp higher power mismatch. Constant 3e-7 full-GC and separate GC–Mamba schedules are unchanged; the 1e-5 exclusion remains. | Use the separately archived 122-date v5 result while preserving 70-date history and original seed/bootstrap evidence. Inspect constant-rate full-GC and GCM completion, then their raw/SWA and annual evaluations; retain the separate horizon-training hold. |
| r006 | [Do the principal and spectral-adjusted corrections preserve extreme-event skill?](directions/r006-extreme-event-evaluation/overview.md) | evaluation | active | Nov14: canonical M1 has lower day-5 primary/global CSI and recall for all four events, with paired 95% difference intervals below zero. Reset/reset extension17564547_0–7→17564548 submitted on regular GPU; initial GPU-test jobs canceled on user instruction. No new result. | Inspect the matched122-start/40-lead three-model report and M1-minus-reset intervals. Reuse thresholds and canonical M1/GC; no spatial filtering or gain-PR sweep. Fixed-threshold evidence does not order whole PR curves. |

Earlier records: [experiment index](../current_experiments.md),
[2023 memory evaluation](../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).

November 12 corrected cross-cluster handoff: [current read-only portable instructions for Lianghong](../docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md)
cover M1-matched GC–Mamba and four fresh full-GC horizons. Full GC now uses
constant3e-7/no-warmup atlambda0; the new package is
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/`.
Validation passed34 CPU tests and four real full-GC configurations while
preserving672 original computational source files. Original cosine archives
remain immutable and superseded for full-GC execution. No remote job or new
scientific result is claimed.

Code-only delivery is now available in
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7_code_only/`:
send its2.28MB code archive and matching checksum. The included README and
download metadata point to the existing495.19MB weights archive on Della,
with an SSH/rsync command and pinned SHA256. Source, templates and launchers
are unchanged; remote SSH/download was not tested. This is a transfer option,
not a new experiment or a public download endpoint.

The dated preparation/ownership records below are historical. Their scheduler
statements describe those earlier checkpoints in the experiment history;
the current schedule and active production IDs are above and in e014.

Prepared 2026-11-09: [full-GC K8/K12/K16/K20 horizons, checkpoint spectra and
SWA](../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md). K20 update500
reduced development loss4.6434% versus public GC. At the user's request, new
submission is on hold while the reviewer-comparison gaps are addressed: the
draft omits M1 and matched2023 evaluation. Existing M1 results can be reused;
the existing K20 continuation is unchanged.

Reviewer-response ownership (2026-11-07): [Lianghong's README](../docs/experiments/README_LH.md)
assigns full-GC fine-tuning, extreme-event analysis and seed replication to
Lianghong, with the spectral/λ/smoothing study owned by Ilya. Maps/gradient
diagnostics remain unassigned. Alternative PEFT methods (such as LoRA) and
matched GRU/LSTM or fixed-history comparisons also remain unassigned, both rated
2/5 importance and lower priority than the main comparison.
The November7 framing revision removes the equal-budget requirement from
full-GC fine-tuning: that item asks whether further adaptation of the already
fine-tuned baseline helps on the small training dataset. The equal-update-budget
comparison with GC–Mamba moves to the unassigned LoRA item. Completed500-update
full-GC calibrations showed no development improvement; broader conclusions await
the remaining runs. Existing submitted schedules and endpoints are unchanged.
Scope clarification: Lianghong's item2 uses the standard GC loss (λ=0); Ilya's
item1 owns the full five-λ sweep for both methods and smoothing, reusing the
standard-loss GC result as its λ=0 reference. Item1 remains first in the work
order, with importance4/5 and moderate difficulty given the existing pipeline.
The proposed first seed scope is M1 plus matched
reset at seeds23/24; the fresh GC/GC–Mamba pair repeats in the broader plan are a
separate question. No new jobs were launched while documenting this handoff.

Historical initial launch (2026-11-07): [fresh GC–Mamba versus full-GC scalar λ sweep and smoothing](../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md).
Implementation is complete; initial training array 17022621, independent GC/Mamba
controllers 17022622/17022623, and GC/M1 smoothing 17022624→17022625 are submitted
and initially queued. They automatically continue through development LR
calibration, five fresh λ arms per method, filter fitting and common annual
evaluation. Later November7 checks established full-GC feasibility on one80GB
GPU. At user request, pending λ0 job17049162_0 now starts fresh at peak3e-7;
continuation17049163_0 is retained. Other GCFT λ arms remain1e-5, so this revision
is not a fixed-LR λ-only comparison. Original500-update calibration results
(dev32 exact loss increases13.8958%/52.2477% at1e-5/3e-5) are preserved.
New step500 evaluation17062379 supplies matching evidence for the existing
history evaluation. The lower-rate production run has no new skill result yet.
See [r005 e008](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e008);
existing MGDA and extreme-event campaigns retain their separate scope/status.
