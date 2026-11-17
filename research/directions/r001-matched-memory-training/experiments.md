# Attempts and evidence

<a id="e001"></a>
## e001 — Reset training matched to selected di16/Mamba-LR×3 model

Event date: 2026-10-15
Recorded: 2026-10-15 (America/New_York)
Execution: completed (training; subsequent forecast evaluation is e002)
Outcome: pending

**Question and change:** Compare memory retention with per-prediction reset.
Change only `sequence.temporal_state_policy` from `carry` to
`reset_every_anchor` in both original stage configurations, plus output paths.
Keep `architecture.temporal_stateful=true`: the same Mamba architecture is
executed with both SSM and convolution state reset before each prediction.

**Setup:** Fresh 500-update warm-up at equal group learning rates, constant
1e-4 after 50-update ramp; independent 2,000-update branch with 50-update ramp,
cosine spatial LR 1e-4→1e-5 and Mamba multiplier 3. Seed22, batch4, width512,
di16/BCG1, two residual processor steps, BPTT24/K20/segment96, all-step loss,
BF16 compute/FP32 tape, frozen GraphCast, ERA5 2015–2021 training. Branch uses
weights-only initialization from its own reset-trained warm-up, resetting
optimizer, RNG, data cursor and memory as in the original run. SWA averages
branch updates500/1000/1500/2000. No reuse of carry-trained warm-up weights.

**Provenance:** [manifest](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/manifest.json),
[source origins](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/source_provenance.json),
[snapshot](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/source/snapshot.json).
Critical training/model code comes from the selected model's saved September
snapshot; missing dependencies are frozen from the current workspace, with the
prepared-array loader restored from original revision56a3c50. The historical
snapshot was incomplete, so bitwise historical reproduction is not claimed.
The snapshot identifies the exact new runtime, including uncommitted launcher
changes. Execution environment: Della, `graphcast311`; baseline and anchor
fingerprints must match the selected model. Prepared metadata/statistics and
software versions are recorded by preflight.

**Resource evidence:** Principal job13740084_6 completed500+2000 updates/SWA in
22:40:45 with97.1GiB peak CPU RAM and23,701MiB sampled GPU RAM. Other completed
same-family jobs reached234.8GiB CPU RAM. Request one A10080,8CPUs,320GiB,30h;
this does not assume reset training will be faster.

**Observation:** Results pending. Slurm submission and CPU checks are recorded
below when complete. No new forecast evaluation has been submitted.

**Next action:** Verify training completion, zero recurrent checkpoint state,
and the prescribed SWA inputs. Evaluate the finished SWA on the same122-start
2023 protocol with per-step reset before interpreting or updating the paper.

### 2026-10-15 — Validation and submission

Submitted job **15622842**, one A10080, 8 CPUs, 320 GiB CPU RAM, 30-hour
limit. Submission is not completion.

Configuration/source/baseline/anchor checks passed. Seven archived batch/LR
CPU tests and six carry/reset behavioral cases passed, including per-prediction
zeroing, zero incoming-state gradient, and agreement with independent loss,
gradient, and update calculations.

[Launch checks](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/checks/launch_validation.json)
and [submission command](../../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/submissions/jobs.json).
Training/SWA outcomes and the subsequent matched 2023 evaluation remain pending.

### 2026-10-15 — Draft comparison restored on user request

The user explicitly requested a draft update after training submission. Restored
the historical reset/reset row and curve alongside the selected checkpoint's
carry/carry and carry/reset results in `Overleaf/arxiv_versoin.tex`. Its verified
2023 reductions are 8.379241789377001% over the rollout and
11.952117594745781% at day 10, from the existing `R_reset` evaluation on the same
122 starts. The table identifies the earlier training configuration and pending
matched replacement. These are historical measurements, not results from
job15622842. The new PDF builds and its revised table/figure were inspected.

### 2026-10-18 — Training completion verified

Job15622842 completed in19:40:32. The completion receipt verifies500 warm-up
and2,000 main updates, plus SWA at main updates500/1000/1500/2000. New SWA:
`artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl`;
SHA256 `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6`.
The frozen original evaluator's checkpoint loader accepts it; parameter-tree
compatibility with M1 passes and all10,348,879 parameter values are finite FP32.
Evidence: `checks/training_complete.json` and `checks/eval_preflight_20261018.json`
under that training directory. Scientific training-policy outcome awaits e002.

<a id="e002"></a>
## e002 — Matched reset-trained 2023 forecast evaluation

Event date: 2026-10-18
Recorded: 2026-10-18 (America/New_York)
Execution: completed (all eight shards15788731 and report15788744)
Outcome: supported for the fixed-checkpoint matched training-policy comparison

User instruction: complete the matched memory-control evaluation on the
original122 initializations and40 leads, with a paired M1 comparison. No new
model selection, training, or test-year tuning is performed.

Workflow: `artifacts/evaluations/v24_matched_reset_2023_20261018`.
Protocol SHA256:
`81520bf89343412676b7cdf7229200f673a868245426f6176d41756b9d12d9cb`.
Original campaign remains
`artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009`, protocol
`51bee2021076fffeabb58ca8bc86c9c08d951a43879e1b75cd62e6da431fe065`.
Original evaluator/source, dataset metadata, statistics, package versions,
baseline checkpoint, input manifest, RNG policy, precision and GPU kind/driver
are checked against that frozen campaign. Extension code/report/wrappers are
copied and hashed under `control/`; original forecasts/protocol are preserved.

Policy: new model ID `R_matched`, training `reset_every_anchor`, forecasting
`reset_state_every_step=True`, `cold_full`, zero initial SSM/convolution state,
full corrected meteorological feedback, alpha1, seed0, no forecast warm-up.
Eight strided shards cover15/16 starts each, from January1 through December30,
2023, at00UTC, with all40 six-hour leads through240h and January2024 truth.

Comparators reuse original M1 (carry/carry) and M1_reset (carry/reset).
The original M1 execution supplies the canonical GraphCast baseline. Primary
new endpoint is carry/carry minus matched reset/reset rollout reduction,
measured in percentage points of canonical GC loss reduction; day10 is
secondary. Ratios pool exact raw GraphCast losses before division. Paired
non-wrapping bootstrap uses10,000 draws, seed20260916, five-start blocks;
sensitivities use3/7-start blocks, calendar quarters, and masking the three
previously exposed pilot starts on the original122-position grid. Repeated
baseline drift and own-baseline denominator sensitivity are reported. Intervals
condition on fixed checkpoints/one year; this new planned contrast is separate
from the original confirmatory endpoint family. No expected ranking is imposed.

Resources per shard: one A10080 PCIe,8CPU,12GiB,1h02m. Exact comparable original
reset-inference shards15297331_24–31 completed in22m12s–23m26s, with7.42–7.68GiB
peak host RAM. No concurrency throttle was added. Array **15788731** and
afterok report **15788744** were accepted by Slurm. First shard successfully
loaded the reset SWA and printed the correct manifest/reset policy.
Submission record: `submissions/jobs.json`; logs: `logs/eval_15788731_*.{out,err}`.

Report validation requires complete eight-shard coverage, all122 starts,
40 finite leads, matching units/input/valid-time metadata and checkpoint/source
identities. It creates `report/report.md`, `report/summary.json`, paired CSVs,
and exact-loss/paired-contrast PNG/PDF figures under
`plots/analyze_models/{data,images}/resolution_eval/v24_matched_reset_2023_20261018/`.
Before freezing,21 focused report tests passed, all16 real reused M1/M1_reset
shards validated, and a temporary integration fixture generated all outputs.
The fixture is synthetic and is not scientific evidence.

Next: inspect actual completion, numerical diagnostics, intervals and
sensitivities; then replace the explicitly provisional historical reset/reset
article result with the matched measurement if validation succeeds. Submission
and test success do not establish the scientific result.

### 2026-10-18 — Completed matched-control evaluation and validation

All eight GPU shards completed successfully in22m15s–24m01s; report15788744
completed successfully in20s. Re-ran the frozen provenance/coverage audit on
all24 files (M1, M1_reset and R_matched), verifying three policies ×122 starts
×40 finite leads. Independently pooled the raw losses and reproduced the
reported reductions. Both figure families were generated; the three-policy
loss figure was visually inspected.

| Training / forecasting | Rollout GC loss reduction | Day10 GC loss reduction |
| --- | ---: | ---: |
| Carry / carry |14.4240%|21.6584%|
| Carry / reset |4.5725%|9.6166%|
| Matched reset / reset |9.1627%|13.6359%|

Carry/carry minus matched reset/reset: **5.2614 percentage points** rollout,
paired pointwise95%CI **4.698–5.862**; **8.0225 points** day10, CI **7.202–9.047**.
The rollout contrast remains positive under3/7-start blocks, quarter blocks
and the119-start pilot-exclusion sensitivity. Own-baseline normalization gives
5.341 points [4.779,5.951], preserving interpretation. The reset-trained model
also exceeds carry/reset, showing why inference reset alone overstates the
training-policy difference.

Numerical warning retained: R_matched repeatedGC baseline pooled drift0.0872%,
worst start/lead drift4.3923%. Canonical and own-baseline conclusions agree.
These are fixed-checkpoint/year intervals; one training realization and the
incomplete historical source snapshot remain limitations. The new contrast is
not part of the original confirmatory endpoint family.

Interpretation: persistent memory during training and forecasting improves
the selected configuration relative to its matched reset-trained control.
The historical provisional reset/reset values8.3792% rollout/11.9521% day10
should be replaced by9.1627%/13.6359% in the article. Manuscript replacement
has not yet been applied in this completion/status check.

Artifacts: workflow `report/report.md`, `report/summary.json`,
`report/per_initialization_losses.csv`, and
`plots/analyze_models/images/resolution_eval/v24_matched_reset_2023_20261018/{exact_loss_reduction,paired_memory_contrasts}.{png,pdf}`.

### 2026-10-18 — Matched control incorporated into the arXiv manuscript

User instruction: replace the provisional historical reset/reset result with
this completed matched control, retaining inference-only reset as the third arm.

Updated `Overleaf/arxiv_versoin.tex` Section 4.5, Figure 6/caption, the memory
contribution, conclusion, and training appendix. New Appendix A.3 describes
paired bootstrap uncertainty, the separate planned extension, fixed-checkpoint
scope, all prescribed sensitivity families, and common/own-baseline numerical
checks. Reset/reset is now 9.16% rollout /13.64% day 10; carry/carry advantage
is5.26 percentage points [4.70,5.86], with the secondary day-10 advantage
8.02 [7.20,9.05]. Carry/reset4.57%/9.62% remains separately interpreted.
The historical8.38%/11.95% and pending/provisional language were removed from
the active manuscript. Original evaluation artifacts remain archived unchanged.

`Overleaf/tools/prepare_arxiv_2023.py --memory-only` regenerates the three-policy
figure from the frozen matched report. It verifies all24 source shards,
three policies ×122 starts ×40 finite leads, raw pooled exact losses, report
receipt and42 provenance hashes. The updated `exact_loss_primary.csv` uses
`R_matched`, and `memory_paired_contrasts.csv` preserves the new pointwise
contrast intervals separately from the original named-endpoint family.
Figure6 presents all three policies. All21 other figure files
were verified byte-identical. CSVs and source hashes accompany the figure in
`Overleaf/Graphs/arxiv_2023/` and the usual resolution-eval output directories.

Validation: independent manuscript review matched all scores, intervals and
training details to frozen evidence. Built with bundled `Overleaf/texmf`,
pdfLaTeX, BibTeX and two further pdfLaTeX passes; no unresolved citations or
references. Inspected rendered pages2,9–11,16–17 and the updated figure.
The27-page output replaces `Overleaf/arxiv_versoin.pdf`. The only remaining
LaTeX overfull-box diagnostic is1.60pt in the unchanged architecture caption.
Build outputs are in `Overleaf/build/arxiv_matched_reset_20261018/`.
No new training, forecast inference, submission, or publication was performed.

<a id="e003"></a>
## e003 — 2026-11-13: five fresh M1 seeds and separate raw memory controls in v4

Event date: 2026-11-13 (existing-evidence inspection and figure preparation)
Recorded: 2026-11-13 (America/New_York)
Execution: completed (score validation, aggregation and figures; manuscript
build/publication pending separately)
Outcome: supported for carry-model replication under the reconstructed recipe;
multi-seed matched memory-training effects remain unmeasured

**Question and scope:** Address seed variability using Lianghong's already
completed replication export, while retaining the original matched memory
comparison. No new training, forecast generation, Slurm job or bootstrap
resampling was performed for this seed-figure update. Existing per-seed date
intervals and original memory intervals are reused, not relabeled seed intervals.

**Setup:** Five fresh carry/carry runs, seeds 22–26, frozen public GraphCast-small,
ERA5 2015–2021, batch four, width 512, Mamba inner/state dimension 16, one B/C
group, two spatial message-passing steps, BPTT 24, K=20 and 96-step segments.
All use 500 warm-up updates (50-step ramp to 1e-4, then constant), followed by
2,000 main updates (50-step ramp, cosine spatial LR 1e-4 to 1e-5, Mamba LR ×3).
SWA averages main steps 500/1000/1500/2000. The resolved configs agree in these
settings and in backbone/anchor fingerprints. Evaluation uses the same 122
every-third-day 00 UTC starts in 2023, all 40 six-hour leads, cold zero memory
and full corrected forecast feedback. Every one of the 12 whitelisted SWA/raw/
control models has 4,880 unique finite start/lead records and valid timestamps.
Losses pool before division; T2m MSE pools before the square root.

**Observed SWA results (exact GraphCast loss reduction):**

| Local training seed | Ten-day rollout | Day 10 |
| --- | ---: | ---: |
| 22 | 14.1421% | 21.1878% |
| 23 | 14.8354% | 22.1537% |
| 24 | 13.6940% | 20.0831% |
| 25 | 14.2663% | 20.7941% |
| 26 | 13.9032% | 20.9226% |
| Mean ± sample SD | 14.1682% ± 0.4332 pp | 21.0283% ± 0.7502 pp |

Each SWA uses its own paired public GraphCast trajectory. Figure 4's dashed
absolute-error baseline is the arithmetic mean of the five paired baseline
curves; its reductions average the five independently computed ratios. The
band is ±1 sample SD across training seeds, not a confidence interval or an
ensemble forecast. The existing date intervals use 10,000 paired non-wrapping
bootstrap draws with five-start blocks and seed 20260916, conditional on each
trained model; they remain distinct from seed variability.

**Raw memory controls:** For the separate local Figure 6 panel, all curves use
the raw local seed-22 run's public GraphCast trajectory as a common denominator.
Five raw carry checkpoints at main step 2,000 give mean reductions
13.5094% ± 0.4224 pp (rollout) and 20.1877% ± 0.7070 pp (day 10).
The same raw seed-22 weights give carry/carry 13.5836% / 20.3820% and
carry/reset 1.5077% / 8.6014%. The separately reset-trained seed-22 raw control
gives 9.2097% / 13.0950%. Both control curves have n=1; these are not SWA
comparisons and do not yield a five-seed matched memory contrast. Inference
reset still introduces training/forecasting mismatch. The original SWA panel
retains e002's 5.2614 pp rollout advantage and its paired date interval unchanged.
Negative early-lead reset results are displayed in full, not clipped.

**Provenance and verification differences:** The local training recipe was
reconstructed from Appendix A.1, public repository configs and the Lianghong
handoff because original Della configs and the frozen training environment
were unavailable. Local seed 22 is fresh retraining, not the historical selected
checkpoint. Source host is the Lianghong OSS workspace
`/ossfs/workspace/weather/Weather_Global`; retained local provenance files are
authoritative locators/hashes, not a claim that all remote files were re-read.
The training manifest records source commit
`fc6a306c32d8ea4a84abd117c5663bd933367638`, with exact source snapshot digest
`e1b04cad8ec370a928cf2388bd1cbe7a666da547247ae5568a64050ed989ddde`.
Manifest SHA256 is
`0037e2f16db6c01b98fa36a65b0635a2fe6d41b5222e0363e183c00990cc6412`.
The 2023 initialization manifest SHA256 is
`eca49e6da7005d6731b7e947055fc498b943d3907c2773a4be15f8e1bb51ab49`.

Local verification replaces stale pressure temperature/geopotential/humidity
fields at 2023-11-10 06/12/18 UTC using independent NCAR ERA5. The audit records
nine changed chunks, nine affected start/lead cells and no initialization-input
overlap. The repaired store metadata SHA256 is
`885f52678add0a90f88aac338d9c047d1d14a750cfb674ec5e169ba63606db27`;
truth-audit SHA256 is
`f331cf80baf2ee4d4cd5190999d9e96ca11cd0cbd099c51363a1e16a5f196cb4`.
Historical paper curves retain their historical evaluation; they are not added
as a sixth exchangeable seed. Finite between-run baseline differences remain
measured variations, not failure criteria.

**Evidence and reproducibility:**

- [Local seed export and qualifications](../../../results/LH_rebuttal_results/weatherdata20261111/03_seed_replication/README.md),
  [training manifest](../../../results/LH_rebuttal_results/weatherdata20261111/03_seed_replication/provenance/seed_training_manifest.json),
  [source snapshot](../../../results/LH_rebuttal_results/weatherdata20261111/03_seed_replication/provenance/seed_source_snapshot.json),
  [truth audit](../../../results/LH_rebuttal_results/weatherdata20261111/03_seed_replication/provenance/native_truth_audit.json).
- [Plot script](../../../scripts/analyze_models/plot_v4_seed_replication.py), SHA256
  `d37d1216c84816a0c689037b0ce0e0a247b1e5166ad19e70096893dc257e5aa6`;
  [summary, per-model checks and 23 source hashes](../../../plots/analyze_models/data/resolution_eval/v4_seed_uncertainty_20261113/summary.json).
  An explicit whitelist removes all full-GC fine-tuning rows before numerical
  parsing; none enter these figures or conclusions.
- [SWA endpoint table](../../../plots/analyze_models/data/resolution_eval/v4_seed_uncertainty_20261113/swa_seed_endpoints.csv),
  [raw/control endpoint table](../../../plots/analyze_models/data/resolution_eval/v4_seed_uncertainty_20261113/raw_memory_endpoints.csv),
  [Figure 4](../../../Overleaf/arxiv_pics/fig4_forecast_skill_v4_seeds.pdf),
  [Figure 6](../../../Overleaf/arxiv_pics/fig6_memory_policies_v4_seeds.pdf).

Both 6-inch figures use 10-point text and were visually inspected. Main means
and SDs exactly reproduce the independent supplied summary; original paired
contrasts reproduce differences of the archived common-baseline reductions.
CSV/JSON outputs retain individual-seed curves, aggregates, checkpoints and
original intervals. The original `arxiv_versoin.tex` and figure assets were not
modified; the new assets are prepared for [v4](../../../Overleaf/v4.tex).
The final v4 build was checked on November 13: 10 main-text pages and 24
total, with no unresolved references/citations or overfull boxes. Source, new
figures, numerical exports, response draft and compiled PDF were pushed to
Non-Markovian-Weather/main as `a5df67a`. The PDF is
[`Overleaf/v4.pdf`](../../../Overleaf/v4.pdf); build hashes and checks are in the
[validation receipt](../../../artifacts/manuscripts/iclr2027_v4/seed_revision_validation.json).
No OpenReview or arXiv upload was performed. The inherited AI-use statement
retains an explicit author-correction TODO for the new analytical assistance.

**Next decision:** Present measured carry-seed variation and single-seed reset
limitations explicitly. Additional matched reset seeds would be needed to
estimate training-seed variation of the memory-policy contrast; no such new
training has been authorized or launched by this existing-data update.

<a id="pilot-selection-clarification"></a>
### 2026-11-13 — User clarification on the earlier 2023 pilot

The user confirms that the earlier eight-start 2023 forecast/spectral pilot was
not used for principal model selection or similar decisions. The recorded
principal model-selection comparison used the 2022 development sample; the
frozen 2023 protocol had recorded pilot selection influence as `none_reported`.
This is a user-confirmed clarification, not a retrospective claim that the
pilot was never run or that its dates did not overlap the annual evaluation.

For the response to Reviewer Jvjy, the user asks not to raise the pilot results.
Do not present prior pilot inspection as evidence of model selection on 2023.
Keep the historical pilot outputs, frozen protocols, and existing provenance
disclosures intact. No manuscript or experiment artifacts were changed by this
clarification.

<a id="pilot-wording-cleanup-20261114"></a>
### 2026-11-14 — Remove irrelevant pilot-exposure framing from v4

The user reaffirmed that the earlier pilot did not influence model selection
or similar decisions and rejected the main-text sentence about prior 2023
exposure. That wording conflated running numerical checks with tuning against
test results. The earlier instruction to retain manuscript disclosures is
superseded for the active manuscript and reviewer response; historical logs,
frozen protocols and numerical artifacts remain preserved as internal evidence.

Removed exactly three related passages from `Overleaf/v4.tex`: the main
evaluation paragraph's pilot cross-reference, the Appendix A.3 excluded-pilot
sensitivity result and exposure caveat, and the Appendix C pilot-exposure
sentence. Retained the 2022 model-selection / 2023 evaluation split, all 122
forecast dates, paired bootstrap specification, other block-length/quarter
sensitivities, numerical repeatability, forecast-window overlap and data-repair
details. Reviewer-response drafts already avoid pilot framing and need no edit.
There are no new runs, exclusions, score changes or uncertainty calculations.

The previous source/PDF are preserved under
`artifacts/evaluations/v4_pilot_wording_cleanup_20261114/before/`.
The manuscript was rebuilt; source/PDF hashes, reference checks, visual review
and synchronization are recorded in the adjacent `validation.json`.
Build and visual review passed (35 pages); the corrected source and compiled
PDF were pushed to the linked manuscript repository on `main` at
`e04115a6cd35912bcb856def62b81f072234f5fa`. The parent repository was not pushed.

<a id="two-memory-policies-20261114"></a>
### 2026-11-14 — Focus the manuscript on the matched training-policy comparison

The user decided to omit the carry/reset inference-only diagnostic from the
paper. Its intervention mixes state removal with a training/forecasting
mismatch and is peripheral to the matched carry/carry versus reset/reset
comparison. This is a presentation decision, not deletion or reclassification
of the historical measurements.

Removed carry/reset from active `Overleaf/v4.tex`, Table 1, Figure 6, conclusions
and appendix descriptions. Updated both reviewer-response formats to acknowledge
the original mismatch concern and center the matched training-policy comparison.
The original arXiv source, full numerical exports, old figure and evaluation
artifacts remain unchanged. No forecasts, training or bootstrap draws were run.

The new figure directly subsets the validated November 13 exports. Retained
original SWA curves and paired date intervals are exact copies; the local raw
carry mean/SD CSV is copied byte-for-byte and independently checked against
the five retained seed curves. Both panels retain their own common baseline;
SWA/raw checkpoints and date/seed uncertainty remain distinct. The figure keeps
its 6 x 4.05 inch canvas and 80% manuscript width. Shared y limits are now
-3% to 27%, covering all retained points and bands. The single reset seed
limitation remains explicit.

Unchanged endpoints: original carry/carry and reset/reset rollout reductions
14.42%/9.16%, paired contrast 5.26 pp [4.70,5.86]; day-10 reductions
21.66%/13.64%, contrast 8.02 pp [7.20,9.05]. Local raw carry means are
13.51% +/- 0.42 pp and 20.19% +/- 0.71 pp, versus reset/reset 9.21% and
13.09%.

Reproduction:
[`scripts/analyze_models/plot_v4_memory_two_policies_20261114.py`](../../../scripts/analyze_models/plot_v4_memory_two_policies_20261114.py).
Evidence: [source/curve manifest](../../../plots/analyze_models/data/resolution_eval/v4_memory_two_policies_20261114/manifest.json),
[portable paper archive](../../../Overleaf/archive/Graphs/v4_memory_two_policies_20261114/).
Prior manuscript/response files are preserved in
`artifacts/evaluations/v4_memory_two_policies_20261114/before/`; build, visual
review and publication details are recorded in its adjacent `validation.json`.

**Completion:** manuscript and response builds and visual checks passed.
The main text remains 10 pages (35 total); the response is two pages. Both
compiled PDFs are updated in `Overleaf/`. The linked manuscript repository was
pushed on `main` at `5d7463f323ae6bdc7c01dd5173b277c37320c088`. The parent
Weather_global repository was not committed or pushed.
