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
Figure6 has pointwise95% bands for all policies. All21 other figure files
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
