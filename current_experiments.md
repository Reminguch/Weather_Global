# Current experiments

## User-authorized production start after data preparation — 2026-09-20 16:36 EDT

The user explicitly said “我觉得正式训练可以直接跑了，”. This overrides the
previous requirement to wait for additional full-statistics GPU smoke, numerical
and performance reports. Full cache verification and full training statistics
remain mandatory inputs. This is an explicit startup-policy exception for the
active scatter_v3 experiment, not a passing full-statistics test report.

- All four independent detailed pilot tests passed 15/15 checks. All four long
  process-resume comparisons now also passed: update50-to-100 pretraining and
  episode3-to-6 k=20 fine-tuning, exact checkpoint/metric equality, and interrupted
  log recovery. Matched-hardware aggregate14199959 completed successfully.
- New active formal pretraining jobs are **14200959** w128/di16, **14200960**
  w128/di32, **14200961** w256/di16, and **14200962** w256/di32. Each is one A100
  80GB PCIe, 8 CPUs, 64 GiB host RAM, gpu-short, 24h maximum. Actual Slurm state
  was verified as PENDING, with only `afterok:14198504`.
- Full cache verification/statistics14198504 remains pending Priority, gpu-short,
  12h. Therefore formal training has NOT started. Prepared data and cache shards
  are complete, but full training normalization must finish before training can
  execute. No pilot statistics are promoted to production.
- Existing additional checks continue independently after14198504: numerical
  14199428, profile14199429, detailed14198731/14199422/14199424/14199426, and
  aggregate14199427. They no longer block training. All GPU tests retain
  gpu-test and <=1h. No extra smoke jobs were added for this scheduling change.
- Original formal jobs14199760–14199763 and old release14199430 were cancelled
  only after replacement jobs, dependencies and records were verified. The
  original queue records are retained with superseded_by/cancellation metadata.
- Executed core source, model, hyperparameters and numerical execution policy
  remain unchanged, source ID
  `02afdad970c37dec9cc67b34a780be0621aba1f24f8773e7917a7ad2aeee2ec1`.
  A separate immutable operational runner imports the frozen worker and replaces
  only its requirement to wait for numerical/profile reports with the explicit
  user policy and required full-data checks. It records the policy and runner
  hashes alongside each actual training startup. Validation, selected-parent
  transfer, optimization and checkpointing use the unchanged frozen worker.
- The operational runner preserves explicit checkpoint continuation at 24h slices
  until all 20 pretraining passes and then 2,000 k=20 fine-tuning updates finish.
  64 GiB resources are based on completed full-size-model pilots; no unmeasured
  full-statistics profile or estimated full runtime is claimed.
- Policy: `pipeline/authorized_training_policy.json`, SHA256
  `f8ea0cf12b42c25deeb72b3fa4e489a80b4e418984a51c17c32d91a8d5d143e3`.
  Runner: `pipeline/authorized_training_v1/run_neuralgcm_authorized_training.py`,
  SHA256 `4472e025c8b5ebdf5778b9c8e7fc8e88419729d43814ee44ec3ad7d13201f5b8`.
  Current graph: `pipeline/authorized_training_chain.json` and updated
  `pipeline/training_chain.json` / `pipeline/cache_chain.json`.
- Validation: 10 targeted CPU tests passed for content-bound authorization,
  rejecting failed/changed smoke evidence, requiring full statistics and matching
  cache receipts, checkpoint progress before continuation, parent-stage receipt
  before fine-tuning, and Slurm resource/dependency construction. The frozen
  runner's validate-policy command also passed against the real experiment and
  its four completed detailed/resume reports.

## Smoke remaining work — 2026-09-20 16:20 EDT

- All four independent detailed pilot tests now PASS all 15 checks. The two w256
  jobs 14199423/14199425 finished in 577.02/555.97 seconds. No additional pilot
  detailed rerun is queued.
- Long-resume reports for w128/di32 and w256/di32 now PASS both pretraining
  (100 updates, separate-process restart at50) and k=20 fine-tuning (six episodes,
  separate-process restart at3), with exact state/metrics and log-tail recovery.
- Only two independent GPU smoke jobs remain active: matched-SXM pretraining
  replays 14199953 (w128/di16) and 14199956 (w256/di16). All four fine-tune replay
  executions have completed. The two remaining CPU comparisons 14199954/14199957
  and aggregate14199959 wait for these GPU replays; no overall pass yet.
- Production preflight still requires full cache verification/training statistics
  job14198504, currently pending Priority. After it succeeds, the already queued
  four detailed jobs14198731/14199422/14199424/14199426 rerun with production
  statistics, along with numerical14199428 and profile14199429. These six GPU
  jobs all use gpu-test, <=1h. Current completed pilot statistics covered only
  96 training records, so they cannot certify production normalization.
- Formal jobs14199760–14199763 remain queued behind release14199430 and all
  mandatory gates. Current queued scope is two active independent GPU replays
  plus six downstream GPU checks, with CPU comparison/aggregation jobs.

## GPU smoke status and matched-hardware resume — 2026-09-20 16:16 EDT

This is the current status; older dated job graphs below are historical. Active
experiment remains `res2p8_w128_256_di16_32_train2015_2021_20260920_scatter_v3`,
source ID `02afdad970c37dec9cc67b34a780be0621aba1f24f8773e7917a7ad2aeee2ec1`.

- All four continuous 100-update pretraining references completed. Independent
  detailed smoke passed all 15 checks for w128/di16 and w128/di32. The two w256
  detailed jobs 14199423 and 14199425 are running; w256/di16 has passed the short
  fine-tune resume check and is doing final validation. Their pilot statistics
  remain isolated from production statistics.
- Strict cross-process update-50-to-100 replay is exactly equal for w128/di32
  (PCIe to PCIe) and w256/di32 (SXM to SXM), including parameters, optimizer,
  recurrent memory, RNG, cursor and metrics excluding elapsed time. The first
  w128/di16 replay moved from A100 SXM to A100 PCIe: parameter max absolute
  difference 1.1920928955078125e-7, memory max absolute difference
  3.573298454284668e-5; RNG/cursor matched but metrics diverged at update51.
  This is NOT an exact-resume pass. Evidence is in
  `checks/resume_smoke_100_v3/pretrain_comparison_in_progress.json`.
- Both di16 pretraining replays had unmatched GPU variants. New GPU-test jobs
  **14199953** (w128/di16) and **14199956** (w256/di16) rerun only the resumed
  half on SXM, matching their saved reference hardware. They preserve the exact
  same executed runner (SHA256
  `6ea671a4ac242e4d0b8ed768538a039aa1c875835d7891e10f1c4b7988b41c6f`),
  immutable source and strict equality criteria. Previous outputs are retained.
  New output root is `checks/resume_smoke_100_matched_hardware`; unchanged
  reference and fine-tune artifacts are linked there, not copied or relabeled.
- All four six-episode k=20 fine-tune references completed. Fine-tune replay
  jobs are 14198584/14198594/14198599/14198589 for
  w128di16/w128di32/w256di16/w256di32. At this observation, 14198584 completed,
  14198589 is running, and the other two are pending. Final strict fine-tune
  comparisons remain outstanding; phase completion alone is not certification.
- New per-arm CPU comparison jobs are **14199954, 14199955, 14199957, 14199958**;
  aggregate **14199959**. Training release **14199430** was actually rewired in
  Slurm to require `afterok:14199428:14199429:14199427:14199959`.
  Old comparison jobs 14198585/14198590/14198595/14198600 and old aggregate
  14198601 were cancelled only after the new dependencies were installed.
  Authoritative updated graph: `pipeline/resume_smoke_chain.json`.
- Pending production, full-statistics GPU gates and remaining fine-tune resume
  jobs now require `a100&gpu80&pcie`; future training continuations inherit this
  from `pipeline/config.json`. All 59 production A10080 nodes observed have the
  PCIe feature. This avoids the observed hardware-variant change during strict
  replay; it is not a claim that hardware alone has been isolated as the cause.
  Scheduling audit: `pipeline/hardware_policy_pcie.json`.
- All GPU smoke jobs still use **gpu-test, maximum one hour**; no explicit
  partition. At 16:15 EDT three GPU tests were running, with four ready tests
  pending for `QOSMaxJobsPerUserLimit`. Full cache verification/statistics
  **14198504** remains `gpu-short`, 12h, pending for Priority. Full-statistics
  numerical/profile/detailed gates therefore have not started.
- Formal GPU training jobs **14199760–14199763** remain submitted, gpu-short,
  24h maximum, and pending on release14199430. No formal training has started,
  and neither pilot smoke nor an approximate resume comparison can bypass the
  remaining gates. Complete smoke/resume certification is still outstanding.

## Formal GPU training jobs submitted — 2026-09-20 16:04 EDT

The user explicitly requested submitting formal training now while smoke/resume
checks finish. All four actual GPU pretraining jobs are now submitted:

- `r2p8_w128_di16`: **14199760**.
- `r2p8_w128_di32`: **14199761**.
- `r2p8_w256_di16`: **14199762**.
- `r2p8_w256_di32`: **14199763**.

Each requests one A100 80GB, 8 CPUs, 64 GiB host RAM, **gpu-short, at most
24 hours per job**. Slurm verified all four as `PENDING (Dependency)` with
`afterok:14199430`. They retain every mandatory full-data/numerical/detailed/resume
gate through that release job. They have **not started running**.

The 64 GiB reservation is above the measured full-size pilot requirement
(5.5–5.9 GiB peak host RSS for the completed detailed w128 arms); all four
100-update references succeeded in the same 64 GiB allocations. Steady cached
updates measured 4.76–6.14 seconds. The release job will still write the full-profile
runtime budget before GPU execution. Initial pretrain submissions use the exact
existing `pretrain-<run>-000` keys, so the automatic release reuses these IDs;
it does not submit duplicates. `training_chain.json` is already written in the
same immutable/idempotent format that the release expects. Full 20-pass
pretraining retains explicit checkpoint continuation, then submits the configured
2,000-update k=20 fine-tuning stage automatically.

Authoritative records: active experiment `pipeline/training_chain.json`,
`pipeline/prequeued_training.json` (includes scheduler-verified resources and
measurement evidence), and `pipeline/jobs/pretrain-*-000.json`. Previous
statements that formal GPU jobs were not submitted describe the earlier state.


## Active NeuralGCM smoke/resume status — 2026-09-20 15:55 EDT

- Active experiment: `res2p8_w128_256_di16_32_train2015_2021_20260920_scatter_v3`, source ID `02afdad970c37dec9cc67b34a780be0621aba1f24f8773e7917a7ad2aeee2ec1`. Earlier experiment/job sections below are historical. Formal training GPU jobs are **not yet submitted**; `pipeline/training_chain.json` does not exist.
- Formal automatic submission control **14199430** is queued with hard dependencies on numerical **14199428**, profile **14199429**, full-statistics detailed aggregate **14199427**, and independent long-resume aggregate **14198601**. No test failure may release training. All GPU smoke/preflight jobs use **gpu-test, <=1 hour**. This QoS permits 3 simultaneous jobs and 25 submitted jobs per user; remove superseded jobs before exceeding the pending quota.
- Prepared data remain complete. All **487/487 k=1 cache archives** have now been generated. Actual full-cache verification/statistics job **14198504** is pending for Priority, with **12h, gpu-short**; no remaining cache dependency. Older cache dependencies aged out after successful completion. No verification receipt was forged or relabeled.
- Fixed exact coordinate comparison: GPU host latitude metadata differs by only 1.3877787807814457e-17 radians; runtime permits angular roundoff <=1e-14 radians while checking exact keys/shapes/levels and finiteness. Longitude and pressure levels matched exactly.
- Diagnosed severe backward slowdown using a read-only Python/native profiler and optimized HLO. The original deterministic scatter lowering generated 33 while loops, including 303104-trip interpolation gradient loops. Enabling `--xla_gpu_enable_scatter_determinism_expander=true` **with nondeterministic ops still excluded** reduced this to one 6-trip loop; measured one backward call 0.2388s. New source pins both flags. See prior gridfix_v2 `checks/scatter_diagnostic.json` and the two HLO files. No architecture, optimizer or training budget change.
- Four v3 pilot jobs **14197167–14197170** passed real 40-step zero equivalence, 24-record cache/live loss, gradient, memory and update parity (all max_abs=0), and chunk carry, then stopped at an overly strict per-element independent-AD comparison. Real diagnostic **14198505** found both AD methods individually bitwise-repeatable, relative L2 difference 2.9667e-7, cosine 0.9999999999999587, max per-layer error-budget ratio 0.06044. Only the independent AD comparator now uses per-leaf max-error budget (atol 1e-6 + 1e-5 * leaf scale), global relative L2 <=1e-5, cosine >=.99999. Exact resume comparisons remain strict. Diagnostic: current experiment `checks/bptt_diagnostic/report.json`.
- Corrected detailed pilot tests reuse verified isolated 96-record pilot artifacts. **14198730** w128/di16 and **14198732** w128/di32 have passed **all 15 checks**, including exact short pretrain/fine-tune resume, real k=20 corrected-state feedback, cold/warm validation, and unchanged frozen backbone. Runtime 563.69s and 535.38s. **14199423** w256/di16 and **14199425** w256/di32 remain queued. Output `checks/quick_smoke_gpu_test_v4`; copied runners have their own hashes and preserve the executed core source snapshot.
- User explicitly requested tens/hundreds of training updates plus resume. Added `scripts/training/smoke_neuralgcm_resume.py`: 100 real cached optimizer updates (25 passes over a 96-record pilot, BPTT24), checkpoint at update50 inside a segment (epoch12/chunk2/nonzero Mamba memory), reload in a **different Slurm process**, continue to100, and compare exact params/optimizer/memory/RNG/cursor and the entire loss/sampling log. Inject valid-uncommitted and torn JSON log tails and require exact archival/reconciliation. Also run 6 closed-loop k=20 episodes (120 predicted steps), resume in a new process after episode3, and require exact final state/log equality.
- All four continuous **100-update references completed successfully**: w128/di16 560.81s, w128/di32 634.13s, w256/di16 692.52s, w256/di32 701.70s. At 15:55 EDT: resumed w128/di16 was at update59; resumed w256/di32 at54; w128/di16 six-episode fine-tune reference also running. Full long-resume comparison is **not yet passed**.
- Long-test job map (pretrain reference / pretrain resume / fine reference / fine resume / CPU compare): w128/di16 **14198581/14198582/14198583/14198584/14198585**; w256/di32 **14198586/14198587/14198588/14198589/14198590**; w128/di32 **14198591/14198592/14198593/14198594/14198595**; w256/di16 **14198596/14198597/14198598/14198599/14198600**. Aggregate **14198601**. Artifacts `checks/resume_smoke_100_v3/`; authoritative graph `pipeline/resume_smoke_chain.json`.
- Final detailed tests using full training statistics are **14198731**, **14199422**, **14199424**, **14199426**, dependent on verification14198504; aggregate14199427. These remain distinct from isolated pilot statistics. Authoritative graphs: `pipeline/cache_chain.json`, `detailed_smoke_chain.json`, `quick_smoke_v4.json`.
- CPU checks: 33 grid/compiler/launcher/gate/resume cases passed before the last test-runner adjustment; then 20 independent-gradient/gate/log-recovery cases passed. Old gridfix_v2 verification, six GPU gates, aggregate and release were cancelled after replacement/reconciliation. Failed submission intents are preserved as `.failed_dependency.json` / `.failed_limit.json`; retries were reconciled against actual Slurm accounting.
- At 15:55 EDT, remaining independent pilot+long-resume work was estimated at **30–45 minutes**, assuming no new failures and similar GPU-test throughput. This estimate does **not** include full-statistics final checks: their upstream gpu-short verification job is still queued, so a reliable formal-training start time is unavailable.


Persistent user preference (2026-09-20): **all GPU smoke tests use `gpu-test`, normally at most one hour per job**, including numerical/performance preflight. This is recorded in `/home/sh4809/weatherforecast/AGENTS.md`. Run a representative-data smoke independently of full-cache preparation; keep pilot statistics separate. Cache production, dataset verification/statistics and production training are not reclassified as smoke jobs. Migration is complete; see the active revision and job graph below.

Latest scheduling override (2026-09-20): at the user's request, verification/statistics job **14190503** now has a **12-hour** limit and remains `gpu-short`. It was previously reduced to 24 hours. The measured cache lane completed 744 records in 993.44 seconds; a full verification/statistics runtime has not yet been measured, so 12 hours is a reservation estimate. All dependencies and smoke gates are unchanged. The workspace launcher caps future verification jobs at 12 hours; production training still uses slices of at most 24 hours. See `pipeline/adjustments/verify_12h.json`.

## Active NeuralGCM revision and GPU smoke status — 2026-09-20

- The active experiment is `res2p8_w128_256_di16_32_train2015_2021_20260920_gridfix_v2` under the same scratch checkpoint parent. Source ID: `21c640af8688938572976f8f0b7f9f2fbfdf4dc1ba23136138d9d17f2f3b45ea`. Older sections below are historical; their superseded job IDs must not be treated as current.
- Independent pilot jobs 14194059–14194062 DID run (13:36/13:39 EDT) and failed after about 2 minutes at an exact floating-point grid comparison, before the detailed checks. They are not passes. Their 96-record cache/statistics preparation succeeded and is isolated from production.
- GPU diagnostic 14194944 completed successfully in 18 seconds. Longitude and pressure levels matched exactly; Gaussian latitude differed by only 1.3877787807814457e-17 radians on the GPU host. Repeated model loads reproduced the same discrepancy. CPU-host coordinates matched exactly.
- Fixed `runtime.validate_data_grid`: exact keys, one-dimensional shapes, finite coordinates, exact pressure levels; angular coordinates accept only absolute roundoff <=1e-14 radians, rtol=0. No grid resampling, model arithmetic, loss, or hyperparameter change. Eleven new coordinate regression cases pass; grid + launcher + smoke-gate tests pass 22/22. The previous launcher + smoke-gate + watcher suite also passed 22/22.
- New independent smoke jobs: **14195396** w256/di32, **14195397** w128/di16, **14195398** w256/di16, **14195399** w128/di32. All request one A100 80GB, 8 CPUs, 64 GiB RAM, **gpu-test, maximum 1 hour**. Latest observation: first three running, fourth waiting on QOSMaxJobsPerUserLimit (3 simultaneous test jobs). The three running jobs completed pilot preparation and passed `zero_residual_40`: corrected zero-branch and original NeuralGCM outputs match exactly at leads 1, 20 and 40 (max_abs=0). The full detailed suite is NOT yet certified passed. Logs: new experiment `checks/quick_smoke_gpu_test_v2/logs/`.
- The 13 outstanding cache jobs 14189205–14189217 are preserved with their 45-minute limits; completed cache content is reused unchanged. Artifact-producing core modules are hash-identical between revisions. The new manifest explicitly references the same prepared data/cache/statistics paths. No receipt is relabeled: the new revision will run real complete-cache verification and fit full training statistics before new GPU gates.
- New full-data job graph: verification/statistics **14195485** (12h, gpu-short) waits for remaining cache lanes; numerical **14195490**, profile **14195491**, and detailed four-arm checks **14195492–14195495** all use gpu-test/1h and wait for verification. Aggregate **14195496** requires all four detailed checks. Training release **14195497** requires numerical + profile + aggregate. Formal training has not begun.
- Old verification 14190503, preflights 14194243–14194244, detailed jobs 14194245–14194248, aggregate 14192302, and release 14189221 were cancelled after replacement submission. Audit: old experiment `pipeline/superseded_by_gridfix_v2.json`; new experiment `pipeline/revision.json`, `cache_chain.json`, `detailed_smoke_chain.json`, `quick_smoke_gpu_test.json`.
- All GPU smoke tests must keep using gpu-test (normally <=1 hour); see `/home/sh4809/weatherforecast/AGENTS.md`. Pilot statistics/checkpoints cannot release production. Do not mutate executed frozen snapshots in place.

## NeuralGCM detailed smoke gates and 24-hour limit — 2026-09-20

- User requested a detailed real-model smoke test before spending compute on production training, and a maximum 24-hour job limit.
- Added `scripts/training/smoke_neuralgcm_detailed.py`, executed against the existing immutable production snapshot. The four real GPU jobs are **14192298** (w128/di16), **14192299** (w128/di32), **14192300** (w256/di16), and **14192301** (w256/di32). Each requests one A100 80GB, 8 CPUs, 64 GiB host RAM, `gpu-short`, and at most four hours.
- These tests are **submitted, not yet passed**. They wait for complete cache verification and training-only statistics. Production training has not started.
- Coverage per arm: exact zero-residual backbone equivalence through 40 steps; 24-record fresh/cache loss, memory, gradient and optimizer parity; chunk carry; independent two-step BPTT gradient reference; stopped physical-input gradients; actual production pretraining over 96 consecutive records (four 24-step updates); exact resume from update 2; fixed-sample loss reduction over 12 updates; pretrained-parent transfer with fresh optimizer/RNG; actual cold and warm 20-step fine-tuning episodes; exact fine-tune resume; cold/warm 2022 validation and selection; unchanged frozen backbone parameters. Checkpoints are isolated smoke artifacts.
- Closed-loop audit: each of the 19 subsequent NGCM inputs must exactly equal the preceding corrected state, the branch must also receive that state, forcing must persist, every correction must be nonzero, future frames may only be targets, and encode counts must be one (cold) or five (four warm-up frames plus one origin). Cached k=1 pretraining still re-encodes truth per record and carries only Mamba memory.
- CPU regressions passed: 34 core tests + 11 watcher tests + 4 new fail-closed report-gate tests = 49 unique tests. Four report-gate tests were rerun successfully after final runner edits.
- Aggregate pass gate **14192302** depends on all four GPU tests. Training release **14189221** now requires `afterok:14189219:14189220:14192302`; its temporary user hold was replaced with these checked dependencies. Any failed test prevents automatic production training.
- Artifact root, relative to the active experiment: `checks/detailed_smoke_20260920_v1/`. Read `definition.json`, `jobs/`, `logs/`, per-arm `progress.jsonl` / `report.json`, and final `PASSED.json`. The copied runner has its own hash; the production source/config identities are unchanged. Durable job graph: `pipeline/detailed_smoke_chain.json`.
- Queue adjustment: completed cache jobs 14189202–14189204 each took 16:38–17:07. The remaining 13 cache jobs (14189205–14189217) now request **45 minutes**, retaining their existing IDs and completed work. Latest observation: these remain pending for Priority, with no running Slurm jobs for sh4809.
- Verification/statistics job **14189218** (26 hours, gpu-medium) was replaced with **14190503** (24 hours, gpu-short). Both numerical/profile jobs now depend on 14190503; the old pending job was cancelled after rewiring. The adjustment history is in `pipeline/adjustments/verify_24h.json`; `pipeline/cache_chain.json` and submission records were reconciled. Future launchers cap verification at 24 hours as well.
- Scheduler details: Della rejects explicit `--partition`; allow its submit policy to choose the partition. Completed dependencies may age out of the controller; verify their successful accounting/receipts and depend only on unfinished jobs when replacing a queued job.

## NeuralGCM 2.8-degree four-arm launch — 2026-09-20

- User scope: finish 2.8-degree data and train widths 128/256 × Mamba inner 16/32; 1.4-degree remains paused.
- Prepared data: all 108 months and 13,148 frames present, full SHA256 verification passed, merged manifest and resolution READY markers written. Split: 2015–2021 train, 2022 validation, 2023 test.
- Experiment: `/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/artifacts/checkpoints/neuralgcm_residual/res2p8_w128_256_di16_32_train2015_2021_20260920/`.
- Submitted bootstrap `14188996` and cache-release `14188997` (`afterok:14188996`). The durable pipeline measures two cache shards, builds/validates all k=1 records, fits train-only statistics, runs numerical/profile gates, then launches four 20-pass pretraining runs and their 2,000-update k=20 fine-tunes.
- Full production training has not yet passed its upstream gates at submission time. Follow `pipeline/jobs/`, `pipeline/cache_chain.json`, `pipeline/training_chain.json` and `logs/` for actual progress and later job IDs.
- Prior post-download smoke `14188458` failed from missing vendored GraphCast in its snapshot. The snapshot packager is fixed and the new bootstrap reruns that smoke with complete dependencies.
- The new bootstrap's full-data GPU smoke passed: live/cache and resume comparisons have zero maximum absolute difference, with frozen backbone parameters unchanged. Cache pilot generation follows in the same allocation.
- Bootstrap and cache pilot have completed. The automatic release submitted cache lanes `14189202–14189217` for 11,678 records / 487 shards, verification/statistics `14189218`, numerical checks `14189219`, GPU profile `14189220` and training release `14189221`. Four production pretraining IDs will appear only after all gates pass; each then submits its selected-parent fine-tune.
- Details and validation: `docs/experiments/neuralgcm_residual/IMPLEMENTATION.md`.

## v24 res1 baseline cache — 2026-09-07

- GPU pilot: `13570387`; four representative chunks (first/last train and validation), 96 predictions, online parity and exact FP32 disk round-trip checks. Allocation: one gpu80 GPU, 8 CPUs, 32G host RAM, 1:02 walltime.
- Automatic production-submission job: `13570390`, dependency `afterok:13570387`. It requires a successful pilot parity report and completed accounting, sizes memory/time from measured peak and throughput, submits an unthrottled eight-task GPU array, and submits final CPU verification after the whole array succeeds.
- Initial pilot `13570194` generated all four chunks but failed an incorrect bitwise fresh-model-repeat check. It took 4:43, peaked at 16.22 GiB batch MaxRSS, and measured about 5s per steady chunk. Corrected the disk-round-trip check to use original predictions, and enabled deterministic GPU reductions per the existing repository parity-test convention. The `1e-5` normalized online parity tolerance remains unchanged. Failed output is preserved at the pilot path with `_failed_13570194` appended; obsolete release job `13570198` was cancelled.
- Cache: `data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/`; pilot uses the same path with `_pilot` appended.
- Coverage: 424 train chunks (2015–2021), 60 validation chunks (2022), 11,616 predictions, 234.03 GiB FP32 payload. Existing incomplete-segment exclusions: 50 train and 18 validation anchors.
- Schedule: exact submitted training boundaries, 24 steps per chunk, four observed input windows followed by baseline-only AR feedback. No Mamba model/state is constructed by generation.
- Code/usage: `scripts/preprocessing/BASELINE_CACHE.md`; tests: nine passing in `tests/v24_Ilya/test_baseline_cache.py`. Full-data dry run and Python/shell checks passed.
- Logs: `logs/v24_baseline_13570387_4294967294.{out,err}` (Slurm non-array task placeholder); release logs `logs/v24_baseline_release_13570390.{out,err}`. Production IDs/resources will be written to the cache's `submission.json` and appended below; `READY.json` marks successful full verification.
- Budget evidence: res1 evaluation `12530971_3` completed in 40:15 with about 12.04 GiB batch MaxRSS; di64 training preflight `13435249_1` completed in 12:22 with about 20.97 GiB batch MaxRSS. Production uses the new baseline-only pilot's measurements.

## v24 res1 open-loop di64/BCG2 — 2026-09-07

- Training job: `13560724`, submitted with `scripts/experiments/train_v24_Ilya_res1_open_loop_di64_bcg2.slurm`.
- Config: `configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json`.
- Fresh residual branch using official GraphCast Small spatial-weight initialization; frozen baseline, baseline-only feedback, stateful Mamba1 di64/BCG2, state size 16, two layers, convolution width 4.
- Training: 10k updates, segment 96/BPTT 24/AR tail 20, all-step loss, AdamW LR 1e-4 with betas (0.9, 0.98), 200-update warmup, cosine to 1e-5, seed 22. Checkpoints every 1k; fixed validation every 2k.
- Resources: one gpu80 GPU, 8 CPUs, 288G host RAM, 42h. Evidence: completed res1/di16 jobs `13435248_6/7` took about 26.2h with about 224 GiB batch MaxRSS; di64/BCG4 preflight `13435249_1` measured 10.06s per steady update.
- Output: `artifacts/checkpoints/v24_Ilya/res1_open_loop/mamba1_di64_bcg2_all24_lr1em4_b1p9_b2p98_cos10k_seed22/`.
- Logs: `logs/v24_r1_open_di64_bcg2_13560724.{out,err}`; GPU telemetry uses the same prefix with `_gpu.csv`.
- This initial reference run computes GraphCast online; offline residual caching remains subsequent work.

## v24 res1 optimizer matrix — 2026-09-04

- Goal: stabilize the successful full-loss/SWA recipe while comparing Mamba1 and legacy Haiku initialization.
- Models: `{legacy,mamba1} × {di16/BCG1, di64/BCG4}`; `d_state=16`, two temporal layers, zero output init.
- Optimizers: peak LR `{1e-4,5e-5}` × Adam `(β1,β2)={(0.9,0.999),(0.9,0.98),(0.8,0.98)}`.
- Common training: res1, all 24 loss steps, 10k updates, seed 22, 200-step warmup, cosine decay to 0.1× peak LR, AdamW decay `1e-4`, clip norm 1, checkpoints every 1k.
- SWA windows: 2k–8k, 4k–8k, and 6k–10k.
- Evaluation: screen checkpoints 2k/4k/6k/8k/10k plus all SWAs on 8 anchors; exact matched 32-anchor evaluation for the best checkpoint and SWA from each run.
- Scale: 24 full runs. `di64/BCG4` is released only after two architecture preflights succeed.
- Resources: di16 array uses A100-80GB, 8 CPUs, 224G, 36h; di64 array uses A100-80GB, 8 CPUs, 256G, 48h.
- Selection: exact original GraphCast rollout-loss reduction; also track clipping rate, gradient percentiles, late drift, and SWA gain. Re-run the top two settings with two additional seeds.
- Training jobs: `13435248_[0-11]` di16/BCG1; `13435249_[0-1]` di64/BCG4 preflight; `13435250_[0-11]` di64/BCG4 full matrix (`afterok:13435249`).
- Evaluation jobs: `13435335_[0-11]` di16 and `13435336_[0-11]` di64; each task uses `aftercorr` to screen and exact-evaluate its matching successful training run.
- Requested early-SWA comparison: built six-member SWA(2k,3k,4k,5k,6k,7k) for the ten di16 runs that had reached 7k; exact matched 32-anchor/40-lead evaluation is `13548842_[0-9]`, followed by summary job `13548843`. Outputs go to `results/v24/res1_optimizer_swa2k7/` and compare day-5/day-10 2m-temperature RMSE and exact allvars GraphCast-loss improvement with the incumbent V22 legacy SWA(2k--8k).
