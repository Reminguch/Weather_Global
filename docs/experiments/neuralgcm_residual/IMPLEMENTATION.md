# NeuralGCM implementation and operations

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


Persistent user preference (2026-09-20): **all GPU smoke tests use `gpu-test`, normally at most one hour per job**, including numerical/performance preflight. This is recorded in `/home/sh4809/weatherforecast/AGENTS.md`. Run a representative-data smoke independently of full-cache preparation; keep pilot statistics separate. Cache production, dataset verification/statistics and production training are not reclassified as smoke jobs. Migration of the previously queued four-hour checks is in progress; consult live Slurm state and pipeline submission records.

Latest scheduling override (2026-09-20): at the user's request, verification/statistics job **14190503** now has a **12-hour** limit and remains `gpu-short`. It was previously reduced to 24 hours. The measured cache lane completed 744 records in 993.44 seconds; a full verification/statistics runtime has not yet been measured, so 12 hours is a reservation estimate. All dependencies and smoke gates are unchanged. The workspace launcher caps future verification jobs at 12 hours; production training still uses slices of at most 24 hours. See `pipeline/adjustments/verify_12h.json`.

## Detailed GPU smoke gate, 2026-09-20

The user requested full-size smoke coverage for all four 2.8-degree configurations
before production training. The original small real-ERA5 smoke passed, but it did
not cover all four architectures, a complete 96-record training segment, or
trained cold/warm closed-loop episodes. It is not sufficient by itself to release
production training.

The new runner is `scripts/training/smoke_neuralgcm_detailed.py`. It imports the
existing immutable production source and loads the exact run configurations,
checkpoint, verified cache and training-only normalization. Only test budgets are
shortened: one 96-record pretraining segment, two fine-tuning updates, and one
2022 validation origin. Architecture, optimizer, 24-step BPTT and 20-step live
rollouts are unchanged. No production checkpoint is written by this runner.

Every arm must pass zero-residual equivalence at 1, 20 and 40 steps; cached/live
24-record loss, recurrent-state, gradient and optimizer parity; chunk carry;
an independent direct-autodiff two-step BPTT reference; zero physical-input
gradients; four real pretraining updates; exact midpoint resume; a 12-update
fixed-sample loss-decrease check; compatible parent transfer; cold and warm
20-step fine-tuning updates; exact fine-tuning resume; cold/warm validation and
selection; finite FP32 training state; and unchanged NeuralGCM weights.

The live-loop audit checks all 19 feedback transitions exactly. NGCM and the
branch must receive the preceding **corrected** state, every correction must be
nonzero, forcing must remain fixed, and cold/warm initialization must use the
expected observed timestamps and memory. The cold episode encodes once, and the
warm episode encodes four history frames plus the origin. The gradient policy
remains `closed_loop_sg`: corrected physical feedback in the forward pass,
stopped gradients through that feedback, and recurrent-memory BPTT.

GPU jobs **14192298–14192301** correspond to w128/di16, w128/di32, w256/di16 and
w256/di32. Each requests one A100 80GB, eight CPUs, 64 GiB host RAM, `gpu-short`
and a four-hour limit. All depend on verification/statistics **14190503**.
The aggregate CPU gate **14192302** requires complete, passing, GPU-backed
reports for all four arms with matching source, config and runner identities.
Training release **14189221** now requires both previous numerical/profile gates
(**14189219**, **14189220**) and **14192302**. A missing or failed detailed smoke
cannot release training. **The detailed GPU tests are submitted and pending;
they have not yet passed, and production training has not started.**

Artifacts are under `checks/detailed_smoke_20260920_v1/` in the active experiment:
`definition.json`, a separately frozen runner in `code/`, submission intents in
`jobs/`, logs, per-arm progress and final reports, then `PASSED.json` only after
all four reports validate. `pipeline/detailed_smoke_chain.json` records the new
job graph. CPU validation passed 49 unique tests (34 core, 11 watcher, four new
report-gate regressions); the final four gate tests were also rerun successfully.

The user's 24-hour limit is now reflected in the queued verification job:
**14190503**, `gpu-short`, 24 hours, replaces **14189218**, which requested 26
hours and was cancelled after downstream dependencies were updated. The remaining
13 cache jobs now request 45 minutes, based on three completed lanes taking about
17 minutes each. These are reservation limits, not runtime predictions. Production
training already uses checkpoint-resumed slices of at most 24 hours. The frozen
production source was preserved; the current workspace launcher also caps future
verification reservations at 24 hours.

## 2.8-degree training launch, 2026-09-20

The 2.8-degree prepared data is complete: **108 months, 13,148 six-hour frames,
2015-01-01 00Z through 2023-12-31 18Z**, 78,784,027,301 compressed frame bytes.
All frame hashes passed the full-data watcher. The merged manifest and both
resolution-specific READY markers have been written to the scratch prepared
root below. The dataset identity is
`f51534c365518f4c5ecf7b8b71a8182e253365cdabf6511345122219e0236e4a`.
1.4-degree staging remains paused with its 319 existing frames retained.

The user authorized training the four 2.8-degree width/inner-width combinations
first. The active experiment is now
`/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/artifacts/checkpoints/neuralgcm_residual/res2p8_w128_256_di16_32_train2015_2021_20260920/`.
Its immutable source identity is
`c47ff8c21f8298339ec3ba4d00e6aa54810475bb8bfd7203e5ac435ccecd036d`.
It explicitly selects `active_resolutions: [res2p8]`, retains all four arms,
uses existing official checkpoint files in home, and writes caches and training
artifacts to scratch. The older dual-resolution `_v2` experiment is historical
and points to the removed home prepared-data path.

The automatic post-download smoke `14188458` failed during import because its
old snapshot omitted vendored GraphCast. Snapshot staging now includes both
GraphCast and NeuralGCM, with a regression test. The new experiment snapshot
contains those dependencies and reruns the real full-data smoke in bootstrap.
No failed smoke report is reused as a successful gate.

Bootstrap's full-data GPU smoke has now passed on origin `2015-01-02T00`.
Cached/live loss, memory, gradients and optimizer updates agree exactly, as do
checkpoint state and resumed continuation. Frozen parameters remained unchanged.
The report is `checks/full_data_smoke/report.json` in the active experiment.
This smoke remains separate from the seasonal numerical and resource gates.

Initial submitted jobs are GPU bootstrap **14188996** and CPU cache release
**14188997**, with `afterok:14188996`. Bootstrap performs the real-data smoke,
checkpoint inspection, cache plan and two measured cache shards. The release
job schedules 16 disjoint cache lanes, then full cache verification and train-only
statistics, seasonal numerical checks and a width256/di32 GPU profile. A CPU
release job starts all four training arms only after these gates pass. Subsequent
job IDs, dependencies and resource calculations are recorded under `pipeline/`.
All GPU stages request the same `a100&gpu80` constraint.

Bootstrap and cache release have completed successfully. The cache plan contains
11,678 eligible train/validation records in 487 shards. Two measured shards took
49.25 seconds for 24 records and 38.07 seconds for 15 records, with 2.21 GiB peak
host RSS. The release submitted **14189202–14189217** (16 GPU cache lanes,
8 CPUs, 16 GiB host memory and a two-hour limit each), full verification/statistics
**14189218**, seasonal numerical checks **14189219**, large-arm resource profile
**14189220**, and training release **14189221**. The latter three retain their
successful-upstream dependencies. These are preparation jobs; the four production
pretraining job IDs will be written to `pipeline/training_chain.json` only after
the gates pass. Each successful pretraining arm then submits its own fine-tune.

Training follows the README: 20 full passes over eligible 96-record segments,
24-record BPTT, then each arm's selected cold-2022 parent transfers to 2,000
twenty-step fine-tuning episodes. Resource estimates include measured validation
costs. Individual jobs are at most 24 hours; a timed slice resumes explicitly
from its latest committed checkpoint and stops if no progress was checkpointed.
Failed numerical or training jobs do not release downstream work. The 2023 test
set is not evaluated by this training launch.

Launch implementation: `scripts/experiments/launch_neuralgcm_2p8_pipeline.py`.
Each submission persists intent before `sbatch` to avoid duplicate submissions
after an ambiguous interruption. Source/config changes require a new experiment.
The new matrix, cache lanes, gate checks, resource sizing, submission recovery,
snapshot dependencies and existing NeuralGCM/GC tests passed (45 tests total).

```bash
# Inspect the durable job graph and logs in the active experiment.
ls /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/artifacts/checkpoints/neuralgcm_residual/res2p8_w128_256_di16_32_train2015_2021_20260920/pipeline/
squeue -u sh4809
```

## Staging update, 2026-09-19 evening

The current user-selected priority is **2.8 degrees only**. 1.4-degree staging is
paused and its existing frames are retained. The data root is
`/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/data/neuralgcm/prepared/era5_2015_2023`.
The checkout and isolated training environment remain in home. The old home
prepared-data copy has been removed. Older dual-resolution commands below are
historical and are superseded by this staging procedure.

`scripts/run_neuralgcm_parallel_vis1.sh` now uses a complete CPU runtime on the
staging node's local disk, by default `/tmp/sh4809-neuralgcm-cpu`. The current
runtime is `/tmp/sh4809-ngcm-cpu.jotkcJ`. It contains the real Python interpreter,
stdlib, required shared libraries, pinned Python dependencies, frozen source and
checkpoint copies. It excludes CUDA plugins and NVIDIA libraries. The runtime is
disposable, but **all prepared data, receipts and progress logs stay on scratch**.

The runtime is produced by `scripts/preprocessing/build_neuralgcm_cpu_runtime.py`
on a responsive login host and transferred as one tar file for local extraction.
Its measured archive size was about 1.1 GB, versus the previous 5.7 GiB venv.
Copying only the old venv did not relocate its symlinked Conda interpreter.
The previous 80-worker run also suffered SIGBUS crashes in a GPFS-backed
`jaxlib/libjax_common.so` mapping. The node-local real-data benchmarks completed.

The launcher uses 20 workers with four CPUs each, staggered by 0.5 seconds at startup.
An initial 80 single-CPU run exceeded the user's cgroup limit of
129,713,782,784 bytes (about 121 GiB) and triggered OOM kills despite ample
host-level free memory. Worker concurrency must respect this account limit.
Workers reuse imports and model geometries across their assigned months. A
2.8-degree three-frame benchmark took 11.2–13.7 seconds per frame with one CPU
and 11.6–12.6 seconds with four CPUs. These are low-concurrency measurements,
**not a measured 80-worker throughput or an ETA**. Disk compilation caching is
disabled because JAX 0.9.2 emitted CPU target-feature mismatch warnings while
loading cached executables. In-process compiled functions are reused normally.
Core dumps are disabled and the working directory is local, not home.

`active_download.json`, `supervisor-res2p8.log` and `logs/local-lane-*.log` record
the active resolutions, CPU layout, runtime identity and intermediate startup
events. Old lane status files are archived under `logs/lane-history-*` on resume.
Each selected resolution can finish independently. Completion produces
`res2p8/READY.json` and `READY.res2p8.json` after checking all 13,148 timestamps.
It does not mark the entire dual-resolution dataset ready.

```bash
# Run on della-vis1. A second supervisor is rejected by the existing file lock.
NGCM_LOCAL_ENV=/tmp/sh4809-ngcm-cpu.jotkcJ \
  bash /home/sh4809/weatherforecast/Weather_Global/scripts/run_neuralgcm_parallel_vis1.sh
```

The implemented study has eight independent arms at 2.8°/1.4°, widths 128/256
and Mamba inner widths 16/32. Following the user's amended request, data splits
are **2015–2021 training, 2022 validation, 2023 test**. All statistics are fitted
on training records. Twenty passes refer to the full eligible seven-year segment
set, not the original two-year subset.

## What is implemented

`src/models/neuralgcm_residual/` contains the frozen backbone, native state
schema/increments, graph/Mamba branch, causal data access, normalization, decoded
loss, genuine k=1 cache, host-tape recurrent BPTT, live k=20 fine-tuning, checkpoint
resume/transfer, validation selection, diagnostics, reporting and stage workers.
No existing GC model, cache producer or cache manifest was modified.

The complete official NeuralGCM v1.2.2 source is in `third_party/neuralgcm/` at
commit `c91d2007ca37a4c0842f3ff83256d32e1ec30815`. Code is Apache-2.0 and the two
official checkpoint weights are CC BY-SA 4.0. `UPSTREAM.json`, the original
license and each downloaded checkpoint's manifest record provenance.

The isolated environment is `.venv_neuralgcm`, Python 3.11.15, JAX 0.9.2,
NeuralGCM 1.2.2 and Dinosaur 1.3.3. Full CPU and CUDA 12 dependency locks are in
`configs/experiments/neuralgcm_residual/requirements-*.lock.txt`.

```bash
source scripts/neuralgcm_env.sh
# To reproduce in a new environment, from repository root:
# python3.11 -m venv .venv_neuralgcm
# .venv_neuralgcm/bin/pip install -r configs/experiments/neuralgcm_residual/requirements-cuda12.lock.txt
```

Only the branch parameters enter the optimizer. The decoder input remains
differentiable. Solver calls execute outside the reverse pass, and physical
feedback and next-step physical features are detached. A step consumes one native
frame, plus its own recurrent SSM and causal-convolution memory. No baseline
forecast is added to branch inputs. The native increment modifies only the seven
declared dynamic field groups, with masked spectral conversion and no clipping.

## Verified so far

- Fifteen CPU contract tests cover the eight-arm matrix, seven-year completeness, split boundaries,
  Gaussian loss, native zero/carry preservation, partial-cache rejection,
  immutable source, dry-run submissions, decoded/recurrent gradients,
  20-step stop-gradient feedback, native Gaussian graph connectivity and exact
  interrupted training replay. The combined new and GC regression run passed
  all 26 tests.
- Eleven existing GC/Mamba regression tests passed in the isolated environment.
- Real 2.8° width128/inner16 and 1.4° width256/inner32 GPU smoke jobs passed
  exact zero-residual native/decoded comparisons at 1, 20 and 40 steps, and
  finite nonzero decoder-input gradients. Jobs were `14152384` and `14152386`.
- The inspected checkpoints both have 32 sigma levels and 193 dynamic channels.
  Their exposed native nodal grids are 128×64 and 256×128 respectively. The code
  uses inspected coordinates, not nominal degree spacing or README assumptions.
  Both checkpoints use one-hour internal timesteps, so `advance_6h` performs six.
- The small branch has 780,739 trainable parameters and 12,461,568 recurrent-state
  bytes. The large branch has 2,818,179 parameters and 24,923,136 recurrent bytes.

The zero-residual smoke uses official demonstration data. It is **not** a seasonal ERA5 skill
evaluation, a full training-memory benchmark or a production gate. Real seasonal
checks, verified full caches and GPU training profiles are required by production
workers. The eight pretraining and eight fine-tuning experiments have not finished.

GPU numerical preflight found non-deterministic graph reductions: identical
inputs produced different recurrent states (maximum absolute difference about
0.0012) and gradients on repeated calls. Input/native/cache states were exactly
equal, so this was not a cache serialization discrepancy. Diagnostics are
preserved under `implementation_smoke/real_training_diagnostic/`.

The experiment now locks `JAX_DEFAULT_MATMUL_PRECISION=highest`, disables x64,
and enables `--xla_gpu_exclude_nondeterministic_ops=true` before importing JAX.
This uses deterministic reductions and disables non-deterministic autotuning,
with a possible throughput cost that production profiling must measure.
The original tolerances were retained. Real ERA5 k=1 checks with nonzero residuals
and incoming memory passed with **zero absolute difference** in loss, memory,
gradients and optimizer updates on 2.8° (job `14154365`). The policy is recorded
in experiment, cache and resume identities and applies to every matrix arm.
See [OpenXLA determinism](https://openxla.org/xla/determinism) for the setting's
semantics. Checks on another GPU architecture must pass the same gates.

## Downloading the requested ERA5 data

The official source is public and requires no CDS key:

`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`

All seven pressure-level fields plus SST and sea ice are available. We select the
checkpoint's 37 pressure levels at six-hour intervals. Each raw timestamp is read
once and conservatively regridded to both Gaussian grids. Dinosaur's nearest-value
fill follows conservative `skipna=True` regridding. Raw forcing timestamps remain
unchanged in storage. The reader takes SST/ice from origin minus 24 hours and
persists it for the entire forecast.

A six-frame real-data pilot is in `data/neuralgcm/pilot_20200101/`. It measured
about 16–20 seconds per timestamp and about 29 MB total compressed output per
timestamp for the pair. Scaling that sample to 13,148 timestamps gives roughly
360 GiB for 2015–2023 prepared data. This is a sample-based storage estimate,
not a measured total or a cache-size estimate.

The complete resumable monthly download is defined in
`data/neuralgcm/prepared/era5_2015_2023/download_manifest.json`. There are 108
months. Per-frame hashes support recovery after interruption. Monthly and global
READY files appear only after full verification. The two final prepared stores
are `res2p8/` and `res1p4/`, referencing shared monthly shards without copying them.
Production workers reject any missing or additional six-hour timestamp in the
declared 2015–2023 interval. Early pilot shards cannot release production training.

This cluster's normal compute nodes cannot resolve public internet hosts. An
actual probe confirmed this. The existing repository's staging workflow uses
the internet-enabled `della-vis1`, so the downloader supports a persistent staging
supervisor there. The launcher defaults to 64 concurrent month workers with one
CPU thread each, for 64 CPU threads total. Set `NGCM_DOWNLOAD_WORKERS=32` to use
32 total threads. The supervisor pins each worker to distinct CPUs and rejects a
configuration larger than the host's available CPU affinity. This staging-host
resource allocation does not throttle the eight independent GPU experiments.

```bash
# Inspect progress from repository root.
python scripts/preprocessing/download_neuralgcm_era5.py status \
  --root data/neuralgcm/prepared/era5_2015_2023

# Resume on the internet-enabled staging host, if the supervisor stopped.
# A file lock rejects a second supervisor for the same dataset.
# Default: 64 workers x 1 thread = 64 CPU threads.
bash scripts/run_neuralgcm_parallel_vis1.sh

# Reduced mode: 32 workers x 1 thread = 32 CPU threads.
NGCM_DOWNLOAD_WORKERS=32 bash scripts/run_neuralgcm_parallel_vis1.sh
```

For another cluster whose compute nodes have internet, `submit --probe` followed
by `submit` schedules one array task per month and a merge job with `afterok`.
There is no Slurm array concurrency cap. Do not submit this route on Della until
the actual compute-node probe succeeds.

## Preparing and running the experiment

The current prepared experiment is
`artifacts/checkpoints/neuralgcm_residual/res2p8_res1p4_w128_256_di16_32_train2015_2021_v2/`.
Its executed source identity is
`c6c01aa49d346347cb707485986ec2c2d4983f14015b574f63959b8648827636`.
Both checkpoint/geometry inspection workers and submission dry-run succeeded.
The earlier definition-only root without `_v2` was rejected during integration
inspection; it has a `REJECTED_INSPECTION.txt` marker and must not be used.

`prepare` freezes source and writes eight locked configurations. It does not
start a training job. Worker commands re-execute the frozen source even if called
from a subsequently edited checkout. Changing source/config after preparation
requires a new experiment root. Data/cache files have their own producer identities.

```bash
python scripts/experiments/run_neuralgcm_residual.py prepare \
  --experiment-root artifacts/checkpoints/neuralgcm_residual/res2p8_res1p4_w128_256_di16_32 \
  --prepared-root data/neuralgcm/prepared/era5_2015_2023

python scripts/experiments/run_neuralgcm_residual.py submit \
  --experiment-root artifacts/checkpoints/neuralgcm_residual/res2p8_res1p4_w128_256_di16_32 \
  --stage preflight --phase inspect --dry-run
```

Run the following stages in order. Each worker takes a resolution or a run ID.

1. `preflight --phase inspect --resolution res2p8` and `res1p4` pin checkpoint,
   actual grid, field, time and native-tree manifests. Checkpoints already staged
   locally are reused by their hashes.
2. `data` prepares ERA5 if not already supplied by the monthly downloader. The
   downloader's merged complete stores can be used directly through
   `--prepared-root`. A direct data time shard accepts `--shard-id`, `--start`
   and `--end`; `--shard-id merge` merges completed shards.
3. `execute --stage cache --shard-id plan --resolution ...` defines immutable
   independent k=1 cache shards. `submit --stage cache` then fans out all those
   shards. A specific `--shard-id` selects one. Cache producer identity excludes
   graph width, random seed and training chunk partition.
4. `verify-cache` requires every shard and samples live recomputations. It also
   fits training-only feature/correction/loss statistics and climatology, then
   seals validation/test origin manifests. Keep test origins unused until selection.
5. `preflight --phase numerical` performs real zero-residual, live/cache gradient
   and update, recurrent carry and frozen-parameter checks on seasonal ERA5.
   `preflight --phase profile` measures cached backward and live twenty-step costs.
   Both resolutions must pass before any arm starts production. The profile uses
   the small 2.8° arm and large 1.4° arm.
6. `pretrain --run-id ...` runs twenty complete chronological-segment passes,
   validates one-step every pass and five-day AR at 0/5/10/15/20, and locks the
   selected cold 2022 parent. `finetune` transfers that arm's branch weights,
   resets optimizer/history and runs 2,000 twenty-prediction episodes.
7. `evaluate` requires locked pretrain/fine-tune selections. It compares baseline,
   pretrained and fine-tuned forecasts on 128 identical 2023 origins through day
   ten. Cold and warm are separate. Failures prevent successful completion.
8. `report` requires complete evaluations for every arm. It writes paired scores,
   capacity/lead comparisons, spectra, physical diagnostics and measured cost
   provenance. It does not manufacture unmeasured GPU-hour totals.

`submit` accepts explicit `--after JOB_ID` flags and records `afterok` dependencies.
It requires explicit `--memory-gb` and `--hours` for production stages. Set these
from the measured profiles. Workers independently enforce completeness and
compatibility, so merely supplying a dependency cannot bypass a failed gate.
Without a run ID, submission fans out all eight training arms. Della disallows
explicit partition selection in this account; let its submission policy choose.

`execute --resume PATH` restores parameters, optimizer, RNG, cursor and memory.
Uncommitted metric tails are archived before replay. Fine-tuning checkpoints are
saved only at episode boundaries. Stage transfer is a different operation and
records the selected parent hash. Validation uses fresh memory and does not modify
the training stream.

## Checks

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/neuralgcm_residual
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_graphcast_interleaved_temporal.py tests/test_temporal_mesh_mamba.py \
  tests/v24_Ilya/test_dependency_boundary.py tests/v24_Ilya/test_temporal_routing.py
```

Real-model GPU checks are available as
`scripts/training/smoke_neuralgcm_residual.py` and
`scripts/training/smoke_neuralgcm_training.py`. Their output is explicitly marked
as smoke-only and cannot release production jobs. Original measurements are in
`artifacts/checkpoints/neuralgcm_residual/implementation_smoke/`.

## Official references

- [NeuralGCM source](https://github.com/neuralgcm/neuralgcm/tree/c91d2007ca37a4c0842f3ff83256d32e1ec30815)
- [Checkpoint catalog and weight license](https://github.com/neuralgcm/neuralgcm/blob/c91d2007ca37a4c0842f3ff83256d32e1ec30815/docs/checkpoints.md)
- [Official ERA5 preparation](https://neuralgcm.readthedocs.io/en/stable/data_preparation.html)
- [Native state and inference API](https://neuralgcm.readthedocs.io/en/stable/deepdive_into_models.html)
