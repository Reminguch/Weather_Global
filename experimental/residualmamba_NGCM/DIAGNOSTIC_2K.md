# Automatic 2,000-update H200 loss diagnostic

The user requested a 2,000-update run immediately after data restoration,
using H200 `gpu-test` jobs to determine whether loss decreases. This document
defines that bounded diagnostic and its automated launch sequence.

## Scientific scope

The experiment uses the current GraphCast-Mamba implementation and the official
pretrained NeuralGCM backbone. Resolution, fields, 2015–2021 training split,
2022 validation split, 24-step BPTT, chronological 96-anchor segments, open-loop
inputs, memory carry/reset, normalization and loss match the core specification.
Statistics use every training frame (`stats_stride=1`), and all 424 training plus
60 validation baseline chunks are prepared. Training stops at update 2,000.

The optimizer keeps its original schedule: 200-step warmup to 1e-4, then cosine
decay toward 1e-5 at update 10,000. Bounding this invocation at 2,000 does not
compress that schedule. There is no dynamic weighting or automatic retuning.

This diagnostic has two explicit operational/scope differences from full
production qualification in README.md:

- Its cache uses the authorized **MENGDIW scratch workspace**, because the
  current account cannot access DABANIN. A real project quota audit and at least
  200 GiB available space are required at setup.
- The expensive full year-by-year native forecast audit, C1 bias table, SWA
  over 2,000–8,000 and full E1/E2 evaluation are deferred. The diagnostic does
  not claim full production readiness or forecast improvement.

The original production CLI and its prerequisites remain available. This
separate entry point records `production: false`. Data integrity, full restored
data, train-only statistics, cache verification, zero-head loss census and
qualified restart behavior are still required. If the census reports a
concentrated field/level contribution, the decision explicitly retains the
predeclared scales for this bounded trial and records the warning. Validation
does not select new scales or decide whether finite training may continue.

## Automatic execution

`scripts/training/ngcm_2k_controller.py` owns a durable job journal. Preparation
starts when the downloader's `COMPLETE.json` matches the verified 13,148-frame
store. The callback-stop GPU regression and qualification must pass before the
final training checks and optimizer updates; independent data preparation can
proceed while that regression waits in the queue.

1. A CPU setup job verifies geometry, checkpoint identities, full-data splits,
   quota evidence and scratch space.
2. Full training statistics run on CPU while three disjoint cache lanes run
   on H200. Each lane resumes completed immutable chunks across invocations.
3. Cache/live comparisons on three train chunks and one validation chunk,
   native-tail comparisons and the full validation loss census run before
   training. The diagnostic records a decision on concentration warnings.
4. Training advances toward update 1,000, with complete checkpoint continuation
   between short jobs. All 15 validation segments are evaluated at 1,000.
5. Training continues to update 2,000, followed by the same fixed validation
   set. Validation is stored per segment so an interrupted monitoring job can
   continue independently of the optimizer.
6. A CPU reporting job writes `report/loss-curves.png`, `summary.json`,
   `matched-anchors.json`, `REPORT.md` and a final completion receipt.

Every GPU allocation uses `--account=mengdiw --partition=ailab --qos=gpu-test
--gres=gpu:h200:1 --cpus-per-task=8 --mem=96G --time=01:00:00`. The controller
uses at most three parallel GPU preparation jobs, consistent with the current
QOS limit. Sequential optimizer updates use one GPU.

Workers budget roughly 50 minutes of useful runtime. After a committed update,
the trainer checks remaining time, saves parameters, Adam state, Mamba memory,
RNG and the next data cursor, then exits successfully before the hard limit.
An early stop does not reset memory or advance the schedule independently.
Nominal 500-update checkpoints are retained. Validation at 1,000 and 2,000 is
run in separate jobs, and completes before the next training leg.

Finite loss increases and finite numerical variability do not stop this trial.
Nonfinite computation and corrupt/mismatched artifacts do. Scheduler timeouts,
preemption and node failures may resume from the latest committed checkpoint;
transient explicit cluster submission-capacity rejections are retried. An
ambiguous submission is recorded and never silently duplicated.

## Reading the result

Raw chronological losses can change because consecutive chunks cover different
weather. The report therefore includes raw loss, a 100-update rolling mean,
loss relative to the first encounter of the same anchor, and a paired comparison
of the final 304 chunks against those same chunks in the first epoch. The first
encounter may already contain optimizer updates; it is not labeled an untrained
NeuralGCM comparison.

Validation at 1,000 and 2,000 uses exactly the same 15 segments and reports loss
relative to the zero-head official NeuralGCM baseline, with prefix and tail
lead breakdowns. A declining training objective alone is not a weather-skill
claim. Results are reported even if loss stays flat or increases.

## Paths and operation

The requested run lives under
`Weather_Global_rebuild/artifacts/diagnostic_2k_20261105/`. The plan pins the
source, libraries, runtime policy and orchestrator. A frozen code snapshot is
used by the watcher and batch jobs so subsequent repository edits do not alter
this run. All code snapshots, caches, logs, checkpoints and outputs are on
scratch.

```bash
# Arm once, after recording the actual GPU regression job ID.
../Weather_Global/.venv_ngcm_data/bin/python scripts/training/ngcm_2k_controller.py \
  arm --root "$PWD/artifacts/diagnostic_2k_20261105" --qa-job <job>

# The persistent watcher runs the frozen copy of the controller.
../Weather_Global/.venv_ngcm_data/bin/python \
  scripts/training/start_ngcm_2k_watch.py \
  --root "$PWD/artifacts/diagnostic_2k_20261105"
```

Create `code/` from the recorded Git commit before starting the watcher, and
create its `.tmp` and `.cache` directories on scratch. The detached launcher
records its PID and log under `controller/`; it does not depend on an open SSH
session.

`controller/status.json` distinguishes waiting for GPU regression, waiting for
verified download, preparation, training and completion. `controller/jobs/`
records actual Slurm job IDs and outcomes. `checkpoints/training.jsonl` records
each optimizer update. Reporting uses the latest committed occurrence of a
step if an interrupted invocation produced duplicate log rows, and records the
duplicate count explicitly.
