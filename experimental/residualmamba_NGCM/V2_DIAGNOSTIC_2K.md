# Bounded loss v2 training diagnostic

The user stopped the v1 training and evaluation jobs on 2026-11-05 and requested
starting loss v2. Existing checkpoints, evaluation results, input statistics and
baseline caches are retained. The v1 controllers were stopped before cancelling
their Slurm jobs so they cannot resubmit them.

This run starts a fresh v2 model and Adam state. It uses the already qualified
five-field core (source identity
`bb79520da23d4de9713c38e038dc6726d0d5fc90abcb390ac621ad274f6e384b`), seed 22,
145 outputs, BPTT 24, segment length 96 and `ar_tail_k=20`. The physical native
tail is 21 forecasts, through 126 hours. Cloud fields remain inputs only.

The endpoint is **2,000 optimizer updates**. Learning rate remains the declared
200-update warmup to 1e-4 followed by the original 10k cosine schedule, not a
schedule compressed into 2k. Every 100 updates a callback commits full training
state; wall-clock boundaries also save state. Each ailab H200 gpu-test job has
a one-hour allocation and exits after about 50 minutes plus checkpoint time.
The durable controller submits subsequent jobs until update 2,000.

One independent H200 evaluator discovers committed checkpoints immediately.
It evaluates every saved checkpoint on all 15 fixed 2022 validation segments
(60 chunks), with the same tail20 inputs and per-segment memory reset. Partial
segment receipts can resume across job boundaries. Reports include all five
field contributions and shares. The three separate plots (train loss, eval
loss, per-variable eval losses) refresh after each completed evaluation.
Only completed evaluations appear in plots.

`v2_diagnostic.py` and `scripts/training/v2_ngcm_controller.py` implement the
bounded orchestration. The trainer, optimizer, native rollout, normalization,
model and loss are unchanged from the passing v2 H200 qualification. The worker
uses an immutable committed code snapshot, authenticates all prerequisite
receipts, and reuses the existing v1 raw inputs via `reuse.import_inputs`.
The zero-head census has already recomputed all 60 validation chunks under v2,
with loss 3.5251650667. No new statistics or forecasts are generated.

As in the previously authorized 2k diagnostic, the inaccessible DABANIN cache
location is replaced by existing MENGDIW scratch and the full annual native
baseline audit remains deferred. This bounded run does not waive the production
CLI audit requirements or establish held-out 2023 forecasting skill. Finite
loss increases or weak validation do not stop the experiment. Nonfinite values,
corrupt inputs/checkpoints, source mismatches and resource failures remain
errors. No v1 total is compared with the v2 total.

The controller allows one training job and one evaluation job concurrently.
`controller/user-stop.json` prevents further submissions if the user stops this
run. Job submissions, checkpoint steps and scheduling status are journaled under
`controller/`. A completed run requires both update 2,000 and evaluation of all
saved checkpoints. All data, environments, logs, temporary files and outputs
remain under scratch.

## Independent ar_tail_k=1 v2 run

The user additionally requested this arm on 2026-11-05. Prepare a **new** run
with `--tail1-donor-root` pointing to the completed v1 tail1 raw cache, alongside
the usual tail20 `--donor-root` for shared statistics and common evaluation.
`v2_tail1_inputs.py` authenticates the native producer, all 484 chunk receipts,
READY, train-statistics identity and the existing native/restart evidence.
No v1 learned parameters or optimizer state are imported.

The v2 core remains exactly the qualified five-field implementation. The
unchanged `tail1_extension.py` supplies `Tail1Config`: BPTT 24, segment 96,
ar_tail_k 1, truth prefix 23. Thus the final native tail consists of the 6 h and
12 h forecasts, following 22 independent one-step prefix forecasts. Memory
still carries across four chunks and resets at the next segment. Only tail
length and its derived prefix change. The raw-input preflight describes shared
tail20 evaluation inputs; `plan.json` identifies the training arm and the
trainer's `checkpoints/run.json` records the actual Tail1Config.

Both arms use fresh seed-22 initialization, 2,000 updates, the same optimizer
schedule and five-field loss. The tail1 arm saves every 100 updates and receives
dense evaluation on the **same tail20 validation inputs** as the tail20 arm.
Its plots are labelled tail1; its shorter-horizon training loss must not be
compared directly with tail20 training loss. This comparison changes training
horizon and teacher forcing, so it is not by itself an isolated memory ablation.
The already running tail20 arm uses its original immutable snapshot.

Validation reuses the passing v2 GPU model/restart qualification and the prior
native tail1 loader/continuation evidence, with v2 CPU tests for actual routing,
configuration, continuation and checkpoint labels. `check --sample` additionally
loads a real full tail1 chunk under v2, checks 796 input / 145 target channels,
finite values and all five loss contributions before submission. The first
live H200 updates provide the real-data integration check. Reusing evidence is
not described as a newly executed tail1-specific GPU restart experiment.
