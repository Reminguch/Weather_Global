# Dense checkpoint validation during both 2k runs

The user requested concurrent, more frequent checkpoint evaluation on
2026-11-05. This monitor evaluates every available committed checkpoint from
both independent runs on the exact same complete 2022 validation set used by
the original tail20 experiment: 15 segments, 60 chunks, all 24 prediction
positions, including the native 6–126 h tail. It uses the original train-only
normalization and fixed loss weights. Validation does not tune weights, select
training examples, stop training on finite losses, or alter the optimizer.

## Checkpoint cadence without changing training mathematics

The currently running batch keeps its original behavior and completes normally.
Existing checkpoints, including update 323 and subsequent walltime checkpoints,
are evaluated immediately when scheduler capacity permits. Future continuation
jobs add a save at each multiple of 100 updates using the qualified trainer's
existing callback. After a save, the driver re-enters the same trainer in the
same job, restoring all parameters, optimizer state, memory, RNG and cursor.
The original 500-update periodic setting remains in the scientific config;
the extra saves come from the host callback, not a changed model configuration.

The model, gradients, learning-rate schedule, sample order and run identity
remain unchanged. This approach can add compilation/restore overhead at the
100-update boundaries. It avoids changing the source or parameters of an
already-running training process. Existing checkpoint files are immutable.
No missing early checkpoints are reconstructed or presented as measured.

A separate controller process replaces only future training dispatch. It
imports the original immutable controller, reuses its existing job journal and
stage planner, and replaces only the training command. Cache, QA, 1k/2k
validation and final reporting continue through their original workers.
Replacing a controller does not cancel its submitted or running Slurm jobs.
The dispatch receipt records the pinned new code and previous controller PID.

## Evaluation and restart

One independent ailab H200 gpu-test evaluator runs at a time, alongside the
training/preparation jobs when the shared three-job limit allows. Evaluation
jobs request one hour and save each segment result atomically. A partial
validation resumes from its completed segments. No checkpoint is used until
both its payload and its checksum receipt exist; checksums and run identity
are verified before loading.

Mamba memory resets at each validation segment boundary and carries across
its four chunks. The evaluation scan returns per-step, per-variable and
per-level contributions to the existing weighted normalized MSE. Aggregation
matches the qualified validation definition. Both trained arms are evaluated
on common tail20 inputs. The tail1 experiment's own-horizon validation at
1k/2k continues separately.

The generated `eval-loss.png` has eight panels: total validation objective and
seven variable contributions. Each plot uses optimizer updates on the x-axis.
Step zero is the zero-output residual head, equivalent to the official
pretrained NeuralGCM baseline. These are weighted normalized losses, not raw
physical-unit RMSE. Compare checkpoints on this fixed validation set rather
than treating a decrease across changing training chunks as forecast skill.
The monitor also writes `summary.json` and `REPORT.md` after each complete
checkpoint evaluation. Only complete 15-segment evaluations enter the plot.

## Validation evidence

Two targeted CPU tests passed. One compares the scan evaluator against the
qualified per-step validation implementation and checks memory resets. The
other crosses multiple callback-save/restore boundaries and a segment boundary,
then checks every checkpoint leaf against uninterrupted training bitwise.
Evidence is in `validation/20261105_dense/cpu-tests.txt`.

The existing actual H200 fresh-process continuation tests remain applicable
because the underlying trainer and deterministic runtime policy are unchanged.
The tail1 real-data GPU test also completed with zero differences in parameters,
memory, optimizer, RNG and cursor, and zero native prefix/tail discrepancy.
