# K=2 learning-rate schedule control

This experiment tests whether the existing learning-rate schedule contributes
to the physical forecast regression. It is not a replica of GraphCast training.

The numerical source is reused byte-for-byte from the completed strict >30 hPa
campaign. The only training changes are peak LR `0.002 -> 0.0001` and warmup
`2000 -> 200` updates. Adam betas `(0.9, 0.95)`, epsilon `1e-6`, no clipping,
no weight decay and the existing exponential-decay schedule remain unchanged.
The latter starts decaying after 15k updates, outside this 2k budget. Therefore
this control has 200 warmup updates followed by a constant LR, not GC's cosine.
It identifies the schedule as a joint factor, not peak LR versus warmup separately.

Keep K=2, seed 22, width 128, d_inner 16, batch 2, fresh zero residual head,
all seven strict >30 hPa objectives, native loss off, calibration origins,
training order, validation origins, and adaptive controller W=100/probe=20.
Train 2,000 updates. Preserve the baseline as a checkpoint-selection candidate.
Evaluate all variables and all 37 levels, including independent 120-hour
forecasts, with the same held-out origins as the reference. Plot against
training compute time as requested, retaining update counts in the CSV.

`prepare.py` creates an isolated scratch campaign and explicit driver arguments.
It does not submit jobs. The inherited real-model gradient/mask smoke and frozen
source are recorded by hash. `schedule_smoke.py` additionally runs a representative
pilot in separate GPU processes at the exact new LR/warmup and checks uninterrupted
versus interrupted training, parameters, Adam state, RNG, controller, and recovery
from torn derived logs. Pilot statistics and checkpoints are separate from
production. Production requires this new smoke to pass.

H200 jobs use ailab / gpu-test and one-hour allocations. Production checkpoints
near 55 minutes and requeues the same job, with six allocations maximum and no
500-update cap. This does not alter the running fresh K=20 control. Transfer of
a K=2 model into K=20 must be a separate, explicitly recorded experiment after
the K=2 evaluation, not an implicit continuation with reset optimizer state.
