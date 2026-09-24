# NeuralGCM with an added residual head: initial 12 h training

**User direction, September 24, 2026:** keep the experiment as close as possible
to the original deterministic NeuralGCM and add our residual head. Start with
K=2, a 12 h forecast made of two consecutive six-hour correction intervals.

**Status:** this is the revised experiment specification. It is not an activated
training configuration or evidence that the complete paper loss and full
physical-state backward pass have been implemented. Existing v2 checkpoints,
plots and immutable jobs retain their historical definitions.

This direction replaces the proposed equal-variable objective as the primary
next experiment. It also replaces the original plan to require cached K=1
pretraining before the new experiment's live training. The earlier six-hour
normalizations and stop-gradient physical feedback are historical variants,
not the target protocol below.

## What “12 h per update” means

The paper's Supplementary Table 5 (Table 5 in the arXiv HTML) starts both the
2.8-degree and 1.4-degree deterministic models at a **12 h unroll length**.
It changes to 24 h at optimizer step 2,000, then progressively extends to
36, 48, 60 and 72 h. Twelve hours is the initial forecast horizon, not the
integrator timestep, the loss-normalization interval, or a permanent training
horizon. The 0.7-degree model has a different curriculum, beginning at 6 h.

For our initial K=2 experiment, one optimizer update processes a batch of
complete 12 h trajectories. Within a trajectory:

1. Encode ERA5 at the forecast origin once and initialize the residual memory.
2. Advance the original NeuralGCM for 6 h, apply the residual increment, and
   retain the corrected native state and residual memory.
3. Advance that corrected state for another 6 h and apply the second increment.
   Do not replace the 6 h state with ERA5.
4. Evaluate the trajectory objective using the 6 h and 12 h predictions and
   matching ERA5 targets, including the native and decoded representations.
5. Differentiate the batch objective through the whole trajectory and perform
   one optimizer update. The residual parameters stay fixed during both steps.

If the batch contains B forecast origins, one optimizer update uses B distinct
12 h trajectories. K=2 does not specify B. Origins may be sampled independently;
the next optimizer update need not start where the previous forecast ended.

The original solver keeps its checkpoint-defined internal timesteps and
learned-physics schedule. Our six-hour residual injection interval is an added
interface choice, not a claim about the original learned-physics update rate.

## Trainable parameters and gradient paths

The pretrained encoder, dynamics, learned physics and decoder remain the
reference NeuralGCM. Only residual-head parameters are optimized. Frozen
parameters are constants for optimization; derivatives with respect to their
inputs must still propagate.

The 12 h loss must reach the first residual increment through the intervening
six-hour physical integration, as well as through the recurrent memory and
the next residual input features. All of these paths are part of the new
backward pass. The current host-tape reverse pass carries only memory
cotangents and cannot implement this by changing K alone. Removing one
`stop_gradient` is also insufficient.

Use gradient checkpointing/recomputation to manage activation memory, preserving
the differentiated function. Independent training episodes reset physical
initial conditions and memory. Any later observed-history memory initialization
is a separately recorded extension. Existing K=1 cache/statistics can support
diagnostics but cannot substitute for the second live forecast or supply the
missing solver-state derivatives.

Zero residual output must recover the frozen NeuralGCM trajectory and decoded
predictions. Original numerics, grids, units, state carry, encoder and decoder
must be preserved. The existing pressure and degree-zero increment restrictions
are properties of our residual interface, not original NeuralGCM loss terms.
Their compatibility with the original state constraints must be recorded; neither
their removal nor a new clipping/filtering rule follows automatically from this
change in objective and gradients.

## Objective to align

Use the deterministic paper's main-stage objective as the reference:

```text
L = 20 * M_data + M_model
    + 0.1 * (M_data_spectrum + M_model_spectrum)
    + 2 * M_bias
```

Data and model accuracy terms use the paper's lead-dependent filtering in
pressure and sigma representations. Spectrum terms compare spectral amplitudes.
Bias aggregation occurs across the specified batch and forecast-time axes
before squaring. In particular, averaging independent per-example squared
bias losses is not equivalent to a squared batch-mean bias.

Use 24 h ERA5 differences for loss scales, pooled across levels except for
specific humidity, with the paper's amplitude factors (Z=2, q=0.66,
cloud species=0.05, internal log surface pressure=5). Apply the documented
lead-dependent scaling separately for the relevant terms. A 12 h training
horizon does **not** change the normalization interval to 12 h.

Do not silently retain the custom pressure-proportional level weighting,
seven-field average, six-hour scale floors or RMS-of-per-level-STD pooling
under an “original NeuralGCM loss” label. Do not substitute the proposed
equal-baseline-error normalization for the paper objective.

The vendored reference code provides transformed L2, spectrum, batch-bias and
scaling components. The inspected inference configs do not establish the full
original training-loss bindings. Exact variable/level selection, reductions,
normalization constants, native target construction, filter parameters and
initial-time treatment still require a recorded reference configuration and
numerical checks. Code defaults alone are not evidence of the paper's bindings.
Where exact original artifacts are unavailable, disclose the reconstruction
and its differences rather than claiming exact reproduction.

## Other comparisons that must remain explicit

- The paper trains NeuralGCM parameters; our experiment trains an added head on
  a frozen pretrained model. This is the intended intervention.
- The user-selected split remains 2015–2021 training, 2022 validation and 2023
  test. It differs from the original model's training data. Fit any new loss
  statistics only on the selected training split and share them across arms.
- The paper uses Adam with beta1=0.9, beta2=0.95 and epsilon=1e-6. Its 2.8-degree
  peak learning rate is 0.002 with a 2,000-step warmup. Current residual AdamW
  settings are different. Record the chosen optimizer, batch size and schedule
  explicitly; the paper's settings do not demonstrate an optimal learning rate
  for our head.
- Start directly with live K=2 and a fresh zero-output head. The later choice
  to extend the horizon is separate from validating this initial stage. Staying
  at K=2 for the entire experiment would differ from the full paper curriculum.
- The paper's separate decoder fine-tuning stage is not automatically included
  when the intended trainable component is only our residual head.

## Initial verification

Run an independent representative-ERA5 GPU smoke before relying on full new
training statistics. Use Slurm `--qos=gpu-test`, a GPU request, no explicit
partition, and at most one hour. Keep pilot statistics and outputs separate.

Verify zero-head baseline identity; the loss against reference components;
the 12 h loss derivative with respect to the first increment using a nonzero
perturbation; finite gradients and an actual residual-parameter update; frozen
backbone parameters; and that the second forecast consumes the corrected first
state. Measure memory and elapsed time rather than assuming full gradients
are infeasible. A diagnostic loss used only for gradient checks must be labeled
as such and cannot certify paper-loss parity or production training.

Report training curves against optimizer updates and physical-unit errors
separately at 6 h and 12 h. Baseline and residual use identical forecast origins,
truth, loss constants and verification grids. Report each of the five loss
terms and their variable contributions alongside aggregate scores.

## References and implementation locations

- [Paper training and curriculum, Sections 7.1–7.2 / Supplement G.1–G.2](https://arxiv.org/html/2311.07222v3#S7.SS1)
- [Paper rescaling and losses, Sections 7.3–7.5 / Supplement G.3–G.5](https://arxiv.org/html/2311.07222v3#S7.SS3)
- [Current gradient kernels](../../../src/models/neuralgcm_residual/kernels.py)
- [Current live rollout](../../../src/models/neuralgcm_residual/finetune.py)
- [Current locked configuration](../../../src/models/neuralgcm_residual/config.py)
- [Reference metric implementations](../../../third_party/neuralgcm/neuralgcm/reference_code/metrics.py)
- [Reference transforms](../../../third_party/neuralgcm/neuralgcm/reference_code/linear_transforms.py)
- [Historical normalization audit](../../../plot/loss_alignment_audit_20260924/README.md)
