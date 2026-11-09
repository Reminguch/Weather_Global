# Fresh GC–Mamba versus GraphCast fine-tuning: accuracy, spectra, and smoothing

**Date:** 2026-11-06, America/New_York.  
**Status:** implemented and submitted on 2026-11-07; full-GC calibration completed. Pending GCFT λ0 production was redirected to fresh peak-LR `3e-7` training on November 7; the revision below supersedes the original λ0 calibration-selection rule.  
**Workspace:** `/scratch/gpfs/DABANIN/iv9432/Weather_global`.  
**Related records:** [accuracy/spectral experiments](../../research/directions/r005-mgda-accuracy-spectral-adjustment/overview.md), [extreme-event evaluation](../../research/directions/r006-extreme-event-evaluation/overview.md).

## GCFT λ0 revision — 2026-11-07 evening

The user authorized reusing pending job **`17049162_0`** for a fresh public-GC
start at peak LR **`3e-7`**, without carrying the earlier 500-update weights or
optimizer state. The new run is `gcft_s22_l000_lr3em7_fresh_20261107`; it retains
seed22, frozen training order, batch4, BPTT24/K20, 100-update warmup and the full
2,500-update cosine schedule, ending at `3e-8`. The existing first/second parts
stop at 1,250/2,500; continuation job **`17049163_0`** is preserved. Both training
job IDs, original submission age and resource requests are retained. Slurm start
times remain estimates; the activation check showed 23:00 Eastern, pending priority.

This authorization changes **λ0 only**. The four positive-λ GCFT arms remain at
peak `1e-5`, including the retained eight-update λ0.5 profile. Consequently this
revision is **not a fixed-LR λ-only comparison**. The new λ0 LR is provisional
and user-selected, not a completed development-calibration winner. Separate
`1e-6`/`3e-6` pilots discussed in chat were not part of this narrow activation.

The completed GCFT λ0 calibrations remain preserved: at update500, peak `1e-5`
gave exact dev32 rollout loss12.339464 versus its GC reference10.833993
(13.8958% increase), and `3e-5` gave16.492603 versus10.832740 (52.2477% increase).
These finite degradations motivated the user-directed fresh start; they are not
execution failures or automatic stopping gates. Full-GC profiling and both
500-update calibrations completed on one80GB GPU, establishing feasibility for
this shape. Measured steady p90 was approximately59–60s/update; no LoRA fallback
is needed for feasibility. No forecast skill is yet established for the new run.

The frozen runtime and original campaign manifest/configs are unchanged. A
versioned [revision record](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7/manifest.json)
preserves the old and new selected-arm mappings, the new config and helper hashes,
and [activation evidence](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/revisions/20261107_gcft_l000_fresh_lr3em7/activation.json).
The active `selected_arms_full_gc.json` points task0 to the fresh run with
`resume_step=0`. Other task mappings are unchanged.

Extra development job **`17062379`** evaluates the fresh saved step500 checkpoint
after the first training part completes, using the same32 starts and40 leads.
It supplies the actual score required by the original frozen history evaluator;
it does not reuse the old higher-LR result. Existing history job **`17049164_0`**
now waits for both `17049163_0` and `17062379`. The final smoothing/evaluation
controller follows the revised active mapping. Focused CPU checks verified fresh
and continuation dispatch, configuration validity, unchanged other arms and
frozen-source integrity; no extra GPU qualification training was submitted.

## Initial execution update — 2026-11-07

The user authorized implementation and launch of fresh GC–Mamba training, vanilla-GC fine-tuning, and smoothing. The scalar objective, full-GC training/evaluation adapter, frozen segment replay, smoothing fitter/observer, and staged orchestration are implemented and tested. The implementation work listed in §7 is no longer only proposed.

| Workstream | Initial Slurm jobs | What follows automatically |
| --- | --- | --- |
| Full-GC fine-tuning | `17022621_0,1,4`; controller `17022622` | Retain eight updates per profile; measure full-tuning fit/cost; calibrate two λ0 learning rates to 500; train five λ arms to 2,500 |
| Fresh GC–Mamba | `17022621_2,3,5`; controller `17022623` | Same calibration/sweep, independently of the GC controller; public spatial overlay, fresh Mamba1/seed22/zero heads, no trained Mamba initializer |
| GC/M1 smoothing | First development forecast `17022624`; controller `17022625` | Complete 32 development starts, fit both filters, evaluate 122 annual starts and report |

All were **PENDING** when checked after submission. This establishes launch, not GPU execution, full-GC memory feasibility, or forecast skill. LoRA has not been substituted: successful full-GC profiling will establish that full tuning fits this configuration. A resource failure requires diagnosis; there is no automatic reduction of batch/horizon or silent switch to LoRA.

- [Training campaign](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifest.json), [six initial configurations](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/configs), [submission records](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/submissions).
- [GC/M1 smoothing campaign](../../artifacts/evaluations/v24_smoothing_m1_20261107/manifest.json) and [submission records](../../artifacts/evaluations/v24_smoothing_m1_20261107/submissions).
- [Implemented training driver](../../scripts/experiments/run_v24_scalar_lambda.py), [smoothing driver](../../scripts/experiments/run_v24_smoothing.py), and [aggregate report](../../scripts/analyze_models/report_v24_scalar_lambda.py).

Frozen runtime SHA-256 digest: `ab39c404b062a55f2ee93883672442973fa4a2351699529ac11492a1a6b88475`. Runtime lives in the training campaign's `source/` directory; submitted jobs do not follow later workspace edits. Both initial resource requests passed Slurm test-only validation. Training profiles request one 80 GB GPU, 8 CPUs, 384 GiB RAM and 180 minutes each; the first smoothing forecast requests one 80 GB GPU, 4 CPUs, 32 GiB RAM and 120 minutes. Later budgets use these retained runs' timing/RSS measurements. No array throttle is imposed.

Each method's λ0 candidates use both proposed learning rates. A λ0.5 profile at the lower rate measures spherical-loss cost; its eight updates are retained if that rate wins, otherwise its bounded overhead is recorded and λ0.5 starts fresh at the selected rate. Production is split at 1,250/2,500 updates with exact resume and a fixed 2,500-update schedule. Intermediate development forecasts use the saved 500/1,000/1,500/2,000 checkpoints; the final development pass also fits filters. Historical stage metrics are copied into immutable proof files before subsequent updates append to the live log.

Each completed method automatically launches its five final models' development fitting and annual raw/filtered evaluation. The aggregate report waits for all ten plus the standalone M1 reference and uses that reference's public-GC forecast as the canonical denominator. It writes λ-labeled tradeoff plots, exact-loss pointwise paired block intervals and same-λ method contrasts. Spectral coordinates are descriptive in this automatic summary; per-date powers remain available for the nonlinear spectral resampling specified in §6.4. Seed repeats and the optional AR study are not submitted.

Actual source-time verification found 106 eligible 96-position training segments, with the last target at `2021-12-19T12:00:00`, and 2,500 frozen update mappings. Development32 spans `2022-01-02T00:00:00Z` to `2022-12-01T12:00:00Z`; it is the existing matched32 cohort resolved from the actual source time coordinate. See the [development manifest](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifests/development_2022_32.json) and [training order](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/manifests/train_order_seed22.json).

Monitor the initial jobs with:

```bash
squeue -j 17022621,17022622,17022623,17022624,17022625
```

Later job IDs are written under each campaign's `submissions/`. The [research handoff](../../research/directions/r005-mgda-accuracy-spectral-adjustment/devel.md) records execution state and failure/resume rules.

## 1. Scientific question and scope

Compare a fresh GC–Mamba correction, with its original GraphCast backbone frozen, against fine-tuning all parameters of the same original GraphCast (GC). Both methods optimize the same exact-grid-MSE-plus-spectral-adjustment objective at five fixed adjustment strengths. Independently fitted output filters test how much of the accuracy improvement can be obtained by attenuation alone.

The primary questions are:

1. At the same optimization-update and data budget, how do full GC fine-tuning and GC–Mamba differ in their MSE–spectral-fidelity tradeoffs?
2. Does either learned method outperform unmodified GC after GC receives a development-fitted smoothing filter?
3. Does a GC–Mamba advantage remain when both trained methods receive the same filtering opportunity?
4. What GPU time, memory, and inference cost are required for each operating point?

The campaign must report the full tradeoff even if GC–Mamba does not win. A single model size and data budget cannot establish a general scaling law. Measured accuracy versus GPU-hours can support a bounded efficiency claim. Neither this comparison nor the existing memory ablation establishes that Mamba is uniquely necessary among recurrent architectures.

**Core deliverable:** ten final seed-22 models, unmodified GC and principal historical M1 references, and raw/Gaussian-filtered/band-filtered evaluations on the same 2023 starts. The existing matched reset-trained model is an additional useful reference; it is not a new arm.

**Follow-ups:** complete the separately running extreme-event evaluation; replicate a development-selected model pair; only then consider shorter AR training. LoRA, MGDA, an LSTM replacement, and an AR-length sweep are not required for the core comparison.

## 2. Experiment matrix and initialization

| Item | Fresh GC–Mamba | Full GC fine-tuning |
| --- | --- | --- |
| Run IDs | `gcm_s22_l000`, `l025`, `l050`, `l075`, `l100` | `gcft_s22_l000`, `l025`, `l050`, `l075`, `l100` |
| Outer loss coefficient | 0, 0.25, 0.5, 0.75, 1 | Same five values |
| Starting GC | Public GraphCast-small checkpoint in §11 | Identical checkpoint |
| GC parameters | Frozen | All trainable |
| Correction | Fresh spatial baseline overlay and fresh Mamba1 initialization | None |
| Initial output | Public GC forecast plus zero correction | Public GC forecast |
| Training budget | 2,500 optimizer updates, including warm-up | Same |
| Effective batch | Four independent weather streams | Same |
| Training sequence | 24-step BPTT, K20, 96-step segments | Same weather windows |
| Weather feedback gradient | Full gradient within each 24-step window | Same |
| Main checkpoint | Native checkpoint after update 2,500 | Same; no SWA |

Every λ arm starts independently. Within one method and seed, use identical initial parameter values, initial optimizer state, and data order across λ. Save and hash a step-zero initialization for each method/seed. Resuming an interrupted arm restores that arm's complete state; it does not initialize another λ arm.

GC–Mamba initialization follows the existing `baseline_overlay` recipe: overlay compatible public GC spatial parameters into the freshly constructed correction, retain the prescribed fresh Mamba parameters, and use the zero-initialized residual output head. Do not initialize from M1, the September MSE parent, or any trained AMSE/MGDA checkpoint. Verify the initial full prediction against public GC and preserve the overlay report.

The architecture is fixed at the existing 1°/13-level configuration:

- Native grid: 181 latitudes × 360 longitudes; mesh size 5; GC width 512 and 16 processor message-passing steps.
- Correction: width 512, two processor message-passing steps, two interleaved temporal layers per processor block, four temporal layers in total.
- Mamba: `d_inner=16`, `d_state=16`, `d_conv=4`, `bc_groups=1`, `init_scheme=mamba1`, stateful memory, dropout 0, zero temporal output initialization.
- Other temporal settings: `bias=false`, `conv_bias=true`, `dt_rank=auto`, `dt_init=random`, `dt_scale=1`, `dt_min=0.001`, `dt_max=0.1`, `dt_init_floor=0.0001`. Copy and hash the complete architecture configuration; do not rely on future defaults.

Sources: [principal M1 configuration](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/run_config.json), [model construction](../../src/models/mamba/v24_Ilya/model.py), [initialization/overlay in the runner](../../src/models/mamba/v24_Ilya/training/runner.py).

**Interpretation relative to M1:** M1 used stopped weather-feedback gradients and its reported checkpoint is an SWA. The proposed shared full-weather-gradient, final-checkpoint protocol is a new matched comparison. The new λ=0 model must not be described as reproducing M1. Include M1's actual spectrum and MSE separately.

## 3. Exact training objective

### 3.1 Notation and exact GraphCast grid loss

Let $\widehat y_{bhvzij}$ and $y_{bhvzij}$ be predicted and target physical fields, where $b$ indexes streams, $h$ supervised steps, $v$ variables, $z$ pressure levels (one dummy level for a surface variable), $i$ latitude, and $j$ longitude. Here $B=4$, $H=24$, and $N_{\rm lon}=360$. Let $s_{vz}$ be the corresponding value in `diffs_stddev_by_level.nc`, not the ordinary state standard deviation.

Pressure levels in hPa are

$$
\{50,100,150,200,250,300,400,500,600,700,850,925,1000\}.
$$

Define normalized pressure weights

$$
\pi_{vz}=\begin{cases}
p_z/\sum_{r=1}^{13}p_r,&v\text{ is upper air},\\
1,&v\text{ is a surface variable}.
\end{cases}
$$

This equals the code's mean over levels after multiplying by $p_z/\operatorname{mean}(p)$.

Variable weights $w_v$ are:

| Weight | Variables |
| --- | --- |
| 1 | `2m_temperature`, `temperature`, `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`, `specific_humidity` |
| 0.1 | `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`, `total_precipitation_6hr` |

For latitude $\phi_i$, with $\Delta\phi=\pi/180$, use exact pole-cell latitude weights

$$
\widetilde q_i=\begin{cases}
\sin^2(\Delta\phi/4),&\phi_i=\pm\pi/2,\\
\cos\phi_i\sin(\Delta\phi/2),&\text{otherwise},
\end{cases}
\qquad q_i=\frac{\widetilde q_i}{\sum_r\widetilde q_r}.
$$

The exact grid objective is

$$
\boxed{G=M_{\mathrm{GC,grid}}=
\frac1{BH}\sum_{b,h}\sum_v w_v\sum_z\pi_{vz}
\sum_i\frac{q_i}{N_{\rm lon}}\sum_j
\left(\frac{\widehat y_{bhvzij}-y_{bhvzij}}{s_{vz}}\right)^2.}
\tag{1}
$$

Sum the variable contributions; do not divide by the number of variables or their total weight. All 83 primitive variable/level channels enter the objective: five surface channels plus six upper-air variables at 13 levels. No diagnostic wind-speed channel is an extra training target.

Implementation sources: [exact metric reducer](../../src/models/mamba/v24_Ilya/metrics.py), [GraphCast weighted MSE](../../third_party/graphcast/graphcast/losses.py), [latitude/normalization handling](../../src/models/mamba/v24_Ilya/training/spectral_loss.py).

### 3.2 Spherical analysis convention

Apply the same transform $A$ separately to each full normalized forecast $\widehat y/s_{vz}$ and truth $y/s_{vz}$. Do not transform only the correction or subtract the climatological mean. Retain degrees $0\leq\ell\leq179$ and orders $0\leq m\leq\ell$; degree zero contributes to training.

For a normalized real field $x$, first form

$$
\widetilde x_{im}=\frac1{360}\sum_{j=0}^{359}x_{ij}
\exp(-2\pi\mathrm{i}mj/360).
$$

Use real associated-Legendre functions $p_{\ell m}(\sin\phi)$ with the code's Condon–Shortley phase and normalization
$\frac12\int_{-1}^{1}p_{\ell m}(x)^2\,dx=1$.
For each $m$, fit latitude coefficients by weighted least squares:

$$
(Ax)_{\ell m}=\underset{c}{\arg\min}\,
\sum_i q_i\left|\widetilde x_{im}-\sum_{\ell=m}^{179}
p_{\ell m}(\sin\phi_i)c_{\ell m}\right|^2.
\tag{2}
$$

Equation (2) denotes the corresponding component of the minimizer. Use the existing thin-QR construction rather than a normal-equation inverse. Geometry is constructed in float64; analysis uses float32/complex64 with highest-precision dot accumulation. The source of the basis recurrence, normalization, grid validation, and cached QR weights is [spectral_loss.py](../../src/models/mamba/v24_Ilya/training/spectral_loss.py). Save the latitude/longitude arrays, transform settings and source hash.

This fitted continuous spherical norm is not identical to the exact discrete grid norm. In particular, summing T179 coefficient powers is not an exact substitute for (1).

### 3.3 Stable AMSE excess and outer λ

For one degree and one normalized channel/stream/step, write $f_m=(A\widehat y/s)_{\ell m}$, $t_m=(Ay/s)_{\ell m}$, with multiplicity $\nu_m=1$ for $m=0$, and 2 otherwise. Define

$$
P_f=\sum_{m=0}^{\ell}\nu_m|f_m|^2,\quad
P_t=\sum_{m=0}^{\ell}\nu_m|t_m|^2,\quad
C=\sum_{m=0}^{\ell}\nu_m\operatorname{Re}(f_m\overline{t_m}),
$$

$$
\epsilon=10^{-12},\qquad a=\sqrt{P_f+\epsilon},\quad b=\sqrt{P_t+\epsilon},
$$

$$
Z_\ell=\sum_{m=0}^{\ell}\nu_m
\left|\frac{f_m}{a}-\frac{t_m}{b}\right|^2
+\epsilon\left(\frac1a-\frac1b\right)^2.
\tag{3}
$$

This stabilization is equivalent to appending the same real coefficient $\sqrt\epsilon$ to both coefficient vectors. Thus
$Z_\ell=2(1-\rho_\epsilon)$, where
$\rho_\epsilon=(C+\epsilon)/(ab)$. Calculate (3) directly; do not replace it with a cancellation-prone subtraction of a correlation from one.

With $a_+=\max(a,b)$, $a_-=\min(a,b)$, define

$$
\boxed{D_\ell=a_+(a_+-a_-)Z_\ell.}
\tag{4}
$$

For interpretation, the stabilized degreewise terms satisfy

$$
M_{\mathrm{spec},\ell}=(a-b)^2+abZ_\ell
=\sum_m\nu_m|f_m-t_m|^2,
$$

$$
\mathrm{AMSE}_{1,\ell}=(a-b)^2+\max(a^2,b^2)Z_\ell,
\qquad D_\ell=\mathrm{AMSE}_{1,\ell}-M_{\mathrm{spec},\ell}.
$$

Aggregate the excess with exactly the weights in (1):

$$
D=\frac1{BH}\sum_{b,h}\sum_v w_v\sum_z\pi_{vz}
\sum_{\ell=0}^{179}D_{bhvz\ell}.
\tag{5}
$$

The campaign objective is

$$
\boxed{L_\lambda=G+\lambda D,\qquad
\lambda\in\{0,0.25,0.5,0.75,1\}.}
\tag{6}
$$

Use the stable expression (4), not a numerical subtraction of two large losses. The existing `amse_excess` diagnostic already includes the internal AMSE coefficient: **set internal `amse_lambda=1` and apply the experiment's outer λ exactly once**. Disable internal λ ramps, degree multipliers, and variable-specific spectral multipliers. There is no MGDA, objective RMS normalization, truth-power division, or per-update search in this scalar objective. At λ=0, dispatch directly to the exact-grid-MSE path without the spherical training computation; evaluation still computes spectra.

At λ=1,

$$
L_1=\mathrm{AMSE}_1+(G-M_{\rm spec}),
$$

so the endpoint is a **hybrid**, not pure AMSE. Also, $D$ is an amplitude/pattern adjustment, not simply squared power error: it vanishes when powers match, and it is small when patterns align. Direct power-spectrum diagnostics remain necessary.

Log `grid_mse`, `spectral_mse`, `amse_excess_lambda1`, `lambda_times_excess`, `total_loss`, gradient norm, and update norm separately. For ordinary λ=0 training steps, mark the uncomputed spherical diagnostics as absent, not zero; `lambda_times_excess=0` and `total_loss=grid_mse` remain known. Compute the spherical diagnostics at the scheduled development evaluations. Reuse [mse_amse_excess_loss.py](../../src/models/mamba/v24_Ilya/training/mse_amse_excess_loss.py) and [the stable implementation](../../src/models/mamba/v24_Ilya/training/spectral_loss.py), while adding an ordinary scalar-loss dispatch. The current paired helper alone does not implement this campaign.

## 4. Data, sequence, and reproducible optimization

### 4.1 Splits and common data order

| Purpose | Fixed source and rule |
| --- | --- |
| Training | [Prepared 1° stream](../../data/graphcast/graphcast/dataset/prepared_stream/res1), targets within 2015–2021 only |
| Training split source | [Anchor metadata](../../data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/metadata.json) and [train indices](../../data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/anchors/split_train.npy) |
| Development | The existing 32-start, 40-lead 2022 cohort; exact reference and indices below |
| Evaluation | [Canonical 122-start 2023 manifest](../../artifacts/evaluations/v24_res1_2023_stride3_memory_20261004/initializations_2023.json), 40 six-hour leads |
| Evaluation fields | [2023 evaluation Zarr](../../data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr) |

The prepared store includes 2022. An anchor's year alone does not determine whether all subsequent targets belong to training. Before the campaign, materialize an eligible-segment manifest and check every input/target timestamp for each 96-position segment, including all generated-weather targets. Reject segments with gaps or targets after `2021-12-31T18:00:00`. Store the exact eligible indices and timestamps plus a hash. Do not silently change the store or split between architectures.

**Proposed shared sampling rule:** build chronological, nonoverlapping 96-position segments from valid contiguous training anchors. For epoch $e=0,1,\ldots$, permute segment IDs with NumPy `Generator(PCG64(SeedSequence([seed, e])))`; form consecutive batches of four IDs and drop the final incomplete batch. Within each batch, consume offsets 0, 24, 48, 72 in that order. Save the resulting update-to-segment mapping for all 2,500 updates, common to both methods and every λ at that seed. This seeded permutation is a small addition to the current chronological loader; record it as part of the new protocol. It makes later training-seed comparisons meaningful for full GC as well as for Mamba. Save the NumPy version and the actual mapping, so replay need not regenerate a permutation.

For development, use the `chosen_idx`/`global_chosen_idx` cohort in the [M1 matched32 result](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json):

```text
4, 23, 47, 57, 104, 241, 246, 369,
384, 422, 548, 693, 699, 751, 771, 774,
836, 870, 873, 893, 930, 1006, 1017, 1065,
1117, 1130, 1161, 1181, 1191, 1256, 1294, 1338
```

These are evaluator-selection indices, **not raw Zarr time indices**. Resolve them through the saved protocol and existing loader into an explicit 32-initialization timestamp manifest before training. Reuse its input-window and anchor-history conventions; do not reinterpret index 4 as the fifth weather timestamp. The available protocol is `protocols.exact` in this [campaign shard manifest](../../artifacts/evaluations/v24_mgda_power_2023_20261106/shards/00/manifest.json). Its source dataset is external to this checkout:

```text
della:/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
```

This directory was accessible when preparing the plan. Its metadata hash is recorded in the inventory. The historical `warmup_steps=24` also affects cohort/anchor construction; `cold_full` evaluation with zero state does not imply feeding 24 truth steps to initialize Mamba memory.

The 122 evaluation origins are 00 UTC every third day, January 1–December 30, 2023. Forecast leads are $6,12,\ldots,240$ hours; late-December verification extends through January 9, 2024. The manifest's dates are authoritative. We have already explored 2023: describe this as common, reused evaluation data, not an untouched confirmatory test year. All new selection/tuning uses 2022.

### 4.2 K20/BPTT24 and gradient semantics

For each 24-position window, index predictions $r=0,\ldots,23$. At positions 0–3, construct the two-frame weather inputs from observations. At positions 4–23, update the weather input from previous generated predictions, preserving the same known forcings/static fields as the existing runner. The prediction at position 3 starts the generated-weather chain: the longest linked chain is therefore 21 predictions, though `ar_tail_k=20` counts the 20 generated-input positions. Supervise all 24 predictions with equal weight $1/24$.

Differentiate through both previous weather frames and through Mamba state wherever they depend on trainable parameters within the window. In GC–Mamba, public GC weights are frozen but derivatives through GC's weather inputs remain active. In full GC, GC weights themselves enter the differentiated parameter tree.

At a window boundary, carry Mamba state values but detach their history; restart weather inputs from observations. At a new 96-position segment, reset Mamba state to zero. Full GC has no recurrent memory state. Do one optimizer update per four-stream, 24-position window. Retain post-window state and the exact next data cursor for resumption.

Use the shared, one-step-rematerialized, host-tape implementation in [weather_feedback_step.py](../../src/models/mamba/v24_Ilya/training/weather_feedback_step.py). Its current kernel explicitly freezes GC parameters and adds a residual; the full-GC adapter must change that parameter ownership and prediction composition.

### 4.3 Optimizer and learning-rate calibration

Use ordinary AdamW on the single gradient $\nabla G+\lambda\nabla D$. Clip the global norm of the complete trainable gradient to 1 before applying Adam and parameter groups:

$$
\bar g_k=\frac{g_k}{\max(1,\|g_k\|_2)},\quad
m_k=0.9m_{k-1}+0.1\bar g_k,\quad
v_k=0.98v_{k-1}+0.02\bar g_k^2.
$$

With standard bias correction (first update exponent 1), update each parameter group $r$ by

$$
\theta_{k+1,r}=\theta_{k,r}-\eta_r(k)
\left[\frac{\widehat m_{k,r}}{\sqrt{\widehat v_{k,r}}+10^{-8}}
+10^{-4}\theta_{k,r}\right].
\tag{7}
$$

Apply decoupled weight decay to all trainable parameters, including biases and normalization parameters, matching `weight_decay_policy=all`. GC has one LR multiplier of 1. GC–Mamba spatial parameters have multiplier 1 and Mamba parameters multiplier 3, using the existing parameter-label function. The multiplier scales both the adaptive step and the decoupled decay. Frozen public GC parameters receive neither update nor decay.

For zero-based update counter $k=0,\ldots,2499$, define $\eta_{\min}=0.1\eta_{\max}$ and

$$
\eta(k)=\begin{cases}
\eta_{\max}k/100,&0\le k<100,\\
\eta_{\min}+\dfrac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\dfrac{k-100}{2400}\right)\right],&100\le k<2500.
\end{cases}
\tag{8}
$$

This reproduces the existing schedule indexing: the first update has zero LR but advances optimizer moments; the exact minimum is reached at counter 2500, immediately after the last scheduled update. Count 2,500 updates under this convention. Source: [optimizer builder](../../src/models/mamba/v24_Ilya/training/endpoint_step.py).

Use BF16 model computation with FP32 master parameters/optimizer state, FP32 physical-field and recurrent-state boundaries, and FP32 training loss. Record actual precision settings and software versions. Accumulate evaluation/calibration statistics in float64 on host where practical.

Calibrate learning rates using λ=0 only:

| Method | Candidate peak rates | Selection rule |
| --- | --- | --- |
| Full GC | $10^{-5}$, $3\times10^{-5}$ | Lower pooled 2022 40-lead exact GC loss at update 500; exact tie chooses smaller rate |
| GC–Mamba | Spatial $3\times10^{-5}$, $10^{-4}$; Mamba always ×3 | Same rule |

Configure the full 2,500-update schedule before calibration. Stop each candidate operationally after update 500, score development, then continue the chosen trajectory with its own optimizer/data/state to 2,500. Start the other four λ arms fresh with that method's selected rate. Keep λ fixed from the first update; there is no spectral-weight ramp.

This costs ten retained 2,500-update trajectories plus two losing 500-update candidates: **26,000 total updates**, excluding implementation checks and any later seed repeats. Record both candidates' scores and the deterministic selection. There is no new LR search for each λ.

Save native checkpoints every 250 updates and evaluate development at 500, 1,000, 1,500, 2,000, 2,500. Final update 2,500 is primary regardless of intermediate performance. Intermediate checkpoints explain convergence and measured cost; they are not opportunities to select a better-looking 2023 point.

## 5. Smoothing baseline: exact operator and fitting rule

### 5.1 What receives filtering

Apply both filter families to unmodified GC, all ten final trained models, and M1. If the matched reset reference is included, give it identical filtering treatment. Fit each predictor's filter independently on its own 2022 forecasts, with the same parameter capacity and fitting rules. Keep identity/raw forecasts in every comparison.

Filtering is an **output observer**: score the transformed forecast, but pass the unfiltered forecast to the next AR step. Do not filter ERA5 truth, change forcings, alter Mamba state, or feed the filtered weather back into training/evaluation rollouts.

Let $S$ synthesize a real grid field from the T179 coefficients using the same Legendre basis and inverse-longitude FFT as [spherical_observer.py](../../src/models/mamba/v24_Ilya/spherical_observer.py). Define the exact discrete mean

$$
\mu_q(x)=\sum_i\frac{q_i}{360}\sum_j x_{ij}.
$$

For degree gains $g_\ell$, with $g_0=1$, define

$$
\delta_g(f)=S[(g-1)Af],\qquad
\boxed{F_g(f)=f+\delta_g(f)-\mu_q(\delta_g(f)).}
\tag{9}
$$

The last term is a constant field. It preserves the exact discrete mean, including the small discrete-mean effects of fitted higher modes. The unresolved residual $f-SAf$ stays unchanged. Thus this baseline attenuates represented spherical modes; it does not replace the native field by a T179 projection. When all gains equal one, bypass the transform and return the input exactly. Merely setting $g_0=1$ and reconstructing $SgAf$ would not satisfy these requirements.

### 5.2 Two fixed filter families

**Gaussian:** use

$$
g_\ell(\sigma)=\exp[-\ell(\ell+1)\sigma^2/2],\qquad
\sigma=\sigma_{\rm deg}\pi/180,
$$

with the fixed candidate set
$\sigma_{\rm deg}\in\{0,0.25,0.5,1,2,4,8\}$.
Choose the candidate with the lowest exact development grid MSE for that filter group; an exact tie chooses the smaller width. This parameter is an angular standard deviation, not a latitude-dependent zonal distance.

**Four-band attenuation:** set degree knots $\kappa=(1,10,40,179)$. At positive integer degree $\ell$ between $\kappa_j$ and $\kappa_{j+1}$, set

$$
t=\frac{\log\ell-\log\kappa_j}{\log\kappa_{j+1}-\log\kappa_j},
\qquad g_\ell=(1-t)a_j+ta_{j+1},\qquad 0\le a_j\le1.
\tag{10}
$$

Use the endpoint knot values at degrees 1 and 179 and $g_0=1$. Equivalently, $g_\ell=\sum_{j=1}^4 h_j(\ell)a_j$, where $h_j$ are the piecewise-linear hats in log degree, sum to one for positive degrees, and are zero at degree zero. Gains need not be monotone: the family can represent an intermediate-degree dip. It cannot amplify a positive-degree represented coefficient before roundoff. The explicit discrete-mean correction can change the degree-zero coefficient.

Use four fixed lead groups for both families: **6–24 h, 30–72 h, 78–168 h, 174–240 h**. Fit variable/level-specific parameters, tying U/V gains at each pressure level and tying U10/V10. This gives 69 channel groups × four lead groups: 276 Gaussian widths or 1,104 band gains per model. Other levels/lead groups are fitted independently; there is no extra cross-level or cross-lead regularization.

### 5.3 Reproducible band fitting, without another hyperparameter sweep

For one channel group $C$ and lead group $H_g$, define the pooled development loss

$$
E_{C,H_g}(a)=\frac1{32|H_g|}\sum_{n=1}^{32}\sum_{h\in H_g}
\sum_{c\in C}w_c\pi_c\left\langle
\left[\frac{F_a(f_{nhc})-y_{nhc}}{s_c}\right]^2
\right\rangle_q,
\tag{11}
$$

where $\langle x\rangle_q=\mu_q(x)$. Pool squared errors before dividing or comparing. For a tied wind group, include both components with their canonical weights and normalization; derive wind speed only after filtering both components.

Let $E_0=E_{C,H_g}(1,1,1,1)$. With a fixed, predeclared regularization strength $\eta=0.01$, solve

$$
\boxed{\min_{a\in[0,1]^4}\;\frac{E_{C,H_g}(a)}{E_0}
+0.01\left[\frac13\sum_{j=1}^3(a_{j+1}-a_j)^2
+\frac14\sum_{j=1}^4(a_j-1)^2\right].}
\tag{12}
$$

If $E_0=0$, use identity. Do not add a denominator floor. The first penalty discourages sharp differences between neighboring band gains; the second shrinks toward identity. Equation (12) is a strictly convex box-constrained quadratic. This is a regularized attenuation baseline, not unrestricted optimal smoothing.

An implementation can accumulate sufficient statistics without retaining all forecast grids. Set

$$
B_j(f)=S[h_jAf]-\mu_q(S[h_jAf]),\qquad
F_a(f)=f+\sum_j(a_j-1)B_j(f).
$$

For normalized error $e=(f-y)/s_c$ and normalized bases $b_j=B_j(f)/s_c$, accumulate the weighted means
$q_j=\langle e b_j\rangle$ and $Q_{ij}=\langle b_i b_j\rangle$, with exactly the sample/lead/channel weights in (11). Then

$$
E(a)=E_0+2q^\top(a-\mathbf1)+(a-\mathbf1)^\top Q(a-\mathbf1).
\tag{13}
$$

Use float64 sufficient statistics and deterministic bounded optimization with analytic derivatives. Proposed solver: SciPy `minimize(method='L-BFGS-B')`, start `a=ones(4)`, bounds `[0,1]`, `ftol=1e-12`, `gtol=1e-8`, `maxiter=1000`; record SciPy version, termination status, gradient residual, fitted gains and objective. Compare the returned finite feasible point with identity and retain the lower penalized objective; record a convergence warning if necessary. Nonfinite fits or corrupted statistics are errors.

Fit on all 32 development starts after each model's final checkpoint is fixed. Freeze the complete filter artifact, including source checkpoint hash, data-manifest hash, lead/channel mapping, transform convention, widths/gains and fit statistics, before 2023 scoring. The fixed regularization avoids an additional tuning grid. Report both families separately; do not choose a family by 2023 performance.

Linear spherical filtering may produce negative humidity or precipitation through overshoots. Primary scores use the specified linear operator without hidden clipping; record the negative fraction and minimum for those fields. A predeclared secondary diagnostic may apply `max(0, x)` to humidity/precipitation for **every** predictor, including raw GC. Label that nonlinear sensitivity separately because mean preservation and the quadratic fit no longer apply.

### 5.4 What the baseline can establish

Compare raw GC with independently filtered GC, raw learned predictors with filtered GC, and independently filtered full-GC fine-tuning with independently filtered GC–Mamba. A learned advantage that survives equal filtering opportunities is stronger evidence than improvement over raw GC alone.

The original GraphCast paper also fitted isotropic spherical filters independently by variable, level and lead on a separate year, then evaluated on another year. Our constrained families and **exact grid loss** are deliberately specified here; they are not an exact reproduction of GraphCast's optimal spectral-MSE filter. See [GraphCast, Supplement §7.4, Optimal blurring](https://arxiv.org/html/2212.12794v2).

Existing saved zonal power summaries cannot recover the exact global filtered MSE: phase, global latitude structure and all 83 channels are required. The observer must receive complete physical forecasts during the planned evaluations. The raw and both filtered predictions can be scored in one forecast pass once filters are frozen.

## 6. Evaluation, plots, and inference

### 6.1 Forecast protocol and MSE

Evaluate every final checkpoint on the same 122 starts and all 40 leads. Use two observed input frames, zero Mamba state, cold full-feedback forecasts, residual scale 1, and evaluation PRNG seed 0. Forecasting after initialization uses no verifying truth. Full GC uses the same known forcings, input normalization, output decoding and timing conventions.

For model $a$, let $G_{a,n,h}$ be (1) with one initialization and one lead. Define

$$
\overline G_a=\frac1{122\cdot40}\sum_{n,h}G_{a,n,h},\qquad
I_a=100\left(1-\frac{\overline G_a}{\overline G_{\rm GC}}\right).
\tag{14}
$$

Use one canonical public-GC result as denominator for all arms, after checking model/data/protocol identities. Do not average per-date percentage improvements. Report exact loss and relative improvement for the complete rollout, each lead, and day 10; include physical RMSE by variable/pressure level. For comparisons of two models, report their difference in loss or in percentage points of the common-GC improvement.

Repeated public-GC evaluations can measure numerical variation. Preserve those measured differences and bounds as warnings; use the canonical baseline consistently. Nonfinite results, missing required forecast cells, and actual checkpoint/input mismatches remain errors.

### 6.2 Spherical spectra aligned with training

Use **physical fields**, without training normalization, for interpretable per-channel spectra. Apply the transform in (2) to raw/filtered forecasts and truth. Save per-initialization, per-lead, per-channel, per-degree powers $P_f,P_t$, real cross-power $C$, and forecast-error power

$$
P_e=\sum_m\nu_m|f_m-t_m|^2=P_f+P_t-2C.
$$

Pool raw statistics over starts first, denoting means by a bar. Then plot

$$
R_{a,c,h,\ell}=\frac{\overline P_{a,c,h,\ell}}{\overline P_{t,c,h,\ell}},\qquad
\rho_{a,c,h,\ell}=\frac{\overline C_{a,c,h,\ell}}
{\sqrt{\overline P_{a,c,h,\ell}\overline P_{t,c,h,\ell}}},
$$

$$
A_{a,c,h,\ell}=\frac{|\overline P_{a,c,h,\ell}-\overline P_{t,c,h,\ell}|}
{\overline P_{t,c,h,\ell}}.
\tag{15}
$$

Retain degree zero in artifacts, but exclude it from the primary anomaly-spectrum score. Define the spherical macro-score

$$
E_{\rm sph}(a)=\frac1{|\mathcal I|}\sum_{(c,h,\ell)\in\mathcal I}A_{a,c,h,\ell},
\quad\mathcal I=\{83\text{ channels}\}\times\{40\text{ leads}\}\times\{1,\ldots,179\},
\tag{16}
$$

restricted to truth-positive bins if any exact zeros occur; save the mask and excluded counts. Use the same truth-derived mask for all methods and filters. Report 100×(16) as percent mean relative power error. This macro-average gives channels equal reporting weight; it is a diagnostic and is not the weighted training loss. Also show per-field curves and the complete degree/lead arrays, so a scalar cannot hide an intermediate-scale deficit.

Do not use the training epsilon to silently floor diagnostic denominators. Zero truth power makes a power ratio or relative mismatch undefined. Zero forecast power with positive truth power gives a valid power ratio of zero and mismatch of one, but undefined correlation. Undefined values remain NaN with counts. If reporting the previously requested GC-relative mismatch, use

$$
Q_{a,c,h,\ell}=\frac{|\overline P_a-\overline P_t|}
{|\overline P_{\rm GC}-\overline P_t|}.
\tag{17}
$$

This is **power mismatch relative to GC**, not forecast-error power. A zero GC mismatch makes (17) undefined. For an overall relative mismatch reduction, compute $100[1-E_{\rm sph}(a)/E_{\rm sph}(\mathrm{GC})]$, not the mean of (17); this aggregate is undefined if $E_{\rm sph}(\mathrm{GC})=0$.

Also save the exact discrete grid norm, synthesized-projection error and unresolved grid error using [the spherical observer](../../src/models/mamba/v24_Ilya/spherical_observer.py). Do not call the difference between continuous spectral MSE and grid MSE “unresolved energy”; those are different norms.

### 6.3 Existing zonal diagnostics for continuity

Retain [the established zonal definition](../../src/models/mamba/v24_Ilya/spectral_diagnostics.py): remove each latitude row's zonal mean, compute `rfft(norm='forward')`, use both hemispheres' $30^\circ\le|\phi|\le60^\circ$ rows with normalized cosine weights, and double positive one-sided powers except Nyquist $k=180$. Set $k=0$ to zero. This differs from global spherical degree.

The nine primitive fields are T2m, U10, V10, MSLP, TP6, T850, Q700, U850 and V850. Derived WS10 and WS850 are additional diagnostics computed in physical space from U/V before the transform; do not count them twice in the nine-field score. For continuity with the retrieved historical λ plots, define

$$
E_{\rm zonal}^{\rm legacy}(a)=\frac1{9\cdot4\cdot180}
\sum_{c\in C_9}\sum_{h\in\{24,72,168,240\}}\sum_{k=1}^{180}
\left|\frac{\overline P_{a,c,h,k}}{\overline P_{t,c,h,k}}-1\right|,
\tag{18}
$$

with the same explicit zero-denominator handling. Label (18) separately from the new full-rollout/global score (16). Save all 40 leads even when only days 1/3/7/10 appear in a figure.

### 6.4 Tradeoff plots, selection, and uncertainty

Required figures/tables:

1. Exact rollout MSE versus spherical power error, with method colors, λ labels and separate raw/Gaussian/band-filtered markers; include public GC and M1.
2. The same plot with the legacy zonal score, to connect to existing results.
3. Forecast/truth power, correlation, and forecast-error spectra at days 1, 3, 7 and 10; display the intermediate-degree region without hiding either attenuation or amplification.
4. MSE and spectral metrics versus training updates and measured cumulative GPU-hours. Show all fixed development checkpoints, plus inference latency and memory.
5. A table of all ten final models, both filters and their fitted parameters, rather than just the selected best model.

If a single operating point is needed, predeclare this rule: within each method's **raw** λ grid, choose the lowest development MSE subject to development $E_{\rm sph}$ being no larger than public GC's, breaking an exact tie by smaller λ. If no arm qualifies, report no qualifying point; still finish and report every valid run. This is a scientific reporting criterion, not a stop condition. Keep all filtered outcomes visible instead of using 2023 to select a filter family. An alternative application threshold must be fixed on 2022 before 2023 scoring.

For differences on the 122 dates, use the existing paired non-wrapping moving-block bootstrap: blocks of five consecutive starts (12 days between first and last origin), 10,000 replicates, seed 20260916, percentile 95% intervals. Reuse [the matched-evaluation implementation](../../scripts/analyze_models/report_v24_matched_reset_2023.py) and its helper rather than inventing an incompatible resampler. Resample complete paired dates, all leads and all arms together. Recompute nonlinear spectral ratios from each draw's pooled powers. These are pointwise fixed-checkpoint intervals for the reused evaluation cohort, not simultaneous confidence bands, selection-corrected claims, or estimates of training-seed variability.

After the primary comparison, repeat the development-selected GC/GC–Mamba pair with seeds 23 and 24 using the same frozen rates, λ values, budgets and seeded segment rule. This adds four runs, not a second full grid. Freeze the selection before the repeats. Report seed-level results separately from date-bootstrap uncertainty.

## 7. Implementation work and checks

| Component | Existing evidence/code | Required change |
| --- | --- | --- |
| Scalar loss | Exact grid MSE, AMSE and stable excess already exist | Add explicit scalar `G + outer_lambda * D`; λ=0 fast path; separate logs; internal λ=1 |
| Training configuration | [config.py](../../src/models/mamba/v24_Ilya/training/config.py) | Add an unambiguous scalar-loss kind/weight; permit it in full-weather-feedback checks; bypass paired MGDA/RMS/backtracking machinery |
| GC–Mamba | Fresh initialization, optimizer, shared weather BPTT and native resume | Wire the new scalar objective and campaign configuration |
| Full GC | Public loading; [legacy fine-tuner](../../src/models/graphcast/training/finetune.py) | Add a full-GC predictor/parameter adapter to the shared window engine, optimizer, checkpoint format and evaluator |
| Sampling/splits | [data.py](../../src/models/mamba/v24_Ilya/training/data.py), anchor manifests and cursor | Validate all target timestamps; materialize common segment/order manifests; add seeded epoch permutation/replay |
| Smoothing | Spherical analysis, basis/synthesis in observer, exact grid reducers | Extract reusable synthesis; implement mean-preserving filters, sufficient-statistic fit, filter artifact, streaming scoring |
| Evaluation | [observer interface](../../src/models/mamba/v24_Ilya/evaluation.py), [existing worker](../../scripts/analyze_models/evaluate_v24_amse_regime.py) | Dispatch by model family, attach raw/filtered/all-channel observers, preserve identical forecast semantics |
| Campaign/report | Existing manifests, Slurm wrappers, reporting patterns | Generate ten arms plus calibration; immutable source/config/data hashes; all-model tables and tradeoff figures |

Proposed configuration names such as `loss_kind=gc_mse_amse_excess`, `spectral_excess_weight`, and `model_family=full_gc` are **specification names, not currently supported CLI options**. Publish actual invocation commands after implementing and validating the entry points.

For full GC, the differentiated parameter tree must contain public GC parameters. Removing a freeze flag from the existing residual-only kernel is insufficient. A common predictor interface should return `(full_physical_forecast, next_state)`; GC–Mamba returns frozen-GC-plus-correction, while full GC returns its direct forecast and an empty recurrent state. Do not add the original frozen forecast to an already full GC forecast. The legacy fine-tuner's short-rollout objective and weight-only checkpointing do not provide the matched protocol by themselves.

Required focused checks before using the new campaign:

- Equation (6) and gradients agree with its separate components; λ=0 equals exact grid MSE; internal λ is not accidentally applied twice. Exercise zero-power and unequal-power cases for (3)–(4).
- On a small deterministic sequence, streamed BPTT agrees with direct autodiff through both weather inputs and recurrent state. Verify the documented boundary detaches and teacher prefix.
- Initial full outputs match public GC; trainable parameters change in the intended tree; frozen GC is unchanged for Mamba; full GC is genuinely trainable.
- Native save/resume preserves parameters, optimizer, RNG, next data mapping/cursor and carried memory. Record finite numerical drift and its bounds as warnings, not a hard acceptance gate.
- Filter identity returns the original array; constant fields and discrete means are preserved; a represented harmonic receives the prescribed gain modulo the explicit mean correction; raw scoring matches the exact reducer; fitted statistics match direct grid evaluation on a small fixture.
- Forecast/date/variable/level identities match across arms. Nonfinite computation, corrupted or mismatched inputs/checkpoints, and incomplete required records are hard failures.

Finite stalls, rising losses, finite restart drift, or weak intermediate skill do **not** cancel valid bounded runs or their evaluations. Do not add a monotonic-descent requirement or a noise-derived acceptance margin. Preserve warnings, measured noise and whether an observed decrease exceeds that noise. Completing training and demonstrating a scientific improvement are separate outcomes, as required by [AGENTS.md](../../AGENTS.md).

## 8. Resource plan and order of work

Full fine-tuning is the primary proposed vanilla-GC comparator. Its feasibility with the shared K20 full-feedback engine has not yet been measured. LoRA is a contingency if full tuning cannot fit after sensible rematerialization/streaming; it is not an interchangeable substitute for all-parameter fine-tuning. If needed, define and label the adapter locations/rank and count that as a separate protocol revision before the λ sweep.

Start useful retained training chunks with one A100 80 GB, eight CPUs, and 320–384 GiB host RAM as a **measurement envelope**, not a proven fit. Measure actual peak GPU/CPU memory, startup/compilation, and steady update times for both methods, including positive λ. Keep `max_steps=2500` fixed and use an operational stop for profiling; resume exactly. Do not shorten the optimization schedule to make a profile run.

Existing resource observations provide context only:

| Evidence | Observation | Limitation |
| --- | --- | --- |
| October scalar 500-update continuation, jobs `15025777_0–3` | About 4 h 8 min, 231–234 GiB peak host RSS, 320 GiB request | Stopped weather gradients; does not price the new full-gradient scalar runs |
| Shared full-weather-gradient MGDA, jobs `16402493_0–1` | Steady p90 about 137 s/update; host RSS about 235 GiB | Multiple objective/backward operations; does not price one-gradient AdamW or full GC |
| Full-GC K20 scalar training | Unmeasured | Need retained first-chunk measurements |

Sources: [historical scalar submission](../../artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004/submissions/pilot_train.json), [MGDA development/resource history](../../research/directions/r005-mgda-accuracy-spectral-adjustment/devel.md).

After measurement, estimate GPU-hours by summing $(\text{startup seconds}+N_{\rm updates}\times\text{steady seconds/update})/3600$ for each job part, including losing calibration candidates, evaluation and restarts. Choose walltime/host-memory headroom from measured variability and scheduler limits. Report actual Slurm elapsed/allocated GPU time afterward; avoid a numerical completion-time promise based on the historical regimes.

Implementation/execution sequence:

1. Freeze this scientific protocol, eligible training windows, 32 development dates and 122 evaluation dates. Implement the scalar loss, full-GC adapter and manifest replay, then run the focused checks.
2. Begin four λ=0 LR candidates, retaining their 500-update states and measuring real resources. Implement/filter-check the observer while these run.
3. Select one LR per method on 2022. Continue the two winners, and start eight independent positive-λ arms. Let every valid arm reach its planned endpoint.
4. Score prescribed development checkpoints. Fit both smoothing families on final models' 2022 forecasts and freeze the artifacts.
5. Evaluate all final models and references on the common 2023 cohort, scoring raw and filtered outputs in the same pass. Reuse existing forecast artifacts only when their fields, dates and provenance actually support all requested metrics.
6. Produce complete tradeoff/cost tables, spectra and paired uncertainty. Add the completed extreme-event results when available. Then decide whether the selected-pair seed repeats or an AR-length follow-up is justified.

## 9. Extreme events and optional AR study

The separately submitted [extreme-event campaign](../../artifacts/evaluations/v24_extremes_2023_20261106/campaign.json) evaluates principal M1 and the existing relative-power MGDA SWA against GC on the full 122-start set. It covers temperature, wind and 24-hour precipitation occurrence/intensity; [r006](../../research/directions/r006-extreme-event-evaluation/overview.md) is the authoritative status and protocol record. It does not yet evaluate future scalar-λ checkpoints. Preliminary results are not evidence that full extreme-event skill is preserved.

Once a new operating point has been selected using 2022, reuse that protocol/observer for the selected fresh GC and GC–Mamba models if making an extreme-event claim about them. Do not automatically extend the claim from M1 or MGDA to a new checkpoint. Report raw extremes as primary; any filtered-extremes assessment is separately labeled.

Shorter AR training is optional. Complete the λ comparison first because changing the training horizon simultaneously changes another major factor. A later K12/K20 comparison should fix the loss and LR recipe, use fresh starts for both methods, and explicitly define the teacher-prefix/BPTT convention. With a fixed four-position teacher prefix, K12 implies BPTT16 and needs backend support beyond the currently supported 2/4/8/24 lengths; K12 with BPTT24 instead implies 12 teacher-input positions and is a different intervention. Do not silently mix those definitions.

The original GraphCast work fine-tuned with progressively longer rollouts up to 12 steps. Its Supplement §7.4.4 reports that shorter-horizon models can retain sharper fields, while longer-horizon models retain an accuracy advantage even after both receive optimal filtering ([training and horizon/filtering analysis](https://arxiv.org/html/2212.12794v2)). That motivates examining horizon as a control variable, but does not establish that K12 will repair this model's spectrum or that additional K12 updates will cross an improvement threshold. Existing K12 learning curves are exploratory evidence, not a guarantee of either outcome.

## 10. Reproducibility artifacts and proposed output layout

The following output root is **proposed and is not an existing trained campaign**:

```text
artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261106/
  campaign.json
  source/                         # frozen runtime, not a pointer to mutable source
  manifests/                      # train segments/update mapping, dev32, eval122
  initializations/                # method/seed step-zero weights and overlay reports
  calibration/{method}_{lr}/
  runs/{gcm|gcft}_s22_l{000|025|050|075|100}/
    run_config.json
    provenance.json
    checkpoints/checkpoint_step00002500.pkl
    train_metrics.csv
    development/{00500|01000|01500|02000|02500}/
    filters/{gaussian|four_band}/filter.json
    evaluation_2023/{raw|gaussian|four_band}/
  report/
```

Native checkpoint payloads must include a model-family tag, trainable parameters, optimizer moments/counter, RNG, next sampler cursor and permutation mapping, carried recurrent state, architecture/objective configuration, public-baseline/statistics identities, and runtime/manifest hashes. Preserve zero-output initialization and all resume provenance. An optional standard GC `.npz` export is for inference exchange, not an optimizer-resume checkpoint.

Record environment/package versions (including JAX/jaxlib, CUDA, Haiku, Optax, NumPy and SciPy), device type, precision/XLA flags, source snapshot hash, all configuration values, data identity/timestamp manifests, checkpoint hashes and Slurm resource/accounting records. Current checkout HEAD is `fc6a306c32d8ea4a84abd117c5663bd933367638`, with relevant uncommitted work present; **HEAD alone does not identify the current runtime**. Freeze an immutable implementation snapshot once the missing code exists.

This plan's companion [verified inventory](v24_fresh_gc_mamba_gc_lambda_smoothing_inventory_2026-11-06.json) contains existing artifact paths, sizes and SHA-256 values, plus development/evaluation cohort locators. Checkpoint hashes were recomputed from file bytes without unpickling. Dataset-directory existence and metadata were checked; this is not a hash of every data chunk. Verification was performed on 2026-11-06 local time (2026-11-07 UTC).

## 11. Existing checkpoint locators

All relative paths below are relative to the workspace root. Markdown links are relative to this document. Files on this filesystem will not be uploaded by sharing this Markdown; collaborators need access to the checkout/data mount or an explicit copy.

### 11.1 Public initializer and principal references

| ID | Role | Existing checkpoint | Configuration |
| --- | --- | --- | --- |
| `public_gc` | Only pretrained GC initializer for both new methods; unmodified baseline | [GraphCast-small public NPZ](<../../data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz>) | Embedded model/task configuration; full path below |
| `M1` | Principal historical K20 GC–Mamba; separate reference | [M1 SWA](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl) | [run_config.json](../../artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/run_config.json) |
| `R_matched` | Matched reset-trained control; optional existing reference | [Reset SWA](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl) | [run_config.json](../../artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/run_config.json) |

```text
public_gc:
data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz

M1:
artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/swa/swa_step00500-02000.pkl

R_matched:
artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015/runs/branch/swa/swa_step00500-02000.pkl
```

### 11.2 Historical spectral checkpoints: context, not new-arm initialization

| ID | Existing checkpoint | Interpretation |
| --- | --- | --- |
| `mature_september24_mse_parent` | [September MSE SWA](../../artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924/runs/mse_fp32/swa/swa_step00500-02000.pkl) | Parent of the matched October 500-update pilot; not principal M1 |
| `historical_lambda025_native500` | [λ0.25 at update 500](../../artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004/runs/amse_l025/checkpoints/checkpoint_step00000500.pkl) | Mature-parent continuation, spectral-MSE interpolation, stopped weather gradient |
| `historical_lambda050_native500` | [λ0.5 at update 500](../../artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004/runs/amse_l050/checkpoints/checkpoint_step00000500.pkl) | Same matched pilot; confirms λ0.5 has already been tested in that older regime |
| `historical_lambda010_swa` | [Earlier λ0.1 SWA](../../artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924/runs/amse/swa/swa_step00500-02000.pkl) | Different initialization/training history from the 500-update pilot |
| `historical_lambda100_swa` | [Earlier λ1 SWA](../../artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919/runs/amse/swa/swa_step00500-02000.pkl) | Separate earlier regime; cannot join the pilot as a controlled five-point sweep |
| `relative_power_mgda_swa` | [Relative-power MGDA late SWA](../../artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_recovery_20261007/runs/mgda/swa/swa_step01100-02000.pkl) | Existing annual-spectrum/extremes reference; distinct objective from (6) |

The old scalar objective was $M_{\rm spectral}+\lambda(\mathrm{AMSE}_1-M_{\rm spectral})$, with continuation/ramping choices recorded in its configurations. These checkpoints are not fresh runs of (6). Existing comparison figures and provenance: [MSE/power overview](../../plots/analyze_models/images/resolution_eval/amse_lambda_comparison_20261106/mse_and_power_overview.png), [numeric summary](../../plots/analyze_models/data/resolution_eval/amse_lambda_comparison_20261106/summary.csv), [source records](../../plots/analyze_models/data/resolution_eval/amse_lambda_comparison_20261106/provenance.json).

### 11.3 Verified checkpoint SHA-256

| ID | SHA-256 |
| --- | --- |
| `public_gc` | `e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29` |
| `M1` | `b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0` |
| `R_matched` | `db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6` |
| `mature_september24_mse_parent` | `aa8043a6f3c03c7d994a9aba0db06dfd93ba5a7072743ce99475fc73f087cd98` |
| `historical_lambda025_native500` | `7fef3311c0cf9141dbdf4d03501d3263ecd481ad79d3a4f8c1c9cd979050b869` |
| `historical_lambda050_native500` | `05c77b85ee198bbabe20b09ff476f5b6859122779438a30759dd33659737d065` |
| `historical_lambda010_swa` | `1718971910e144262831116cd658c9c7eb9a93646f8a7047a55ac7997b1571af` |
| `historical_lambda100_swa` | `3521f2b0cabb3cca2bda74f82fdaaee796375e3baf58ee19a0609952c31d842b` |
| `relative_power_mgda_swa` | `1398d7faab0881983764a77f1a3db2b2ee9341deed9bf34a2451f46413135284` |

Normalization files are in [stats/](../../data/graphcast/graphcast/stats):

| File | SHA-256 |
| --- | --- |
| `mean_by_level.nc` | `e6e724b94cd27707903cdc5cf5784d420bab306f722e631f4455e25c96da9931` |
| `stddev_by_level.nc` | `29f941d8c34906b87a1745f93b2308d1e169ec9bbdc4c5491479c771fe92c3cb` |
| `diffs_stddev_by_level.nc` | `e688fc51fe72e1dc897678ef5ce9889add64e28e13f3cec4758d8e47d429c1ce` |

The canonical 122-start manifest SHA-256 is `eca49e6da7005d6731b7e947055fc498b943d3907c2773a4be15f8e1bb51ab49`.

### 11.4 Safe commands to locate and verify existing artifacts

These commands inspect existing files; they do not launch experiments.

```bash
cd /scratch/gpfs/DABANIN/iv9432/Weather_global

# List the main existing checkpoint families.
rg --files artifacts/checkpoints/v24_Ilya | rg 'res1_(batch4_temporal_lr_20260908|selected_matched_reset_20261015|amse_joint_regime_20261004)/.*(swa_step|checkpoint_step).*pkl$'

# Verify the public initializer. Keep the filename quoted because it has spaces.
sha256sum 'data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'

# Use the project environment before running Python.
source scripts/graphcast_env.sh
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
inventory = json.loads((root / 'docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_inventory_2026-11-06.json').read_text())
for item in inventory['checkpoints']:
    record = item['checkpoint']
    path = root / record['relative_path']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['sha256']:
        raise RuntimeError(f"Checkpoint hash mismatch: {item['id']}: {path}")
    print(item['id'], path, digest.hexdigest())
PY
```

## 12. Primary method references

- [GraphCast paper and supplement](https://arxiv.org/html/2212.12794v2): original model/training method and Supplement §7.4 independent spherical-filter comparison. The proposed training schedule/filter family is our controlled experiment, not a claim to reproduce every original training detail.
- [AMSE formulation, equations 5–6 and §§2.2–2.3](https://arxiv.org/html/2501.19374v2): amplitude/pattern decomposition. Equations (3)–(6) above specify the local stabilization and exact-grid anchoring needed to reproduce this campaign.
- [Local stable spectral loss](../../src/models/mamba/v24_Ilya/training/spectral_loss.py), [shared full-weather-gradient backend](../../src/models/mamba/v24_Ilya/training/weather_feedback_step.py), [spherical diagnostics](../../src/models/mamba/v24_Ilya/spherical_observer.py), and [zonal diagnostics](../../src/models/mamba/v24_Ilya/spectral_diagnostics.py): implementation sources to snapshot when the missing campaign code is added.
