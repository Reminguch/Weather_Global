# residualmamba_NGCM — open-loop GraphCast-Mamba residual on a frozen NeuralGCM

**Status:** normative specification, with a new implementation in
`residualmamba_ngcm/` on `residualmamba_NGCM_rebuild_20261104`.
See [OPERATIONS.md](OPERATIONS.md) for commands, validation evidence and pending
data-dependent checks. This does not establish forecast improvement. Everything below is
normative: if the code disagrees with this file, the code is wrong or this file
must be versioned first.

**Runtime amendment, 2026-11-04 (`deterministic_gpu_v1`).** The user requested
that GPU restart drift be fixed. All production stages now configure XLA before
backend initialization with `xla_gpu_enable_scatter_determinism_expander=true`,
`xla_gpu_exclude_nondeterministic_ops=true`, `xla_gpu_deterministic_ops=true`,
and `xla_gpu_autotune_level=0`. Restored arrays are explicitly placed on device.
The execution policy is part of artifact/run identities. It changes numerical
execution, not the architecture, loss, optimizer or data split. Qualification
includes an actual trainer continuation test across separate Python processes
on H200. Forecast quality and per-update loss decrease remain separate questions.

**One-sentence summary.** Take the frozen deterministic NeuralGCM 2.8° model
(NGCM), roll it out exactly as NGCM itself would, and train a GraphCast-style
encoder–processor–decoder with interleaved Mamba memory to predict
`ERA5 − NGCM` in decoded weather space. Use the *same open-loop logic, sequence
layout, normalization, loss family, optimizer and evaluation protocol* as our
GraphCast residual-Mamba (v24 cached open loop ≡ v22clean residual Mamba with
`feedback_mode=baseline`). NGCM-specific details (grid, fields, forcing, native
rollout) follow NGCM.

Contents

1. [Scope, prerequisites, non-goals](#1-scope-prerequisites-non-goals)
2. [What the model is, theoretically](#2-what-the-model-is-theoretically)
3. [Data contract (grid, fields, levels, splits, forcing, array layout)](#3-data-contract)
4. [Time indexing — the part that is easiest to get wrong](#4-time-indexing)
5. [Residual inputs, outputs, statistics, normalization](#5-residual-inputs-outputs-statistics-normalization)
6. [Model](#6-model)
7. [Loss](#7-loss)
8. [Training loop](#8-training-loop)
9. [Evaluation and controls](#9-evaluation-and-controls)
10. [Stages, package layout, code reuse map](#10-stages-package-layout-code-reuse-map)
11. [Tests that must pass before any production run](#11-tests-that-must-pass-before-any-production-run)
12. [Consistency check against our GraphCast residual-Mamba](#12-consistency-check-against-our-graphcast-residual-mamba)
13. [Common mistakes (checklist)](#13-common-mistakes-checklist)
14. [Open decisions](#14-open-decisions)

All repository paths are relative to the root of the `Weather_Global` repository
(`minimalistic_code` branch, which also contains `src/models/neuralgcm_residual/`).

---

## 1. Scope, prerequisites, non-goals

### In scope

* **Open loop only** (`feedback_mode = "baseline"`): the corrected forecast is
  never fed back to NGCM. NGCM's trajectory is the native NGCM forecast.
* A residual branch with the GraphCast mesh architecture and interleaved Mamba
  memory, as in v24/v22. The head is zero‑initialized.
* Supervision in **decoded weather space** (7 fields on pressure levels) against
  ERA5 regridded to the NGCM grid.
* Evaluation against the native NGCM forecast on identical origins.

### Prerequisites (must already exist on disk — this package never downloads anything)

| Item | Requirement | Checked by |
|---|---|---|
| Python env | `neuralgcm==1.2.2`, `dinosaur`, `jax`, `haiku`, `optax`, our `third_party/graphcast` | `FrozenBackbone.load` pins `neuralgcm==1.2.2` |
| NGCM checkpoint | Deterministic **2.8°** pickle already on local disk. You pass its path and expected sha256 | `FrozenBackbone.load(path, expected_sha256)` |
| Prepared ERA5 store | `PreparedStore` root with `manifest.json` + `READY.json`: every 6‑hour frame from 2015‑01‑01T00 to 2023‑12‑31T18 (13,148 frames) on the NGCM data grid | `require_complete_experiment_data(store)` |
| GC‑small ModelConfig | The GC checkpoint used by v24. **Only its `ModelConfig` is read**; no weights are used | §6 |

If any of these is missing, the stage must fail with a clear message. It must
not fall back to downloading. Do not call `backbone.download_checkpoint` or
`data.prepare_era5`.

### Non-goals (do not implement here)

* Closed loop of any kind (`closed_loop_sg`, `closed_loop`, re-encoding corrected
  forecasts, or native-state increments).
* The native K20 design on the `residual_NGCM` branch
  (`trajectory_training.py`, `paper_loss.py`). It differs on almost every axis:
  native-state correction, one native frame as input, per-origin episodes,
  five-term spectral loss, lead down-weighting and an adaptive controller.
  Do not mix pieces from it into this package.
* NGCM's own training loss or its 24 h statistics (`paper_statistics.py`).

---

## 2. What the model is, theoretically

Notation: `x_t` is the ERA5 truth at 6 h index `t`. `b_k` is NGCM's decoded
forecast at lead `k` from origin `o`. `c_k` is the residual branch output.
`h_k` is the Mamba memory.

```
b_k = D( A^k( E(x_o, F_o) ), F_o )            # native NGCM rollout, frozen, θ-independent
y_k = b_k + c_θ(inputs_k, h_{k-1})            # corrected forecast (levels ≥ 50 hPa only)
r_k = x_{o+k} − b_k                           # residual target
```

* **Open loop = learned post-processing with memory.** Because `b_k` does not
  depend on θ, training is supervised regression on a fixed set of
  trajectories. The trajectories seen in training and evaluation come from the
  same distribution (native NGCM rollouts), so there is no exposure bias.
* **The ceiling is the best possible post-processor of NGCM's trajectory**,
  roughly `E[r_k | truth at the origin, NGCM trajectory up to k]`. The branch
  cannot change NGCM's dynamics, because its errors are never fed back. This is
  the same class of model as our GC open-loop results (paper Fig. 8 is open
  loop).
* **Where gains come from.** Two sources: (i) removal of lead-dependent
  systematic error, and (ii) MSE shrinkage toward the conditional mean at long
  leads. Shrinkage lowers RMSE but also lowers spectral power (smoothing), so
  spectra must be reported (§9). Control C1, a lead-dependent bias table,
  separates (i) from what memory adds.
* **What memory can carry in open loop.** It carries information about the
  truth at the origin, NGCM's own trajectory so far and the implicit lead time.
  It is *not* Mori–Zwanzig memory of the true system beyond the origin.
  Controls C2 and C3 (§9) test this.
* **Why `b_{t+1}` is an extra input (an NGCM-specific change).** In GC the
  baseline forecast `b_{t+1} = G(window_t, forcings)` is a deterministic
  function of the residual's own inputs, so the GC residual is implicitly
  conditioned on its baseline. In NGCM, `b_{t+1} = D(A(z_t))` depends on the
  native state `z_t` (32 sigma levels, 193 native channels). That state cannot
  be recovered from decoded frames, because `E(D(z)) ≠ z`. Passing `b_{t+1}`
  explicitly restores the GC property that "the residual sees what the
  baseline predicted". Control C4 removes it to give the GC-literal variant.

---

## 3. Data contract

| Quantity | Value | Source of truth |
|---|---|---|
| Backbone | NGCM v1.2.2, deterministic 2.8° | `FrozenBackbone.manifest()` |
| Horizontal grid | Gaussian, **128 lon × 64 lat** nodal; latitudes are Gauss–Legendre nodes, **ascending** (S→N) | `model.data_coords.horizontal`; must equal `model.model_coords.horizontal.nodal_shape` (assert) |
| Internal timestep | 1 h; `advance_6h` = `fori_loop` of **6** `model.advance` calls | `FrozenBackbone.internal_steps` (assert == 6) |
| Decoded fields (`FIELDS`, this order) | `temperature`, `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `specific_humidity`, `specific_cloud_ice_water_content`, `specific_cloud_liquid_water_content` | `neuralgcm_residual/config.py` |
| Pressure levels | 37 ERA5 levels, ascending: 1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000 hPa | `model.data_coords.vertical.centers` (assert equal to this list) |
| Corrected levels `P` | the **29 levels with p > 30 hPa** (50…1000), i.e. indices 8…36 | §5 |
| Forcing fields | `sea_surface_temperature`, `sea_ice_cover` | `FORCING_FIELDS` |
| Array layout (store, NGCM decode) | 3‑D fields `[level, lon, lat]`, forcing `[lon, lat]`; **longitude first** | `prepare_era5`, `PreparedStore.frame` |
| Splits | train 2015–2021, val 2022, test 2023 | `data.split_for` |
| Not available | 2 m T, 10 m wind, MSLP, precipitation, vertical velocity | — |

**Forcing policy (`lag24h_then_persist_at_origin`).** This is exactly
`PreparedStore.inputs_and_forcing(model, t)`:

```
inputs  = {FIELDS at t, sim_time(t)}
forcing = {SST[t−24h][None], sea_ice[t−24h][None], sim_time(t)}
```

For one rollout from origin `o`, call this **once**. Then pass the same
`forcing` to `encode`, to every `advance_6h`, and to every `decode` of that
rollout. This matches how the existing `neuralgcm_residual/evaluate.py` defines
the native NGCM baseline. The cache (§4) and the evaluation (§9) must use this
same definition. Never switch to time-varying forcing on only one side.

---

## 4. Time indexing

### 4.1 The GraphCast reference (v24 cached open loop; verified ≡ v22clean)

The sequence constants are: `segment_steps=96`, `bptt_steps=24`,
`ar_tail_k=20`, `truth_prefix_steps = bptt − ar_tail_k = 4` and
`input_steps=2`. A **chunk** is 24 consecutive anchors `a_0 … a_23`, where
`a_i = a_0 + i` and the anchor is the time of the *newest* input frame. The code
(`baseline_cache.iter_trajectory`, `frame_data.load_endpoint_frame_batch`) does
this:

```python
frames = [x[a0-1], x[a0], x[a0+1], x[a0+2], x[a0+3]]  # 5 teacher truth frames
for i in range(24):
    window = (frames[i], frames[i+1])                  # input window of step i
    pred   = G(window, forcings at target time a0+i+1) # frozen baseline
    target = x[a0+i+1]
    if 3 <= i < 23:                                    # truth_prefix-1 <= i < n-1
        frames.append(pred)                            # feed back the BASELINE prediction
```

So:

* steps 0, 1, 2 are one-step forecasts from truth;
* step 3 is the **first step of one rollout started from truth at `a0+3`**;
* steps 4…23 continue that rollout.

The tail therefore covers leads 6 h…126 h, which is 21 rollout steps
(GC‑definition **K = ar_tail_k + 1 = 21**).

### 4.2 The same chunk for NGCM

NGCM is not a 2-frame Markov model in decoded space. The NGCM equivalent of
"the frozen model continues its own rollout" is **native state continuation**.
Never re-encode decoded forecasts.

| step `i` | residual input window (older, newer) | `b_next = b[i]` (NGCM forecast, valid `a0+i+1`) | target |
|---|---|---|---|
| 0 | `x[a0-1]`, `x[a0]` | `D(A(E(x[a0], F_{a0})))` | `x[a0+1]` |
| 1 | `x[a0]`, `x[a0+1]` | `D(A(E(x[a0+1], F_{a0+1})))` | `x[a0+2]` |
| 2 | `x[a0+1]`, `x[a0+2]` | `D(A(E(x[a0+2], F_{a0+2})))` | `x[a0+3]` |
| 3 | `x[a0+2]`, `x[a0+3]` | `D(A¹(z_o))`, `z_o = E(x[a0+3], F_o)`, `o = a0+3` | `x[a0+4]` |
| 4 | `x[a0+3]`, `b[3]` | `D(A²(z_o))` | `x[a0+5]` |
| i ≥ 5 | `b[i-2]`, `b[i-1]` | `D(A^{i-2}(z_o))` | `x[a0+i+1]` |

Every `D(·)` and `A(·)` uses the forcing of its own origin. For steps 0–2 that
origin is `a0+i`; for steps 3–23 it is `o = a0+3`, persisted.

A general rule reproduces the GC `frames` list exactly:

```python
def frame(j):                     # j indexes the GC 'frames' list
    return x[a0 - 1 + j] if j <= 4 else b[j - 2]
window_i = (frame(i), frame(i + 1))
```

`b[j-2]` is valid at `a0+j-1` for `j ≥ 5`, matching `x[a0-1+j]` for `j ≤ 4`.

**Window frames in the tail are NGCM's decoded forecasts from the tail
rollout.** They are never the one-step forecasts of steps 0–2, never truth, and
never corrected forecasts.

### 4.3 Baseline cache producer (normative pseudo-code)

```python
def chunk_baseline(bb, store, times, a0, n=24, prefix=4):
    """Return b[0..n-1]; b[i] is NGCM's decoded forecast valid at times[a0+i+1].

    Shape (n, 7, 37, 128, 64) float32; fields in FIELDS order, levels ascending.
    """
    out = []
    for i in range(prefix - 1):                              # i = 0, 1, 2: one-step from truth
        inp, frc = store.inputs_and_forcing(bb.model, times[a0 + i])
        z0 = bb.encode(inp, frc)
        z1 = bb.advance_6h(z0, frc); bb.assert_time(z0, z1, 1)
        out.append(stack_fields(bb.decode(z1, frc)))
    inp, frc = store.inputs_and_forcing(bb.model, times[a0 + prefix - 1])  # origin o = a0+3
    z0 = z = bb.encode(inp, frc)
    for k in range(1, n - prefix + 2):                       # k = 1..21
        z = bb.advance_6h(z, frc)                            # same persisted forcing
        bb.assert_time(z0, z, k)
        out.append(stack_fields(bb.decode(z, frc)))
    assert len(out) == n
    return np.stack(out).astype(np.float32)
```

* `stack_fields` keeps only `FIELDS`, in that order, as `[level, lon, lat]`.
  Drop `sim_time` and any other keys.
* Cache file per chunk: `b` (fp32) plus a manifest record
  `{a0_time, valid_times[24], origin_times[24], sha256}`. The manifest also
  holds the producer identity: checkpoint sha256, store identity, code hashes,
  library versions and this README's version. Atomic writes; refuse to
  overwrite. This mirrors `v24_Ilya/baseline_cache.py`.
* Size: one chunk is 24 × 7 × 37 × 8192 × 4 B ≈ 204 MB. Train is 424 chunks
  (~86 GB) and val is 60 chunks (~12 GB). Put it on DABANIN, never on BIAO.
* Truth is never cached; it is read from the `PreparedStore`.
* The cache is an optimization, not part of the definition. A `live` mode that
  computes `chunk_baseline` on the fly must give identical arrays (test T1).

### 4.4 Anchors, segments, cursor

* **Train anchors.** Every train-split frame index `t` with `t − 4 ≥ 0` and
  `t + 1 ≤ last`. The `−4` (instead of GC's `−1`) is NGCM-specific: the
  one-step origin `a0` needs its 24 h-lagged forcing frame `a0 − 4`. For
  complete data this gives 2015‑01‑02T00 … 2021‑12‑31T12 = **10,223 anchors**.
* **Segments.** Use `build_segments` (v24, unchanged): non-overlapping
  chronological blocks of 96 anchors, with the incomplete trailing block
  dropped. That gives **106 segments, 424 chunks per epoch**. Chunk `c`
  (c = 0…3) of a segment uses anchors `seg[24c : 24c+24]`.
* **Cursor.** Use `advance_cursor` (v24, unchanged, batch 1): offset += 24;
  then the next segment; then the next epoch from segment 0. Order is
  chronological with **no shuffle**. Every epoch visits the same order.
* **Val.** Built the same way on 2022: anchors 2022‑01‑02T00…2022‑12‑31T12 =
  1,455, giving **15 segments** (60 chunks). Use all 15 (v24 uses 16, but
  only 15 exist here).

---

## 5. Residual inputs, outputs, statistics, normalization

### 5.1 Input channels per grid node (fixed order; record its hash)

| block | content | channels |
|---|---|---|
| A | older window frame, 7 FIELDS × 37 levels (field-major, levels ascending) | 259 |
| B | newer window frame, same layout | 259 |
| N | `b_next = b[i]`, same layout | 259 |
| R | TOA incident solar radiation at `t_A`, `t_B`, `t_target` | 3 |
| Y | `year_progress_sin/cos` at `t_A`, `t_B`, `t_target` | 6 |
| H | `day_progress_sin/cos` (local solar time, depends on lon) at `t_A`, `t_B`, `t_target` | 6 |
| S | SST, sea-ice cover **as NGCM used them for this step** (`F_{a0+i}` for steps 0–2, `F_o` for steps ≥ 3) | 2 |
| G | static: surface geopotential, land–sea mask, from the NGCM checkpoint aux dataset (as `KnownFeatures`) | 2 |
| **total** | | **796** |

* R/Y/H mirror GC's forcings: 5 GC forcing variables at the 2 input times plus
  the target time. Compute them with our `third_party/graphcast` helpers
  (`solar_radiation.get_toa_incident_solar_radiation`, `data_utils` progress
  features) on the Gaussian lat/lon. They are deterministic functions of time,
  so they leak nothing.
* Do **not** add sin/cos lat/lon. GraphCast adds grid-node structural features
  itself.
* S uses only forcing NGCM already saw, so there is no future information.
* `t_A`, `t_B`, `t_target` = `times[a0+i-1]`, `times[a0+i]`, `times[a0+i+1]`.

### 5.2 Outputs

* 203 channels = 7 FIELDS × 29 levels (50…1000 hPa), field-major, levels
  ascending.
* Scatter into the 37-level array. **The correction at 1–30 hPa is exactly 0**,
  so `y = b` there. This is stronger than a zero loss weight and matches the
  existing weather-space design.
* Physical correction `ĉ = σΔ[v,p] · head_output` (no location), as in
  `DirectResidualNormalizer`.

### 5.3 Statistics (train years 2015–2021 only; computed on the NGCM grid)

| name | definition | used for |
|---|---|---|
| `mean_by_level μ[v,p]` | mean of ERA5 over all train frames and all grid points (unweighted, like GC stats) | inputs A, B, N |
| `stddev_by_level σ[v,p]` | std, same sample | inputs A, B, N |
| `diffs_stddev_by_level σΔ[v,p]` | std of **6 h** differences `x_{t+1} − x_t`, same sample | residual target/output scale |
| μ, σ of R, Y, H, S, G channels | same sample | inputs |

* Use every train frame (preferred; at 2.8° it is a one-pass I/O job) or a
  deterministic stride of 7. Record which in the manifest.
* `b_next` (block N) uses the **same μ, σ as the ERA5 field it forecasts**.
  It gets no separate statistics.
* **Floors (NGCM extension, same rule as `experimental/weather_space_residual`).**
  For the two cloud fields, apply
  `scale ← max(scale, 1e-3 × median_positive_scale[v])` separately to σ and σΔ.
  Record every floored level. Fail on non-finite values or a field with no
  positive scale. For other fields, assert `σ, σΔ > 0` and do not floor unless
  the loss census (§11, T10) forces a recorded decision *before* training.
* Never use the 24 h-lag statistics from `paper_statistics.py`. Never estimate
  statistics on 2022/2023.

### 5.4 Normalization (exactly `DirectResidualNormalizer`)

```
inputs_norm   = (inputs − μ) / σ                 # inputs: mean AND std (σ = stddev_by_level, NOT σΔ)
target_norm   = r / σΔ                           # residual: no location
c_physical    = σΔ · head_output                 # output: no location
```

The input rule is the one the 2026‑07‑09 tree-mismatch incident broke: the
drifted tree divided inputs by σΔ without subtracting μ. Test T7 pins it.

---

## 6. Model

The GC residual GNN: grid→mesh encoder, then **2 message-passing steps, each
followed by a 2-layer Mamba block** (`temporal_location="mesh_processor_interleaved"`;
see `third_party/graphcast/graphcast/graphcast.py::_run_mesh_gnn_interleaved`),
then mesh→grid decoder, then a **zero-initialized linear head**.

| parameter | value | note |
|---|---|---|
| base `gc.ModelConfig` | read from the GC-small checkpoint used by v24 (weights not used) | as `v24_Ilya.model.build_model_configs` |
| `latent_size` (width) | 512 | GC2 residual width |
| `gnn_msg_steps` | 2 | `residual_msg_steps` |
| `mesh_size` | **4** (2,562 mesh nodes) | 2.8° analogue of v24's mesh 5 at 1°. Grid/mesh ratio 8,192/2,562 ≈ 3.2. Mesh 5 would have more mesh than grid nodes |
| `resolution` | unused; the graph is built from the Gaussian lat/lon passed to `_maybe_init` | as `NativeResidualGraph` |
| temporal block | `attach_temporal(predictor, arch)` from `v24_Ilya/model.py` — call it, do not re-implement | |
| `temporal_d_inner` | 128 | our v22 "inner128" |
| `d_state / d_conv / bc_groups / dt_rank` | 16 / 4 / 1 / auto | |
| `temporal_layers` | 2 | per interleave point |
| `init_scheme / dt_init / dt_min / dt_max / dt_scale / dt_init_floor` | mamba1 / random / 1e‑3 / 0.1 / 1.0 / 1e‑4 | v24 open-loop config |
| `conv_bias / bias / dropout` | True / False / 0.0 | |
| `temporal_stateful` | True | memory = Haiku state |
| `temporal_zero_init_out` | **True** | GC parity. `NativeResidualGraph` uses False — do not copy that |
| head | `hk.Linear(203, w=0, b=0)` | output exactly 0 at init |
| init | **fresh**, `PRNGKey(22)` | no compatible GC weights exist for these channels and this grid |
| precision | fp32 everywhere | 2.8° is small |

**Orientation (error-prone).** NGCM arrays are `[..., lon, lat]`. GraphCast
grid nodes are flattened **lat-major**. Build `x[lon, lat, ch]`, then
`transpose → [lat, lon, ch]` and `reshape(64*128, 1, ch)`. Invert this exactly
on the output: `reshape(64, 128, 203) → transpose → [lon, lat, 203] →
reshape to [7, 29, lon, lat]`. Copy the pattern in
`NativeResidualGraph.__call__`. Keep the receiver-connectivity assertion from
its constructor. With mesh 4 and `radius_query_fraction_edge_length=0.6`,
every mesh and grid receiver must have degree > 0.

---

## 7. Loss

GC's `losses.weighted_mse_per_level` applied to the normalized residual, with
two NGCM changes: Gaussian quadrature latitude weights and the NGCM field set.

```
ℓ_i = Σ_v α_v · mean_{p∈P, lon, lat} [ w_lat · (p / p̄) · ((ĉ_i − r_i) / σΔ_{v,p})² ]
L_chunk = (1/24) Σ_{i=0}^{23} ℓ_i                    # loss_mode = all_steps
```

| symbol | value |
|---|---|
| `r_i` | `x[a0+i+1] − stop_gradient(b[i])`, on `P` only |
| `P` | 29 levels, 50…1000 hPa |
| `p̄` | mean of the **29** levels = 15470/29 = 533.45 hPa. GC uses the mean over the levels present in the target; do not use the 37-level mean |
| `w_lat` | `64 × loss.gaussian_weights(lat)`, i.e. Gauss–Legendre weights **normalized to mean 1** (GC normalizes cos-lat to mean 1). Broadcast over lon |
| `α_v` | 1.0 for T, Z, u, v, q; **0.05** for each cloud field (NGCM extension; GC has no cloud fields) |
| reduction over variables | **sum** (GC `sum_per_variable_losses`), not mean |

* GC's `normalized_latitude_weights` (and the uniform-grid fallback in
  `training/core/model.py`) raises on Gaussian latitudes. Never pass NGCM data
  through it, and never replace it with cos(lat) or uniform weights.
* `neuralgcm_residual/loss.py::WeatherLoss` has the same per-field reduction
  as GC (its `p/Σp` and `w/Σw` equal GC's mean-1 weights after averaging). But
  it **divides by the number of fields and has no α**. Reuse its per-field
  scores only through `Σ_v α_v · field_scores[v]`, restricted to `P`.
* No spectral terms, no bias term, no lead down-weighting, no adaptive
  weighting. GC has none of these.
* GC's surface terms (2t, 10 m winds, MSLP, precipitation) do not exist here.
  This is an intended difference.

---

## 8. Training loop

```python
state = zeros_like(memory)                              # Haiku state, fp32
for step in range(1, max_steps + 1):
    seg, off = cursor.segment_index, cursor.segment_offset
    if off == 0:                                        # new segment → zero memory
        state = zeros_like(state)
    a0 = segments[seg][off]
    batch = load_chunk(a0)          # truth x[a0-1 .. a0+24], cached b[0..23], features of §5
    (loss, new_state), grads = jax.value_and_grad(chunk_loss, has_aux=True)(params, state, batch)
    state = jax.lax.stop_gradient(new_state)            # carry, truncated BPTT at chunk boundary
    params, opt_state = optimizer_update(grads, opt_state, params)
    cursor = advance_cursor(cursor, cfg, len(segments))

def chunk_loss(params, state, batch):                   # full BPTT through 24 steps
    def step(state, inp_i):
        out, state = model.apply(params, state, key, inp_i.features)   # head output, normalized
        return state, weighted_mse(out, inp_i.target_norm)             # §7, ℓ_i
    state, losses = jax.lax.scan(jax.checkpoint(step), state, batch.steps)
    return losses.mean(), state
```

* **Memory semantics (identical to v24 `cached_runner`).** Zero at
  `segment_offset == 0`. Carried across the 4 chunks of a segment as data
  (stop-gradient). Never reset inside a chunk. The carried state at a chunk
  boundary has just seen a 126 h tail; the next chunk restarts on truth. This is
  intended and identical to GC.
* **BPTT.** `lax.scan` + `jax.checkpoint` is fine at 2.8°. v24 uses a host tape
  with a reverse `jax.vjp` (`explicit_reverse_vjp`); both must give the same
  gradient (test T8). No gradient ever reaches NGCM. NGCM is not even traced
  during training.
* **Optimizer.** Use `v24_Ilya.training.endpoint_step.build_optimizer`:

| setting | value |
|---|---|
| optimizer | AdamW, β1 0.9, β2 0.98, weight decay 1e‑4 on all parameters |
| LR schedule | linear warmup 0 → 1e‑4 over 200 steps, cosine to 1e‑5 at `max_steps` |
| grad clip | global norm 1.0 |
| max_steps | 10,000 (≈ 23.6 epochs of 424 chunks) |
| batch | 1 lane (`per_device_batch_size=1`, single device) |
| seed | 22 |

* **Checkpoints.** Every 500 steps, saving params, opt_state, memory state,
  cursor, step and RNG, for exact resume (`--resume`).
* **Validation (monitoring only, every 1,000 steps).** All 15 val segments,
  memory zeroed at each segment start. Report `ℓ_i` per step class: steps 0–2,
  and tail leads 6–126 h. Compare against the zero-head (NGCM) value. **Val is
  not used to select anything** (see §9 for why).
* **Primary model = fixed-window SWA over steps 2,000–8,000.** Average the 13
  saved checkpoints, uniform weights. Fix this window before training. GC
  experience: convergence ≈ 2k steps, and 7-year training degraded after ≈ 8k.
  Never use per-lead or per-K tuned windows (the old "EMA" checkpoints were
  per-K-window SWA). Also report the final step.

---

## 9. Evaluation and controls

### 9.1 Rollout (cold start, primary; identical to the GC eval with `W=0`)

```python
inp, frc = store.inputs_and_forcing(bb.model, o)
z0 = z = bb.encode(inp, frc); state = zeros; prev, cur = x[o-1], x[o]
for k in range(1, 41):                                 # 40 steps = 240 h
    z = bb.advance_6h(z, frc); bb.assert_time(z0, z, k)
    b_k = decode(z, frc)                               # native NGCM forecast = reference
    c_k, state = model(features(prev, cur, b_k, times, frc, static), state)
    y_k = b_k + scatter29(σΔ * c_k)                    # 1–30 hPa unchanged
    prev, cur = cur, b_k                               # feed back NGCM's forecast, never y_k
```

Eval step `k` corresponds exactly to training step `i = k + 2` of a chunk with
`a0 = o − 3`.

* **Warm-3 (diagnostic).** Before the loop, run training steps 0–2 of
  `a0 = o − 3` (one-step NGCM forecasts from truth at `o−3`, `o−2`, `o−1`; no
  loss). This reproduces the memory state that training has at the start of
  every chunk-0 tail. Report it separately, never mixed with cold numbers.
  It needs frames back to `o−7` (window `x[o−4]` plus its 24 h forcing lag).
  Skip origins where that crosses the split year and report how many were
  skipped.

### 9.2 Start sets

| set | origins | purpose |
|---|---|---|
| **E1 GC-parity** | `results/anchor_manifests_r1/paper_matched32.json` (in the experiments repo): 32 starts in **2022**. `anchor_indices` are GC *final-input* indices (`valid_final_input_indices`) on the 2022 eval dataset, so origin = `eval_ds.time[idx]`. Assert the 32 datetimes against the GC eval logs; do not hand-convert. Includes 06/18 UTC starts | side-by-side with the paper's GC numbers |
| **E2 held-out test** | every 00 and 12 UTC origin in **2023** with 40 steps and the 24 h lag inside 2023 (~700) | clean test, matches WB2 init hours |

Because 2022 is only monitored (no selection, fixed SWA window), reporting E1
on 2022 is legitimate. Never use the selection-free argument if anything was
tuned on 2022.

### 9.3 Metrics (all latitude-weighted with Gaussian weights; never uniform)

* RMSE per field × level × lead. Headline: Z500, T850, U850, V850, Q700
  (`evaluate.unit_metrics`), plus cloud fields.
* % MSE change vs native NGCM: `100·(1 − MSE_y/MSE_b)`.
* `J_k` = the training loss `ℓ` at lead `k` (GC "paper-weighted MSE" analogue),
  for corrected and NGCM; % change at 120 h and 240 h and averaged over leads.
* ACC against one fixed climatology (train years, on the NGCM grid), the same
  for both models.
* Power spectra (Z500, T850, U850) of `x`, `b`, `y` at 24/120/240 h, to make
  smoothing visible.
* Paired block bootstrap 95% CI over origins (`evaluate.block_bootstrap`).
* Absolute RMSE is on the 2.8° grid and is **not** comparable to our 1° GC
  numbers. Compare relative % only.

### 9.4 Controls (same seed, steps, SWA window)

| id | arm | answers |
|---|---|---|
| C0 | native NGCM | reference |
| C1 | lead-dependent bias table: `B[k, v, p, lon, lat] = mean(x − b)` over all 00/12 UTC origins of 2015–2021, k = 1…40 (no learning) | is the gain just bias removal? |
| C2 | Markov: memory zeroed before every step (v24 `temporal_state_policy="reset_every_anchor"`) | what does memory add? |
| C3 | C2 + lead-time input (`lead_hours/240`; 6 h for steps 0–3, `(i−2)·6` h for tail steps) | is memory only a lead-time clock? |
| C4 | main model without block N (`b_next`) | GC-literal inputs vs NGCM-adapted |

---

## 10. Stages, package layout, code reuse map

```
experimental/residualmamba_NGCM/
  README.md                       # this file
  configs/default.json            # every constant in this README; schema-versioned
  residualmamba_ngcm/
    config.py                     # frozen dataclasses; reject unknown/changed fields
    timeline.py                   # anchors (t-4 rule), segments, cursor  (reuse v24 data.py)
    ngcm.py                       # load-only wrapper around neuralgcm_residual.backbone/data
    cache.py                      # chunk_baseline (§4.3), manifest, verify
    stats.py                      # μ, σ, σΔ (6 h), floors, manifest
    features.py                   # 796-channel assembly (§5.1), normalization (§5.4)
    model.py                      # GC residual on Gaussian grid + attach_temporal + zero head
    loss.py                       # §7
    train.py                      # §8
    evaluate.py                   # §9 (cold, warm-3, controls, metrics)
    swa.py
  tests/                          # §11
```

Stages: `preflight → stats → cache → verify-cache → census → train → swa →
evaluate → report`. Each stage writes an immutable manifest that includes the
upstream manifests' hashes.

| reuse unchanged | from |
|---|---|
| `FrozenBackbone.load / encode / advance_6h / decode / assert_time / manifest` | `src/models/neuralgcm_residual/backbone.py` |
| `PreparedStore`, `inputs_and_forcing`, `split_for`, `require_complete_experiment_data` | `src/models/neuralgcm_residual/data.py` |
| `gaussian_weights` | `src/models/neuralgcm_residual/loss.py` |
| `build_segments`, `advance_cursor` | `src/models/mamba/v24_Ilya/training/data.py` |
| `attach_temporal`, zero-head pattern of `GCResidualWithZeroHead` | `src/models/mamba/v24_Ilya/model.py` |
| `build_optimizer` | `src/models/mamba/v24_Ilya/training/endpoint_step.py` |
| orientation + connectivity checks of `NativeResidualGraph` | `src/models/neuralgcm_residual/model.py` (pattern only; its fixed Architecture validation forbids width 512) |

**Do not reuse:** `download_checkpoint`, `prepare_era5`, `paper_loss.py`,
`paper_statistics.py`, `trajectory_training.py`, `NativeAdapter.apply_increment`,
or `WeatherLoss.__call__` as the training loss.

---

## 11. Tests that must pass before any production run

| id | test | pass criterion |
|---|---|---|
| T1 | cache == live `chunk_baseline` for 3 random train chunks and 1 val chunk | max abs diff ≤ 1e‑5 relative to field std (bitwise on the same GPU type) |
| T2 | cache tail `b[3..23]` == `evaluate.forecast_origin` native baseline from `o = a0+3` for 21 steps | same tolerance. Proves we train against the baseline we evaluate against |
| T3 | time bookkeeping | `assert_time` on every advance. `valid_times[i] == times[a0+i+1]`. Window times match §4.2 for all 24 steps |
| T4 | no future leakage | perturbing any truth `x[t]` with `t ≥ a0+4` changes only targets, never any input of any step. (Truth `x[a0−4..a0+3]` may legitimately reach inputs: windows of steps 0–4, forcing lags, and the whole tail through `E(x[a0+3])`) |
| T5 | indexing ≡ GC | with a toy *Markov* baseline (`pred = f(window)`), our window/target sequence equals v24 `iter_trajectory` for the same anchors |
| T6 | zero head | untrained model gives `y == b` bitwise, and the loss equals the NGCM baseline loss |
| T7 | normalization | inputs use `(x−μ)/σ` with `σ = stddev_by_level`. Target uses `r/σΔ`, no location. Output uses `σΔ·out`. Unit test with hand-built stats |
| T8 | BPTT | `scan`+`checkpoint` gradient == step-by-step `vjp` gradient (rtol 1e‑5). The gradient w.r.t. the carried input state at a chunk boundary is 0 |
| T9 | memory reset | state is exactly 0 at every `segment_offset == 0`, and non-zero after step 0 otherwise |
| T10 | loss census | on the val cache with the zero head: share of `L` per (field, level, step class). Any single (v, p) above 10% → record a decision (floor or not) **before** training |
| T11 | orientation | a synthetic field `= sin(lat)` lands on GraphCast grid nodes with matching latitude, and the round trip input→output layout is the identity |
| T12 | Gaussian weights | `gaussian_weights` node check passes. `w_lat.mean() == 1`. `p̄ == 533.45` |
| T13 | output scatter | 203 channels map to the 29 levels ≥ 50 hPa. Levels 1–30 hPa of `y` equal `b` bitwise |
| T14 | statistics provenance | the stats manifest lists only 2015–2021 frames. Floored levels are recorded |

---

## 12. Consistency check against our GraphCast residual-Mamba

Reference: v24 cached open loop (`scripts/training/train_v24_cached_open_loop.py`,
`v24_Ilya/baseline_cache.py`, `training/{frame_data,data,cached_stepwise,cached_runner,endpoint_step}.py`,
`v24_Ilya/model.py`). It was checked line by line on the `minimalistic_code`
branch and is logic-equivalent to v22clean residual Mamba open loop:

* same residual target;
* same `feedback_mode=baseline` semantics;
* `DirectResidualNormalizer` AST-identical;
* same GC weighted MSE;
* same 4 + 20 prefix/tail;
* same memory carry with reset at segment start;
* same 1/24 step weights.

| aspect | GC residual-Mamba (open loop) | residualmamba_NGCM | status |
|---|---|---|---|
| feedback mode | `baseline`: the frozen forecast is fed back, the correction never is | same | **identical** |
| how the baseline continues | 2-frame window of its own decoded predictions (= GC's native rollout) | native state continuation `A^k(z_o)` (= NGCM's native rollout); no re-encode | **equivalent** (each is its model's native rollout) |
| residual target | `truth − stop_grad(baseline)` | same, decoded space, 29 levels | identical |
| residual input window | same 2 frames as the baseline (truth in prefix, baseline predictions in tail) | same 2-frame rule (§4.2) on decoded NGCM frames | identical rule |
| baseline forecast as input | implicit (function of the inputs) | explicit block N (`b_next`) | **NGCM-adapted** (§2); C4 is the GC-literal control |
| forcings / static | TISR + year/day progress at 2 input times + target; orography, LSM | same 15 time channels; orography, LSM from NGCM aux; plus SST/ice that NGCM used | NGCM-adapted (SST/ice are NGCM's own forcing) |
| normalization | `DirectResidualNormalizer`: inputs `(x−μ)/σ`, residual `/σΔ`, no location | same; stats recomputed on the NGCM grid, 37 levels, train years; cloud floors | identical rule, NGCM stats |
| latitude weights | cos-lat cell area, mean 1 | Gauss–Legendre weights, mean 1 | NGCM-adapted (required) |
| level weights | `p/p̄` over the 13 GC levels | `p/p̄` over 29 levels | NGCM-adapted (more boundary-layer levels; see §14) |
| variable weights | atmos 1; 2t 1; 10u/10v/msl/tp 0.1 | T/Z/u/v/q 1; cloud ice/liquid 0.05; no surface fields | NGCM-adapted |
| sequence | segment 96, bptt 24, prefix 4, `ar_tail_k` 20 (GC K = 21), loss on all 24 steps at 1/24 | same | identical |
| anchors | first anchor at final-input index ≥ 1 | ≥ 4 (24 h forcing lag) | NGCM-adapted (3-step shift) |
| memory | zero at segment start; carried across chunks, stop-grad | same | identical |
| data order | chronological segments, no shuffle, batch 1 | same | identical |
| architecture | GC2 width 512, 2 msg steps, interleaved 2-layer Mamba, zero head, mesh 5 at 1° | same; mesh 4 at 2.8°; d_inner 128 | identical, mesh scaled to grid |
| initialization | `baseline_overlay` from GC-small | fresh, `PRNGKey(22)` | **deliberate** (no compatible weights) |
| optimizer | AdamW 1e‑4, warmup 200, cosine → 1e‑5, wd 1e‑4, clip 1, β2 0.98, 10k | same | identical |
| precision | bf16 compute (v24 config), fp32 tape | fp32 | deliberate (cheap at 2.8°) |
| training data | 15 yr ERA5 at 1° | 2015–2021 (7 yr) on the NGCM grid | deliberate (NGCM data contract); 7 yr overfit risk → fixed SWA 2k–8k |
| evaluation | open-loop 40 steps, `W=0` cold, matched 32 starts (2022) | same (E1) + full 2023 test (E2) | identical protocol, plus a clean test set |

**Verdict.** The open-loop logic, sequence layout, normalization rule, loss
family, memory semantics and optimizer are carried over unchanged. The
differences are (a) required by NGCM (Gaussian grid weights, fields/levels,
native-state continuation, forcing lag) or (b) deliberate and covered by a
control (`b_next` → C4; fresh init is unavoidable).

---

## 13. Common mistakes (checklist)

1. Re-encoding decoded NGCM forecasts in the tail (`E(D(z))`). That is a
   different, worse baseline than the one we evaluate against. T2 catches it.
2. Feeding `y` (corrected) back into the window or into NGCM. That is closed
   loop.
3. Filling any tail input, level or field from future ERA5. T4 catches it.
4. Using the one-step forecasts of steps 0–2 as tail window frames.
5. Time-varying SST/ice in the cache but persisted forcing in evaluation, or
   the reverse.
6. Forgetting the lon/lat transpose. T11 catches it.
7. cos(lat), uniform, or sum-to-1 latitude weights. Weights must be Gaussian
   with mean 1.
8. `p̄` over 37 levels instead of 29.
9. Averaging over fields (`WeatherLoss.__call__`) instead of `Σ α_v`.
10. Inputs divided by σΔ, or without mean subtraction (the drifted-tree bug).
    Residual normalized with a location.
11. 24 h-lag statistics, or statistics touching 2022/2023.
12. Resetting memory every chunk, never resetting it, or letting gradients
    cross chunk boundaries.
13. Shuffling chunks or segments.
14. Applying corrections at 1–30 hPa.
15. Copying `NativeResidualGraph`'s `temporal_zero_init_out=False` or its
    width/d_inner validation.
16. Selecting checkpoints or SWA windows on 2022 and then reporting E1 on 2022.
17. Unweighted metrics, or mixing cold and warm-3 numbers in one table.
18. Comparing absolute 2.8° RMSE with our 1° GC RMSE.

---

## 14. Open decisions

1. **Loss levels.** The default is 29 levels with `p/p̄`. It weights the
   boundary layer more than GC's 13 levels. A strict GC-weighting alternative
   is to score only the 13 GC levels and output only those. Decide before the
   first run; do not switch after seeing results.
2. **Specific-humidity floor.** Decided by T10 before training.
3. **NGCM in-sample years.** The public NGCM checkpoints were trained on
   earlier ERA5 years. Check the NeuralGCM paper for the exact range. If it
   overlaps 2015–2021, NGCM errors in our training years are in-sample, and
   possibly smaller than in 2022–2023. Before training, report NGCM's baseline
   `J_k` per year (2015…2023): use C1's rollouts for 2015–2021 and the same
   00/12 UTC rollouts for 2022–2023. If in-sample years are
   clearly better, discuss restricting training to out-of-sample years.
4. **Tail length.** `ar_tail_k = 20` (GC K = 21) for parity. The eval goes to
   240 h, so leads 132–240 h are extrapolated in time, exactly as in the GC
   protocol.
5. **Disk.** About 100 GB of fp32 cache on DABANIN. Confirm the quota first.
