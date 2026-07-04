# v22cl — Closed-Loop Residual Mamba GraphCast (SWA analysis)

Author: Lianghong Mo (lm8598@princeton.edu)
Snapshot: 2026-07-03
Base: extends the `AR-Training-Lianghong` branch's `v22` (open-loop) results with a
**closed-loop** variant (`v22cl`), a **Stochastic-Weight-Averaging (SWA)** step across
per-K training checkpoints, and full-eval + per-city diagnostics.

---

## 1. What this folder contains

```
closed_loop_v22cl/
├── README.md                    (this file)
├── code/
│   ├── eval/
│   │   ├── eval_v22_clean.py           — global lat-weighted MSE/RMSE eval (32-anchor avg)
│   │   ├── eval_city_trace.py          — per-anchor per-city rollout, dumps JSON
│   │   └── diag_ssm_state_rollout.py   — Mamba SSM h_t trajectory dumper (norm + Δ)
│   ├── analysis/
│   │   ├── analyze_SWA_full_eval.py    — build per-K improvement curves + channel heatmaps
│   │   └── plot_v22cl_SWA_city_traces.py — 4-line city-trace comparison plots
│   └── slurm/
│       ├── v22cl_r1_allK_ema_eval.slurm             — 11 K × 2 modes global eval
│       ├── v22cl_SWA_city_v22cleantree.slurm        — v22cl SWA cold_full city traces
│       ├── v22_paper_city_v22cleantree.slurm        — v22 paper K=22 cold_bp city traces
│       └── v22_paper_city_allK_v22cleantree.slurm   — v22 paper K=2..20 cold_bp city traces
└── results/
    ├── SWA_full_eval/
    │   ├── eval_jsons/                 — 22 JSONs (K=2..22 × {cold_bp, cold_full})
    │   └── plots/                      — per-K MSE-improvement curves + channels-improve heatmaps
    │                                     + drift_vs_step / alpha-sweep / SWA-vs-single-ckpt bars
    ├── city_traces/
    │   ├── v22cl_SWA_JSONs/            — 16 JSONs (K=14/18/20/22 × 4 anchors)
    │   ├── v22_paper_JSONs/            — 44 JSONs (v22 paper K=2..22 × 4 anchors, cold_bp)
    │   └── plots/                      — 20 per-city comparison figures (4 seasons × 10 cities)
    ├── extreme_records/
    │   └── plots/K{2,4,6,8,10,12,14,16,18,20,22}/extreme_records_K40.png
    │                                   — heat/cold/wind record-exceedance eval per K
    └── ssm_state_analysis/             — h_t evolution plot + CSV
```

---

## 2. What v22cl is and why we did it

`v22` (the earlier paper snapshot) trains the residual Mamba head **open-loop**:
- Feedback: `cur ← baseline(cur)` — the baseline GC prediction is fed forward
- Residual is computed and added at the output, but the residual is NOT part of the
  input distribution the model sees during training

`v22cl` trains the **same architecture** but **closed-loop**:
- Feedback: `cur ← baseline(cur) + residual(cur)` — the residual is fed back into
  the input stream every step
- The Mamba SSM sees the actual deployment-time input distribution during training

**Motivation**: at inference time we always want the closed-loop trajectory
(`cold_full`) — that's what a deployed forecast does. Training under open-loop
creates a train/deploy distribution mismatch, and empirically the SSM hidden
state `h_t` drifts / spikes late in the rollout for v22 paper when we force
`cold_full` at eval (see `results/ssm_state_analysis/h_t_evolution_v22cl_vs_v22paper.png`).

`v22cl` closes that gap: its `h_t` reaches a stationary regime by rollout step
~10, and stays stable through all 40 forecast steps (240h).

---

## 3. Why we use SWA

Per-K training runs of `v22cl` show a **characteristic drift**: after the model
reaches its best MSE-improvement point (roughly step 6000–10000 depending on K),
the residual head continues to grow in output amplitude, and the paper-weighted
MSE improvement at long leads **starts to decay**. Late checkpoints overfit an
amplitude that no longer helps.

### 3.1 Data supporting the drift claim

**Per-K single-step evaluation** shows peak-then-decay clearly:

```csv
K,step6k_imp%,best_step,best_step_imp%,step20k_imp%,SWA_imp%,SWA-best_gain%
14,+3.24, 6000,+3.24, -2.80, +8.36, +5.11
18,+6.70,10000,+9.27, -6.11,+13.14, +3.88
20,+12.26,8000,+12.99,+4.50,+19.21,+6.22
22,+16.17,6000,+16.17,+7.57,+18.96,+2.79
```
(paper-weighted MSE improvement% @ lead 240h vs GC baseline; cold_full mode; full CSV: [`results/SWA_full_eval/plots/swa_vs_single_ckpts_table.csv`](results/SWA_full_eval/plots/swa_vs_single_ckpts_table.csv))

**Observations**:
- Best single-step ckpt lives at step 6000–10000 for every K ≥ 14
- By step 20k, improvement has decayed **8–14 percentage points** vs the peak
  (K=18: +9.27 → -6.11 = **−15.4 pp collapse**; K=22: +16.17 → +7.57 = −8.6 pp)
- **SWA (uniform average over steps ~6k–16k) beats every single step at every K ≥ 14**,
  with the largest gain at K=20 (+6.22 pp over the best individual step)

See [`results/SWA_full_eval/plots/drift_vs_step_lead240.png`](results/SWA_full_eval/plots/drift_vs_step_lead240.png)
for the per-step trajectory (colored lines) with SWA horizontal reference (dashed).
See [`results/SWA_full_eval/plots/swa_vs_single_ckpts_bar.png`](results/SWA_full_eval/plots/swa_vs_single_ckpts_bar.png)
for a bar chart comparing step 6k vs best-step vs step 20k vs SWA per K.

### 3.2 Is the drift caused by output amplitude?

Partial evidence: **α-sweep** — scale the residual by α at inference:
`full_pred = baseline + α · residual`. Amplitude reduction should undo overshoot.

For K=22 (see [`results/SWA_full_eval/plots/K22_alpha_sweep.png`](results/SWA_full_eval/plots/K22_alpha_sweep.png)):

| ckpt | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|
| step 6k  (near-optimum) | +9.1  | +14.2 | **+16.4** | +15.7 |
| step 20k (overtrained)  | +6.4  | +9.0  | **+9.1**  | +7.6  |
| **K=22 SWA (imp = +18.96%)** for reference |

Interpretation:
- Step 6k's optimal α is ~0.75 → the trained residual is already close to
  right amplitude (slight over-scaling)
- Step 20k benefits from α reduction (α=0.75 gives +9.1% vs α=1.0 gives +7.6%)
  → confirming **the residual output amplitude has grown too large by step 20k**
- BUT even step 20k with best α doesn't reach SWA — so amplitude is not the
  entire story. Direction of the residual also drifts (SWA averages out those
  directional oscillations too)

**SWA is the practical fix** because it doesn't require identifying a per-K
optimal α at inference — the average handles amplitude and direction together.

### 3.3 How SWA is built

The per-K EMA/SWA `.pkl` files live at
`/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22cl_r1_EMA/K{K}_ema.pkl`.
They were built by uniformly averaging `residual_params` across the training
window bracketing each K's optimum (typically 6 checkpoints, steps ~6000–16000
depending on K).

---

## 4. Headline results

### 4.1 Paper-weighted MSE improvement over baseline GC — cold_full mode, all K, 32-anchor lat-weighted

At **lead 240h** (paper's headline metric, 10-day forecast):

| K  | SWA improve% | best-single-ckpt improve% | step 20k improve% |
|----|---:|---:|---:|
| 10 |  +3.9 |  +3.2 |  −8.2 |
| 14 |  +8.4 |  +3.2 |  −2.8 |
| 18 | +13.1 |  +9.3 |  −6.1 |
| 20 | +19.2 | +13.0 |  +4.5 |
| 22 | **+19.0** | +16.2 |  +7.6 |

SWA improves on the best single-step ckpt at every K ≥ 14, and dramatically
improves on step 20k (overtrained ckpt). See
`results/SWA_full_eval/plots/cold_full_vs_baseline.png` for full per-lead + K-scan
overlays and Section 3.1 for the drift analysis.

### 4.2 Channels improving vs GC baseline (out of 82 non-precip channels)

At lead 240h, K=22 SWA cold_full: **78/82 channels improve** (95%). Only 4
channels (u_component_of_wind at level 200-500 hPa) show no improvement.
See `results/SWA_full_eval/plots/channels_improve_cold_full.png`.

### 4.3 Per-variable improvement — surface variables at 240h (K=22 SWA cold_full)

| variable | MSE improve% |
|---|---:|
| 2m_temperature | **+17.0** |
| total_precipitation_6hr | **+23.4** |
| 10m_u_component_of_wind | +26.4 |
| mean_sea_level_pressure | +17.8 |
| geopotential | +13.6 |

Surface variables are net winners globally — not just upper-air.

### 4.4 Per-city 10-day forecasts — 40 city × anchor combinations

Comparing v22cl K=22 SWA cold_full vs v22 paper K=22 cold_bp on the same 4
anchors (Jan/Apr/Jul/Oct 15, 2022), 10 cities each:

| model | mean |Δ| @ 240h (10 cities × 4 anchors) | max |Δ| | # cases |Δ|>10K |
|---|---:|---:|---:|
| v22 paper K=22 cold_bp | 2.39 K | 9.06 K | 0/40 |
| v22cl K=22 SWA cold_full | 2.13 K | 6.05 K | 0/40 |

Both are within normal 10-day forecast error. See per-city 4-panel figures in
`results/city_traces/plots/city_{City}_K22_comparison.png` — each shows Jan/Apr/Jul/Oct
2m_T + precip traces for baseline, v22 paper open-loop, and v22cl SWA cold_full.

### 4.5 Extreme-event record-exceedance eval

We also test whether the SWA closed-loop residual **improves forecast skill on
extreme events** — specifically pointwise heat, cold, and wind extremes that
exceed the local 2015–2021 daily record. For each K, we sample ~118 anchors
(every 3 days in 2022), run K=40 (240h) cold_full rollouts, and bin the
forecast error against the record-exceedance magnitude (in K for
temperature, m/s for wind).

Setup:
- Anchor set: ~118 anchors × 40 leads × global grid (181 × 360)
- Reference climatology: `records_climatology_2015_2021.nc` (per-DOY / per-cell
  min/max of 2m_T and wind speed 2015–2021)
- Comparison lines: GC baseline self-rollout (gray) vs v22cl SWA closed-loop (blue)
- Rows = forecast lead: 24h / 48h / 72h / 120h / 168h / 240h
- Columns = event type × metric: {Heat RMSE, Cold RMSE, Wind RMSE, Heat bias,
  Cold bias, Wind bias}

Result — **v22cl SWA REGRESSES on extremes at almost every lead × variable**
beyond 48h. This is the opposite of the global paper-weighted MSE story
and worth being honest about.

Aggregate RMSE across all exceedance bins @ K=22 SWA cold_full, % change
vs GC baseline (negative = v22cl better):

| lead | Heat  | Cold  | Wind  |
|---:|---:|---:|---:|
| 24h  | **−1.6%** ✓ | **−0.8%** ✓ | **−2.6%** ✓ |
| 48h  |  +0.0%    | −0.4% ✓ | **−1.9%** ✓ |
| 72h  |  +1.5% ✗ |  +1.2% ✗ |  +3.3% ✗ |
| 120h |  +0.0%   |  +2.5% ✗ | **+15.5%** ✗ |
| 168h |  +1.4% ✗ | −0.2%  | **+19.6%** ✗ |
| 240h |  +0.3%   | **−9.7%** ✓ | **+19.9%** ✗ |

Summary:
- **Only short leads (≤48h) show a broad win.**
- Beyond 48h, most cells are neutral-to-slightly-worse, with the exception of
  wind, which regresses by 15–20% between 120h and 240h.
- **The single clear long-lead win is Cold @240h (−9.7%).**
- Heat and Cold are essentially tied at 168h/240h — no meaningful improvement.

Contrast with the global paper-weighted MSE which reports 10m_u @240h at
**+26.4%** (v22cl better). The two numbers don't contradict each other:
the global metric averages over ~64,800 grid points × 118 anchors × 40
leads, most of which are NOT extremes; the extreme-records analysis
restricts to the ~1–3% of grid points where truth exceeds the local
2015-2021 record.

**Interpretation**: SWA optimises lat-weighted MSE which downweights
extreme-tail errors. The residual head learns to pull predictions toward
"typical" values — great for typical grid points, harmful for real
extremes. Extreme-event skill is NOT covered by the SWA optimisation
objective. Fixing this requires a tail-calibrated head (see v25/v26 in the
parent repo) or a loss that penalises variance loss, not just MSE.

Per-K figures live at `results/extreme_records/plots/K{K}/extreme_records_K40.png`.
Same axis convention as `results/2026-05-23-v22/plots/K{K}/extreme_records_K40.png`
in the v22 open-loop branch.

### 4.6 SSM hidden-state stability

At inference, `v22cl` cold_full: `h_t` stabilizes by step ~10, `Δh_t → constant
small` for the rest of the rollout. `v22 paper` (open-loop trained) forced into
`cold_full`: h_t is unstable, big Δh_t spikes at ~step 30 — evidence of the
train/deploy distribution mismatch. See
`results/ssm_state_analysis/h_t_evolution_v22cl_vs_v22paper.png`.

---

## 5. Important reproducibility caveat: source tree matters

The residual Mamba forward pass is **not** identical between two src trees that
coexist in the workspace:

- `/home/lm8598/Weather_Global_experiments/`  (v23+ development tree — DRIFTED)
- `/home/lm8598/Weather_Global_experiments_v22clean/` (v22 training tree — CORRECT)

`v22` and `v22cl` checkpoints were trained in the v22clean tree; **they must be
eval'd there too**. Loading the same ckpt via the newer tree causes catastrophic
divergence in closed-loop rollout — same anchor gives global RMSE 14.86K vs
2.80K @ 240h. Root cause: `graphcast.py` processor loop refactor + Mamba module
argument rename (`d_inner` → `hidden_size`) in the newer tree; Haiku params
load without error because layer names match, but the forward-pass logic
diverges.

All eval scripts and slurm launchers in `code/` cd into `v22clean` and use its
`sys.path`. Do not run these scripts against the newer tree.

---

## 6. Comparison against v22 paper

**v22 paper (open-loop, cold_bp eval — matched to training)** gives its
"headline" improvement numbers. Those numbers were **NOT wrong**; they were
computed via `eval_v22_clean.py` in the v22clean tree.

`v22cl SWA cold_full` improves on v22 paper's numbers at every K, especially at
long leads (240h) — the closed-loop training + SWA combination extends the
regime where the residual head keeps helping.

The two are not directly interchangeable — v22 paper is a stable open-loop
model that expects `cur ← bp` feedback, v22cl SWA is a closed-loop model that
expects `cur ← bp + rp` feedback. Cross-evaluation (v22 paper in cold_full or
v22cl in cold_bp) does not correspond to trained behavior and should not be
used as the headline number.

---

## 7. Regenerating the plots

```bash
cd /home/lm8598/Weather_Global_experiments_v22clean

# Global per-K analysis (produces cold_full_vs_baseline.png, channels heatmaps, per-K subplots)
python ../scripts/analyze_models/analyze_SWA_full_eval.py

# Per-city 4-line comparison
python ../scripts/analyze_models/plot_v22cl_SWA_city_traces.py
```

Both plotting scripts read the JSONs in `results/SWA_full_eval/` and
`results/city_traces/` respectively.
