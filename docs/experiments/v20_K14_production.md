# v20 — Long-AR Mamba residual on GraphCast-small (K=14)

Run directory: `/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v20/`
Best checkpoint: `K14_from_v19K1_23k/v13_residual_step26000.pkl`
Date: 2026-05-20

---

## 1. What v20 actually is

Same architecture as v22 (see `results/2026-05-23-v22/README.md`). v20 differs from v22 only in **autoregressive training horizon** and matching BPTT window — the Mamba block, GraphCast backbone, and all other knobs are identical.

| Layer | Status | Notes |
|---|---|---|
| Frozen GraphCast-small backbone | NOT trained | DeepMind pretrained, 1°, mesh-5, 16 msg_steps |
| Residual head (encoder + 2-step processor + decoder + zero-head) | TRAINED | Warm start from `v18_v2/K1_from_v15v2_20k/v13_residual_step23000.pkl` |
| 4× Mamba S6 blocks (2 insertions × 2 layers, interleaved in residual processor) | TRAINED | Full S6, same as v22 |
| 83×83 per-channel output linear (`temporal_residual_head`) | TRAINED | 6,972 params |

---

## 2. Architecture details

**Identical to v22.** See `results/2026-05-23-v22/README.md` §2 for the full breakdown. Summary:

- Frozen GraphCast-small: 1°, mesh-5, width=512, 16 frozen msg-passing steps
- Residual head: width=512, gnn_msg_steps=2, hidden_layers=1, `v9_corrected_GCResidualWithZeroHead`
- Temporal Mamba (full S6, source `/home/lm8598/Weather_Global_experiments/src/models/mamba/modules/temporal_mesh_mamba_Ilya.py`, snapshot at `/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23/`):
  - hidden_size=128, d_inner=128 (expand=1), d_state=16, d_conv=4, dt_rank=32 (auto), layers=2
  - 2 interleaved insertions × 2 layers = **4 Mamba blocks per AR step**
  - Per-(batch, mesh_node) persistent ssm_state (128×16) + conv_cache (128×3)
  - zero_init_output=True

---

## 3. Parameter counts

Same as v22 (architecture identical): **11,090,319 trainable residual params**.

| Component | Params |
|---|---|
| GraphCast residual head | 10,231,891 |
| Temporal Mamba (851,456 total = 2 insertions × 425,728) | 851,456 |
| `temporal_residual_head` (83×83 output) | 6,972 |
| Frozen GraphCast backbone | ~5.5M (DeepMind, not trained) |

---

## 4. Training hyperparameters

| Parameter | Value | Δ vs v22 |
|---|---|---|
| `precision` | bf16 | same |
| `batch_size` | 1 | same |
| `lr` | 1e-4 | same |
| `weight_decay` | 1e-4 | same |
| `grad_clip` | 1.0 | same |
| `warmup_steps` | 200 | same |
| `seed` | 18 | (v22: 22) |
| `input_duration` | 12h (2 input steps) | same |
| `target_steps` | 1 | same |
| **`ar_tail_K`** | **14** | **v22: 22 ← key difference** |
| **`bptt_steps`** | **16** | **v22: 24** |
| **`sequential_segment_steps`** | **64** | **v22: 96** |
| `max_steps` | 26000 | same |
| `checkpoint_every` | 500 | same |
| `resume_from` | `v18_v2/K1_from_v15v2_20k/v13_residual_step23000.pkl` (start_step=23001) | different K=1 anchor |

**The only architectural-relevance differences are the AR/BPTT/segment lengths**. v20 has roughly 33% less activation memory than v22 (BPTT 16 vs 24).

---

## 5. Headline result

At lead 240h on 2022 eval set (lat-w skill score vs DeepMind GraphCast-small baseline at 1°):

- **Total MSE-improvement: +3.56%**
- Per-variable RMSE-imp peaks (selected): noticeably lower than v22 (v22 peaks at 2m_T ≈ +3.8%, MSL ≈ +4.2%; v20 peaks are smaller because K=14 sits below K\*)
- See `plots/` directory

v20 K=14 sits **below the K\* transition (≈16→18)**, so it does not show the long-horizon stabilization gain that v22 K=22 exhibits. v20 is the strongest "below-K\*" baseline in the K-scan.

---

## 6. Why v20 vs v22 matters for the K\* story

The K-scan shows a transition in long-lead improvement somewhere between K=16 and K=18 (apparent K\*). v20 K=14 vs v22 K=22 brackets this transition:

| | K | BPTT | Total MSE-imp @ 240h |
|---|---|---|---|
| v20 | 14 | 16 | +3.56% |
| v22 | 22 | 24 | +12.69% |
| Δ | +8 | +8 | +9.13 pp |

The ~3.6× improvement at lead 240h going from K=14 → K=22 (same architecture, same everything else) is the strongest evidence for a long-AR training regime change in this codebase.

---

## 7. Audit notes (same as v22)

- Real Mamba source: `src/models/mamba/modules/temporal_mesh_mamba_Ilya.py`
- `src/models/temporal_mesh_mamba_stateful.py` is a leftover earlier prototype — **does NOT represent v20/v22 behavior**.
- Config keys `temporal_d_state / d_conv / dt_rank / d_inner / hidden_size` ARE all consumed via `src/models/mamba/residual_mamba/runtime.py`.

---

## 8. Plots in this directory

- `plots/4metrics_v20_K20.png` — Total MSE-imp + per-var RMSE
- `plots/per_variable_improvement_K20.png` — 12 per-variable curves
- `plots/extreme_records_v20_K20.*` — extreme-event scoring
- `eval_jsons/v20_K14_K20.json` — raw eval

Same eval protocol as v22: 240 anchors × 40 lead steps (6h–240h), lat-weighted, vs frozen GraphCast-small baseline.
