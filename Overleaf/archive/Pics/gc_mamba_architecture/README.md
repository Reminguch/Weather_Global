# GraphCast–Mamba architecture figures

Three coordinated figures for the principal GraphCast–Mamba configuration in
`Overleaf/main_v2.tex`. Each is provided as a vector PDF, SVG, and PNG preview.

| File stem | Content |
| --- | --- |
| `01_full_model` | Shared inputs, frozen GraphCast, separate trainable correction, output addition, and weather feedback |
| `02_correction_branch` | Two graph steps, two Mamba layers after each step, encoder-to-decoder grid features, and spatial versus temporal processing |
| `03_temporal_rollout` | Three six-hour forecast steps with separate weather and memory paths |

The diagram source is `scripts/plot_gc_mamba_architecture.py`. Regenerate from the
repository root:

```bash
source scripts/graphcast_env.sh
python scripts/plot_gc_mamba_architecture.py
```

Use PDFs in LaTeX. `figures.tex` contains three ready-to-use figure environments;
from a manuscript in `Overleaf/`, insert:

```latex
\input{Pics/gc_mamba_architecture/figures.tex}
```

The manuscripts include the correction-branch and temporal-rollout PDFs.
Those PDFs contain manual artwork refinements. The terminology pass preserves
that artwork and refreshes the matching SVG and PNG previews from the PDFs.
Use `Overleaf/tools/standardize_figure_labels.py` to reproduce the approved
labels on the preserved artwork; regenerating with the original diagram
script produces its original layout.

## Notation and scope

- `X_t` is the two-frame weather input plus known forcings. `Yhat_t` denotes the
  next forecast produced from that window, following the manuscript's notation.
- `m_t` bundles all Mamba SSM states and causal-convolution caches. Superscripts
  in the correction figure index graph stages; each stage contains two Mamba
  layers with separate recurrent states. Memory is also separate per mesh node
  and batch element. Parameters are shared across mesh nodes and forecast time,
  not necessarily across stages or layers.
- The rollout shows cold initialization and combined-prediction feedback. Updating
  a two-frame window retains the previous latest frame and appends the forecast;
  forcings advance to the next step.
- The correction has its own encoder, processor and decoder. It does not consume
  the frozen baseline's latent features. Its final output head starts at zero.
- Normalization, edge-feature updates, individual Mamba internals, and optimizer
  details are omitted to keep the figures readable. The illustrated model is the
  implemented GraphCast baseline plus correction, not the proposed NeuralGCM
  native-state correction pipeline.

Architecture references: `src/models/mamba/v24_Ilya/model.py`,
`src/models/mamba/v24_Ilya/config.py`, and the architecture section of
`Overleaf/main_v2.tex`.
