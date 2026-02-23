# Weather Global Project Structure

This layout is designed for multiple model backends (`GraphCast`, `NeuralCGM`, and future additions) with one shared training and evaluation pipeline.

```text
Weather_global/
├─ README.md
├─ GraphCast.md
├─ structure.md
├─ pyproject.toml
├─ .gitignore
├─ configs/
│  ├─ model/
│  │  ├─ graphcast_toy.yaml
│  │  ├─ graphcast_base.yaml
│  ├─ data/
│  │  └─ era5_1deg_13lev.yaml
│  └─ run/
│     ├─ train_toy.yaml
│     ├─ evaluate.yaml
│     └─ rollout.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ Data.md
├─ artifacts/
│  ├─ checkpoints/
│  ├─ stats/
│  └─ logs/
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ rollout.py
├─ src/
│  ├─ models/
│  │  ├─ base.py              # shared Predictor interface/protocol
│  │  ├─ registry.py          # model_name -> builder
│  │  ├─ graphcast/
│  │  │  └─ adapter.py        # GraphCast adapter + runtime logic (stub + real backend)
│  ├─ pipelines/
│  │  ├─ train.py             # generic train loop using base interface
│  │  ├─ evaluate.py
│  │  └─ rollout.py
│  ├─ data/
│  │  └─ contracts.py         # canonical batch schema
│  └─ utils/
│     ├─ io.py
│     └─ logging.py
└─ tests/
   ├─ test_registry.py
   ├─ test_graphcast_adapter.py
   ├─ test_train_pipeline.py
   └─ test_data_contracts.py
```

## Rules

1. Keep all importable code under `src/`.
2. Keep shared logic in `pipelines/` and `models/base.py`.
3. Keep model-specific internals isolated under `models/<model_name>/`.
4. Add new models only via `models/registry.py` and config (`model.name`).
5. Keep one canonical data contract in `data/contracts.py`; model adapters are responsible for conversion.
6. Store runtime outputs under `artifacts/`, not in source directories.
