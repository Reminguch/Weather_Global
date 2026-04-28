# Roadmap: Current Repo Map and Target Organization

## Summary

This roadmap documents the current `scripts/` and `src/` layout as it exists today, then proposes a staged reorganization that separates model families, training entrypoints, inference engines, data operations, and experiment/job wrappers.

The current codebase already contains three practical model and training families:

- `graphcast`: baseline GraphCast training and evaluation
- `graphcast_mamba`: GraphCast with temporal Mamba variants and segmented/chunked training
- `graphcast_residual`: residual-memory correction training built on top of a baseline GraphCast forecast

The recommended direction is thin CLI wrappers in `scripts/` and reusable implementation code in `src/`, introduced gradually rather than through a big-bang rewrite.

## 1. Current Repo Map

### Core data layer

- `src/data/graphcast_dataset.py`
  - Reusable dataset loader
  - Normalizes local/remote NetCDF and Zarr inputs into a GraphCast-compatible layout
  - Provides the cleanest existing boundary between data access and training code

### Core model layer

- `src/models/`
  - Contains temporal Mamba building blocks:
    - `temporal_mesh_mamba.py`
    - `temporal_mesh_mamba_stateful.py`
    - `temporal_mesh_mamba_Ilya.py`
  - `src/models/graphcast/` currently acts as a GraphCast adapter namespace placeholder
  - `src/models/__init__.py` is minimal and does not yet provide a strong model registry or family boundary

### Shared training internals

- `scripts/training/graphcast_train/`
  - Shared training internals used by multiple entrypoints
  - Current responsibilities include:
    - `config.py`: common run configuration and CLI parsing
    - `dataset.py`: local split creation, time handling, dataset preparation, cache policy
    - `batching.py`: sample/window construction and batch builders
    - `segments.py`: contiguous-window filtering, segment scheduling, chunk helpers
    - `model.py`: GraphCast checkpoint/stats loading and predictor construction
    - `eval.py`: evaluation helpers
    - `logging.py`: run config/log/checkpoint persistence
    - `prefetch.py`: background batch preparation
    - `bootstrap.py`: GraphCast import/setup helper

### Training entrypoints

- `scripts/training/train_graphcast.py`
  - Standard GraphCast training entrypoint
  - Uses `scripts/training/graphcast_train/` as its implementation layer

- `scripts/training/train_graphcast_segments.py`
  - Segmented/chunked-BPTT training entrypoint
  - Uses segment scheduling plus optional temporal Mamba settings through `RunConfig`

- `scripts/training/finetune_graphcast.py`
  - Additional GraphCast-oriented training entrypoint in the same family

### Residual-memory training area

- `scripts/training/residual_memory/`
  - Residual-model-specific package nested under training scripts
  - Current responsibilities include:
    - `config.py`: residual segment config and CLI parsing
    - `utils.py`: residual loss/eval/predict helpers and run config augmentation
    - `train_graphcast_residual_segments.py`: residual training entrypoint
    - `.slurm` wrappers for cluster execution
  - This area already behaves like a separate model/training family even though it lives under `scripts/training/`

### Top-level scripts area

- `scripts/`
  - Data download/staging/regridding:
    - `download_wb2_last30d.py`
    - `fetch_era5_rolling_window.py`
    - `stage_wb2_era5_yearly_append.py`
    - `regrid_resolution.py`
    - `run_regrid_4deg.sh`
  - Dataset inspection and utility scripts:
    - `inspect_graphcast_dataset.py`
    - `select_best_checkpoint.py`
    - `plot_train_val_loss.py`
  - Analysis scripts:
    - `scripts/analyze_models/*`
  - Environment helpers, demos, notebooks:
    - `graphcast_env.sh`
    - notebooks such as `demo.ipynb` and `explore_graphcast_dataset.ipynb`

### Infrastructure observation

- `src/pipelines/` currently exists but is effectively empty
- This makes it a possible future home for explicit workflow and inference orchestration, but today it is not a meaningful architectural layer

## 2. Current Dependency Flows

### Baseline GraphCast training

`data source -> src/data/graphcast_dataset.py -> scripts/training/graphcast_train/dataset.py -> scripts/training/graphcast_train/batching.py -> scripts/training/graphcast_train/model.py -> scripts/training/train_graphcast.py -> checkpoints/artifacts`

What this means in practice:

- data loading/layout normalization starts in `src/data/graphcast_dataset.py`
- training-specific filtering, split logic, caching, and task preparation happen in `graphcast_train/dataset.py`
- window construction and batching happen in `graphcast_train/batching.py`
- predictor construction and checkpoint/stat loading happen in `graphcast_train/model.py`
- orchestration and optimizer loop live in `train_graphcast.py`

### Segmented temporal GraphCast/Mamba training

`local ERA5 dataset -> graphcast_train dataset prep -> graphcast_train/segments.py -> temporal config in RunConfig -> src/models/temporal_mesh_mamba*.py via graphcast wrapper -> train_graphcast_segments.py -> checkpoints/artifacts`

What this means in practice:

- the training loop still reuses the baseline shared internals
- the distinct behavior comes from:
  - contiguous-window filtering
  - chronological segment construction
  - BPTT chunk scheduling
  - temporal module configuration in `RunConfig`
- the temporal model implementation is split across `src/models/temporal_mesh_mamba*.py`

### Residual-memory training

`baseline checkpoint + local ERA5 dataset -> graphcast_train shared dataset/batching helpers -> residual_memory/utils.py -> train_graphcast_residual_segments.py -> residual correction checkpoint`

What this means in practice:

- residual training depends on a baseline checkpoint as an explicit input
- it reuses common data prep and batching from `graphcast_train`
- residual-specific predictor/loss/eval behavior is added in `residual_memory/utils.py`
- the resulting artifact is not the same kind of checkpoint as baseline GraphCast; it is a correction model

### Data operations flow

`remote WB2/CDS -> download/stage scripts -> local .zarr/.nc dataset -> regrid script -> training/inference consumers`

Current concrete path examples:

- `gs://weatherbench2/... -> stage_wb2_era5_yearly_append.py -> local yearly Zarr`
- `CDS -> fetch_era5_rolling_window.py -> local NetCDF`
- `local dataset -> regrid_resolution.py -> coarser local dataset -> training scripts`

## 3. Current Pain Points

- Shared training logic lives under `scripts/training/graphcast_train/` instead of `src/`, so reusable implementation code still looks like script support code.
- `scripts/training/residual_memory/` is effectively a separate model family, but it is nested as a training subfolder rather than represented as its own model and training package.
- Data operations are mixed into generic `scripts/` instead of a dedicated data pipeline area.
- Model family boundaries are blurred between baseline GraphCast, temporal GraphCast+Mamba, and residual correction.
- Inference is not clearly separated per model family, and there is no explicit `src/inference/` structure today.
- Many `.slurm` files represent experiment history and execution variants rather than stable productized entrypoints.
- Temporal model code exists in `src/models/`, but training ownership and runtime ownership are still concentrated in `scripts/training/`.

## 4. Target Organization

The target structure should preserve the current script mapping first, then move reusable logic into clearer ownership boundaries.

### Recommended structure

- `src/data/`
  - `loaders/`
  - `download/`
  - `regrid/`
  - `staging/`

- `src/models/graphcast/`
  - baseline model adapter

- `src/models/mamba/`
  - temporal Mamba blocks and configs

- `src/models/graphcast_residual/`
  - residual correction model package

- `src/training/graphcast/`
  - baseline training loop, config, eval, batching interfaces

- `src/training/graphcast_mamba/`
  - segmented and temporal training loop, config, eval

- `src/training/graphcast_residual/`
  - residual training loop, config, eval

- `src/inference/graphcast/`
  - baseline inference engine

- `src/inference/graphcast_mamba/`
  - temporal and segment-aware inference engine

- `src/inference/graphcast_residual/`
  - baseline plus residual correction inference engine

- `scripts/training/`
  - thin CLI launchers only

- `scripts/data/`
  - thin wrappers for download, regrid, and staging

- `scripts/analysis/`
  - plots, diagnostics, comparisons

- `scripts/slurm/`
  - cluster job wrappers grouped by model family

### Ownership principle

The reorganization should treat `src/` as the implementation surface and `scripts/` as the operational entrypoint surface:

- `src/` owns reusable Python logic
- `scripts/` owns launchers, wrappers, and job submission convenience
- experiment-specific `.slurm` history should move out of the main training namespace

## 5. Model Family Definition

The roadmap should treat the repo as having three first-class model families.

### `graphcast`

- No explicit temporal memory module
- No segmented training required
- Standard autoregressive baseline
- Primary current entrypoints:
  - `scripts/training/train_graphcast.py`
  - `scripts/training/finetune_graphcast.py`

### `graphcast_mamba`

- GraphCast plus temporal memory module
- Uses Mamba-style temporal blocks from `src/models/temporal_mesh_mamba*.py`
- Supports segmented/chunked training and stateful temporal variants
- Primary current entrypoint:
  - `scripts/training/train_graphcast_segments.py`

### `graphcast_residual`

- Residual correction model on top of a frozen or precomputed baseline forecast
- Separate from baseline GraphCast
- Separate from pure temporal GraphCast/Mamba
- Primary current area:
  - `scripts/training/residual_memory/`

### Explicit recommendation

`scripts/training/residual_memory/` should become its own first-class model and training package, not just a training variant folder.

## 6. Inference Ownership

Each model family should own its own inference engine.

### Recommended inference split

- Baseline GraphCast inference engine
  - loads baseline checkpoint
  - runs standard rollout/predict flow

- Temporal GraphCast+Mamba inference engine
  - loads temporal checkpoint
  - handles temporal state and segment-aware rollout behavior when needed

- Residual correction inference engine
  - consumes a baseline forecast
  - loads residual correction checkpoint
  - applies correction to produce final forecast output

### Interface rule

Inference APIs should be independent from training scripts and should load checkpoints, stats, and dataset dependencies directly rather than importing training entrypoints.

## 7. Migration Phases

The migration should be incremental and low churn.

### Phase 1: Write and adopt the roadmap

1. Add this roadmap as the current-state map and target-state guide.
2. Start using the three-family vocabulary:
   - `graphcast`
   - `graphcast_mamba`
   - `graphcast_residual`

### Phase 2: Move reusable training internals

1. Move reusable logic from `scripts/training/graphcast_train/` into `src/training/common` or family-specific `src/training/...`.
2. Leave compatibility wrappers in `scripts/training/` so current workflows keep running.

### Phase 3: Promote residual memory to a first-class package

1. Promote `scripts/training/residual_memory/` into:
   - `src/models/graphcast_residual/`
   - `src/training/graphcast_residual/`
2. Keep a thin training launcher in `scripts/training/`.

### Phase 4: Introduce inference packages

1. Create:
   - `src/inference/graphcast/`
   - `src/inference/graphcast_mamba/`
   - `src/inference/graphcast_residual/`
2. Move rollout and checkpoint-loading responsibilities into those engines.

### Phase 5: Separate data operations

1. Move top-level data scripts under `scripts/data/`.
2. Later, move their implementation logic into:
   - `src/data/loaders/`
   - `src/data/staging/`
   - `src/data/regrid/`
   - `src/data/download/`

### Phase 6: Reduce `scripts/` to thin operational wrappers

1. Keep `scripts/training/` for thin CLI launchers.
2. Group analysis utilities under `scripts/analysis/`.
3. Move experiment-heavy `.slurm` files under `scripts/slurm/` grouped by model family.

## Important Interfaces to Define

These interfaces should be proposed now and implemented later.

- `src/training/<family>/cli.py`
  - model-family-specific training CLI entrypoint

- `src/inference/<family>/inference_engine.py`
  - checkpoint loading plus rollout/predict interface

- `src/data/loaders/*.py`
  - dataset opening and normalization

- `src/data/staging/*.py`
  - WB2/CDS fetch and local materialization

- `src/data/regrid/*.py`
  - resolution transforms

## Test and Validation Expectations

When the reorganization is later implemented, validate at least the following:

- each existing training script still maps cleanly to one target family
- baseline training path still works with the current local dataset format
- segmented temporal training still uses contiguous-window logic correctly
- residual training still depends on a baseline checkpoint and residual target logic
- each new inference engine can load its intended checkpoint type
- data download, staging, and regrid tools remain usable without importing training code

## Assumptions and Defaults

- This roadmap prioritizes mapping the existing repo before prescribing the final structure.
- The reorganization should preserve current experiment capability and should not immediately break existing `.slurm` workflows.
- The default direction is thin scripts and reusable logic in `src/`, introduced gradually.
- `roadmap.md` focuses on `scripts/` and `src/`; notebooks and analysis tools remain supporting utilities rather than the core architecture.
