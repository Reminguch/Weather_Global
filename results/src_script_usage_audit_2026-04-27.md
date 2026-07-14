# `src` usage audit for `scripts/` entrypoints

Date: 2026-04-27

Scope:
- Reviewed every source file under `src/` excluding compiled cache artifacts.
- Traced reachability from runnable files under `scripts/` (`.py`, `.sh`, `.slurm`, notebooks where relevant).
- Counted both direct imports from `scripts/` and indirect imports reached through `src` and `third_party/graphcast`.

Assumption:
- "Used for running files in scripts" means reachable from the current `scripts/` workflows, not merely referenced by docs, tests, or historical notes.

## Used by current `scripts/` workflows

These files are reachable from current script entrypoints and should be kept.

- `src/__init__.py`
- `src/data/__init__.py`
- `src/data/contracts.py`
- `src/data/graphcast_dataset.py`
- `src/models/__init__.py`
- `src/models/base.py`
- `src/models/graphcast/__init__.py`
- `src/models/graphcast/adapter.py`
- `src/models/registry.py`
- `src/models/temporal_mesh_mamba.py`
- `src/models/temporal_mesh_mamba_Ilya.py`
- `src/pipelines/__init__.py`
- `src/pipelines/evaluate.py`
- `src/pipelines/train.py`

Notes:
- `src/models/temporal_mesh_mamba.py` and `src/models/temporal_mesh_mamba_Ilya.py` are not imported directly by `scripts/`, but they are imported by `third_party/graphcast/graphcast/graphcast.py` and `third_party/graphcast/graphcast/deep_typed_graph_net.py`, which are reached via `src/models/graphcast/adapter.py`.
- Package `__init__.py` files are considered used because importing submodules such as `src.data.graphcast_dataset` and `src.pipelines.train` executes their parent package initializers.

## Marked for deletion

These are not reachable from current `scripts/` workflows and are the deletion candidates I marked.

Files:
- `src/models/graphcast/Single_run.ipynb`
- `src/models/temporal_mesh_mamba_stateful.py`
- `src/pipelines/rollout.py`

Folders:
- `src/__pycache__/`
- `src/data/__pycache__/`
- `src/models/__pycache__/`
- `src/models/graphcast/__pycache__/`
- `src/models/graphcast/core/`
- `src/models/graphcast/core/__pycache__/`
- `src/pipelines/__pycache__/`

Why these are marked:
- `src/pipelines/rollout.py`: no current script imports or calls it.
- `src/models/temporal_mesh_mamba_stateful.py`: referenced by docs/results text, but not imported by current script-reachable code.
- `src/models/graphcast/Single_run.ipynb`: no current script references.
- `src/models/graphcast/core/`: contains no tracked source files, only compiled cache remnants.
- `__pycache__/` folders: generated artifacts, not source dependencies.

## Key script entrypoints that drive the live set

Direct `scripts -> src` usage is centered around:

- `scripts/train.py` -> `src.data.contracts`, `src.pipelines.train`
- `scripts/infer.py` -> `src.data.contracts`, `src.pipelines.evaluate`
- `scripts/inspect_graphcast_dataset.py` -> `src.data.graphcast_dataset`
- `scripts/training/finetune_graphcast.py` -> `src.data.graphcast_dataset`
- `scripts/training/graphcast_train/dataset.py` -> `src.data.graphcast_dataset`
- `scripts/training/verify_mamba_setup.py` -> `src.data.graphcast_dataset`
- `scripts/analyze_models/mae_vs_lead.py` -> `src.data.graphcast_dataset`
- `scripts/analyze_models/mamba_wmse_vs_res.py` -> `src.data.graphcast_dataset`
- `scripts/analyze_models/nyc_target_lead_eval.py` -> `src.data.graphcast_dataset`
- `scripts/analyze_models/nyc_width_error_by_res_mp.py` -> `src.data.graphcast_dataset`

Transitive `src` chain for training/inference:

- `src.pipelines.train` / `src.pipelines.evaluate`
- `src.models.registry`
- `src.models.graphcast.adapter`
- `third_party/graphcast/...`
- `src.models.temporal_mesh_mamba.py`
- `src.models.temporal_mesh_mamba_Ilya.py`

