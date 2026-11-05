# v2 validation evidence

Implementation source: `db93a3a`, based on README spec `52da7e75`.
All final evidence refers to the same source identity.

- `cpu-tests.log`: 46 CPU tests passed. Dependency deprecation warnings only.
- `cpu-and-import.json`: schema 2 and imported v1 input provenance validated
  through the v2 pipeline context. Existing stats and caches were reused.
- `cache-regression.json`: all 60 fixed validation chunks; zero-head loss
  **3.5251650667**, matching the five v1 contributions and the v2 reference.
- `gpu-smoke.json`: H200, full 796-input / 145-output mesh-4 width-512 model.
  Real cached ERA5 input forward pass and finite loss, three synthetic full
  BPTT24 optimizer updates, bitwise zero head and cloud passthrough. Same-process
  restore maximum difference **0**.
- `fresh-process-resume.json`: uninterrupted 3 updates versus 2+1 in separate
  processes, using the production trainer. Parameters, Mamba memory, Adam state,
  RNG and cursor are all bitwise identical, maximum difference **0**.
- `native-smoke.json`: native rollout and independent evaluator compatibility
  passed using the official pretrained NeuralGCM and actual prepared data.
- `qualification.json` and `qualification-pytest.log`: the frozen job verified
  the GPU/native/restart receipts and reran all 46 CPU tests successfully.
- `gpu-submission.json`: ailab H200 gpu-test job 16928171, one-hour allocation.

These are implementation tests. They do not demonstrate v2 forecasting skill
or constitute a new 2k training experiment. No v1 checkpoint is reused as a v2
model. Historical v1 training and results remain separate.
