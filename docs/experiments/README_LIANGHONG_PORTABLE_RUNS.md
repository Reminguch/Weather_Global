# Lianghong: GC–Mamba lambda sweep and fresh full-GC horizon study

Prepared 2026-11-12. This package runs **both studies on your cluster** using the
included source, local data paths and your scheduler settings. It does not contact
Della or depend on Della job IDs. This revision uses **constant `3e-7` from the
first update for full GC, with no warmup or decay**. It supersedes the earlier
November 12 full-GC cosine template. GC–Mamba's M1 recipe is unchanged.
Use a fresh full-GC output directory; previously trained cosine checkpoints
cannot be resumed or relabeled as this constant-rate study.

The code archive includes the computational source frozen for the November 12
GC–Mamba submission, additional portable launchers, exact configuration templates,
training anchors, normalization statistics, evaluation dates and a pinned Python
dependency list. The separate weights archive includes the public GraphCast-small
initializer and the historical M1 raw/SWA reference checkpoints. Training starts
fresh from public GraphCast; the M1 checkpoints are evaluation references only.

**Code-only delivery:** send the roughly2.3MB code archive and its matching
`SHA256SUMS`. Download the existing weights archive separately from the Della
directory below. This requires an authorized Della SSH account with read access
to the project directory; the filesystem path is not a public download URL.
No model weights are included in the code archive.

## Studies and counters

| Setting | GC–Mamba lambda sweep | Full-GC horizon study |
| --- | --- | --- |
| Fresh arms | lambda 0, 0.25, 0.5, 0.75, 1 | K8, K12, K16, K20; all four fresh |
| Objective | Native M1 MSE + lambda × stable AMSE excess; lambda 0 uses the original MSE path | Standard GC MSE, lambda 0; existing FP32 full-field calculation |
| Weather feedback gradients | Stopped; memory gradients retained | Full gradients |
| Training | Initial 500 + main 2,000 | One continuous 2,500-update schedule |
| Learning rate | Initial `1e-4`; main spatial `1e-4 → 1e-5`, Mamba 3× | Constant `3e-7` throughout |
| LR ramp | 50 updates in each stage | None |
| Data order | Historical contiguous order, restarted at main stage | Fixed shuffled order shared by all four horizons |
| Allocation 1 | Initial 500; parameters-only reset; main 750 | Updates 1–1,250 |
| Allocation 2 | Exact resume main 750→2,000 | Exact resume 1,250→2,500 |
| SWA stored steps | Main 500/1,000/1,500/2,000 | 1,000/1,500/2,000/2,500 |

Both use seed 22, batch 4, 96-step segments, a 24-step training window, BF16
model computation, FP32 weather tape and the same 2015–2021 training data.
GC–Mamba has K20. Full-GC teacher prefixes are 16/12/8/4 for K8/12/16/20.
There is **no reset at the 1,250-update allocation boundary** in either study.
GC–Mamba's initial-to-main boundary resets optimizer, RNG, data cursor and memory
while retaining learned parameters, matching M1. Lambda applies in both stages.

These studies have different training recipes. Comparing them does not isolate
architecture or memory. Report unfavorable results too. Finite loss increases,
numerical drift and weak metrics are warnings; nonfinite computation and corrupted
or mismatched inputs/checkpoints are errors.

## 1. Unpack and activate Python

Run from the directory containing the code archive and its matching `SHA256SUMS`:

```bash
set -euo pipefail
sha256sum -c SHA256SUMS
tar -xzf lianghong_weather_20261112_constant_lr3em7_code.tar.gz
export LH_PACKAGE="$PWD/lianghong_weather_20261112_constant_lr3em7"
(cd "$LH_PACKAGE" && sha256sum -c SHA256SUMS)
```

If the weights archive was not sent with the code, download it using your
authorized Della SSH login. Replace `YOUR_LOGIN@DELLA_HOST` with the login and
host you normally use; credentials are not included in the package.

```bash
export LH_DELLA=YOUR_LOGIN@DELLA_HOST
rsync -a --partial --info=progress2 \
  "$LH_DELLA:/scratch/gpfs/DABANIN/iv9432/Weather_global/artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/lianghong_weather_20261112_constant_lr3em7_weights.tar.gz" \
  ./
sha256sum -c "$LH_PACKAGE/portable/weights_download.SHA256SUMS"
```

The archive is495.19MB. Its source path, size and SHA256 are also recorded in
`portable/weights_download.json`. If SSH access is unavailable, the same weights
archive must be transferred separately or placed in an accessible download
location; this package does not create public hosting.

After obtaining the archive, extract it alongside the code and activate Python:

```bash
tar -xzf lianghong_weather_20261112_constant_lr3em7_weights.tar.gz
export LH_WEIGHTS="$PWD/weights"
(cd "$LH_WEIGHTS" && sha256sum -c SHA256SUMS)
cd "$LH_PACKAGE"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements.lock.txt
export PYTHONPATH="$LH_PACKAGE:$LH_PACKAGE/third_party/graphcast"
```

The lock records the installed Della runtime, including JAX/JAXlib/CUDA plugin
0.9.0.1, rather than the newer JAX 0.9.2 in the earlier Lianghong result archive.
Use your site's compatible NVIDIA driver and CUDA module if required. The pinned
environment installation has not been tested on your cluster. Vendored GraphCast
comes from this package; do not install a newer upstream GraphCast over it.
If your existing environment is used instead, save `python -m pip freeze` and
label that environment difference in the results.

The portable wrappers preserve your activated environment. They do not source
the Della-specific `scripts/graphcast_env.sh`. If batch jobs need explicit
activation, set `WEATHER_ACTIVATE_SCRIPT` to an absolute local shell script that
activates this environment; otherwise Slurm inherits the activated PATH.

## 2. Set local data/output paths

Edit these four paths to locations on your cluster:

```bash
export LH_PREPARED=/absolute/path/to/prepared_stream/res1
export LH_DEV_DATA=/absolute/path/to/wb2_res1_levels13_2015_2022.zarr
export LH_ANNUAL_DATA=/absolute/path/to/arco_res1_levels13_2023_eval.zarr
export LH_OUTPUT=/absolute/path/to/new_weather_experiments
mkdir -p "$LH_OUTPUT"
export LH_GCM="$LH_OUTPUT/gcm_m1_lambda"
export LH_FULLGC="$LH_OUTPUT/full_gc_ar_constant_lr3em7"
export LH_DEV="$LH_OUTPUT/comparison_2022"
export LH_ANNUAL="$LH_OUTPUT/comparison_2023"
```

`LH_PREPARED` is the existing **24-file prepared array store, about 238.35 GiB**,
not a raw ERA5/Zarr directory. Its external source location and complete file-size
inventory are in `portable/external_assets.json`; `portable/prepared_files.txt`
is an rsync file list. The code/weights archives do not contain the large prepared
store or either evaluation dataset. Reuse an identical local copy, or transfer
the complete store. For example, after replacing the login/host:

```bash
mkdir -p "$LH_PREPARED"
rsync -a --info=progress2 --files-from=portable/prepared_files.txt \
  YOUR_LOGIN@DELLA_HOST:/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res1/ \
  "$LH_PREPARED/"
```

Setup checks the original hashes of metadata, coordinates, training anchors,
statistics and public weights. It does not reread all 238 GiB of variable arrays
to hash them. Keep a verified complete transfer and preserve file contents. The
full-GC order file is relocated locally, retaining exactly the same segments and
epoch permutations. Absolute paths inside original template/provenance JSONs
are historical locators; the generated training configurations use your paths.
Choose permanent absolute paths before preparing; exact resume expects them to
remain unchanged.

Evaluation uses exact date manifests, not random sample selection. If you use
Lianghong's revised 2023 truth with NCAR replacements, record that dataset version
and evaluate every model on that same local truth. Do not pool those scores with
the historical Della scores as though the input data were identical.

## 3. Prepare both fresh campaigns — CPU only

```bash
COMMON_INPUTS=(
  --package-root "$LH_PACKAGE"
  --prepared-root "$LH_PREPARED"
  --anchor-manifest-root "$LH_PACKAGE/inputs/anchors"
  --baseline-checkpoint "$LH_WEIGHTS/public_graphcast.npz"
  --stats-dir "$LH_PACKAGE/inputs/stats"
)

JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_gcm_lambda.py prepare \
  "${COMMON_INPUTS[@]}" --experiment "$LH_GCM"
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_full_gc_ar.py prepare \
  "${COMMON_INPUTS[@]}" --experiment "$LH_FULLGC"

JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_gcm_lambda.py plan --experiment "$LH_GCM"
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_full_gc_ar.py plan --experiment "$LH_FULLGC"
```

These commands write configs/manifests and freeze a local source copy. They do
not submit jobs. Fresh output directories are required. GC–Mamba task indices
0–4 map to lambda 0/.25/.5/.75/1; full-GC indices 0–3 map to K8/12/16/20.

## 4. Submit training on your Slurm cluster

Edit `LH_GPU_SITE` to valid local scheduler options. The example account,
partition and GPU name are placeholders, not Della settings:

```bash
LH_GPU_SITE=(--account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:a100:1)

submit_training() {
  local study="$1" campaign="$2" last_task="$3" first second wrapper
  wrapper="$campaign/source/scripts/experiments/v24_portable_${study}.sh"
  first=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL \
    --array="0-$last_task" --cpus-per-task=8 --mem=384G --time=30:00:00 \
    --job-name="${study}-part1" \
    --output="$campaign/logs/part1_%A_%a.out" --error="$campaign/logs/part1_%A_%a.err" \
    "$wrapper" train "$campaign" 1)
  first="${first%%;*}"
  second=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL \
    --array="0-$last_task" --cpus-per-task=8 --mem=384G --time=30:00:00 \
    --dependency="aftercorr:$first" --kill-on-invalid-dep=yes \
    --job-name="${study}-part2" \
    --output="$campaign/logs/part2_%A_%a.out" --error="$campaign/logs/part2_%A_%a.err" \
    "$wrapper" train "$campaign" 2)
  second="${second%%;*}"
  printf 'part1=%s\npart2=%s\n' "$first" "$second" | tee "$campaign/submissions/local_training_ids.txt"
}

submit_training gcm_lambda "$LH_GCM" 4
submit_training full_gc_ar "$LH_FULLGC" 3
squeue -u "$USER"
```

Each array task needs **one 80 GiB GPU**, 8 CPUs and 384 GiB host RAM. Confirm
your site's `--gres` actually selects an 80 GiB card; an A100 name alone may also
include 40 GiB cards. Add your site's constraint/QoS if needed. No concurrency
throttle is prescribed. Do not rerun the submission function while its jobs
already exist. Job IDs are saved in each campaign's `submissions/` directory.

The 30-hour/384-GiB budget comes from comparable Della 1,250-update jobs taking
about 20–22 hours and roughly 234 GiB host RAM. Revised GC–Mamba runtime and
runtime on your hardware are unmeasured. Full-GC gets four fresh trajectories;
the old Della K20 is never reused. If local walltime limits are shorter than
30 hours, arrange a suitable partition before launching; do not change the
scientific LR schedule to match an allocation.

On a GPU allocation without Slurm, the exact commands for one arm are:

```bash
bash "$LH_GCM/source/scripts/experiments/v24_portable_gcm_lambda.sh" train "$LH_GCM" 1 0
bash "$LH_GCM/source/scripts/experiments/v24_portable_gcm_lambda.sh" train "$LH_GCM" 2 0
bash "$LH_FULLGC/source/scripts/experiments/v24_portable_full_gc_ar.sh" train "$LH_FULLGC" 1 0
bash "$LH_FULLGC/source/scripts/experiments/v24_portable_full_gc_ar.sh" train "$LH_FULLGC" 2 0
```

Repeat with the other task indices. Each second part requires its completed
first-part proof. For a failed partial allocation, preserve the logs/checkpoints
for inspection rather than deleting output or silently restarting that arm.

After all training tasks complete, validate the endpoints on a CPU node:

```bash
bash "$LH_GCM/source/scripts/experiments/v24_portable_gcm_lambda.sh" finish "$LH_GCM"
bash "$LH_FULLGC/source/scripts/experiments/v24_portable_full_gc_ar.sh" finish "$LH_FULLGC"
```

## 5. Development comparison: exact MSE, spectra and SWA

After training completes, prepare all 34 new models plus historical M1 raw/SWA
on the same 32 development starts in 2022 and 40 six-hour leads:

```bash
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_comparison.py prepare \
  --package-root "$LH_PACKAGE" --experiment "$LH_DEV" \
  --gcm-campaign "$LH_GCM" --full-gc-campaign "$LH_FULLGC" \
  --data-path "$LH_DEV_DATA" --initialization-manifest portable/development32.json --year 2022 \
  --m1-raw "$LH_WEIGHTS/M1_raw.pkl" --m1-swa "$LH_WEIGHTS/M1_swa.pkl" \
  --m1-config templates/main.json

JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_comparison.py plan --experiment "$LH_DEV"
```

GC–Mamba contributes final raw/SWA for five lambdas (10 models). Full-GC
contributes updates 500/1,000/1,500/2,000/2,500 and SWA for each horizon (24).
M1 adds two references. Each forecast also computes the public GraphCast baseline.
SWA is an equal-weight inference average; it is not an optimizer restart.

```bash
DEV_JOB=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL \
  --array=0-35 --cpus-per-task=4 --mem=20G --time=02:30:00 \
  --job-name=weather-dev32 \
  --output="$LH_DEV/logs/evaluate_%A_%a.out" --error="$LH_DEV/logs/evaluate_%A_%a.err" \
  "$LH_DEV/source/scripts/experiments/v24_portable_comparison.sh" evaluate "$LH_DEV")
printf '%s\n' "$DEV_JOB" > "$LH_DEV/development_job_id.txt"
```

After the array completes:

```bash
bash "$LH_DEV/source/scripts/experiments/v24_portable_comparison.sh" report "$LH_DEV"
```

Inspect `report/report.json` and the generated tables/figures. Preserve exact
loss curves, forecast/truth/error/cross powers, power ratios and spectral
correlations, plus the MSE–power-mismatch tradeoff. Power mismatch pools powers
over dates first, then averages absolute relative errors over 83 channels,
degrees 1–179 and leads. It is not the training objective or phase skill.

## 6. Annual comparison, selected using 2022 only

Full-GC finalists are final raw, SWA and the lowest-exact-MSE development raw
checkpoint for each horizon if distinct; exact ties choose the earlier step.
GC–Mamba keeps all five final raw/SWA pairs. Freeze this selection before reading
2023 metrics. Prior 2023 exposure must be disclosed.

```bash
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_comparison.py prepare \
  --package-root "$LH_PACKAGE" --experiment "$LH_ANNUAL" \
  --gcm-campaign "$LH_GCM" --full-gc-campaign "$LH_FULLGC" \
  --data-path "$LH_ANNUAL_DATA" --initialization-manifest portable/annual122.json --year 2023 \
  --development-report "$LH_DEV/report/report.json" \
  --m1-raw "$LH_WEIGHTS/M1_raw.pkl" --m1-swa "$LH_WEIGHTS/M1_swa.pkl" \
  --m1-config templates/main.json

JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_comparison.py plan --experiment "$LH_ANNUAL"
```

Use the exact task range printed by `plan` (20–24 tasks with both M1 references,
depending on whether intermediate full-GC checkpoints are selected):

```bash
LH_LAST_TASK=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["models"])-1)' "$LH_ANNUAL/manifest.json")
sbatch "${LH_GPU_SITE[@]}" --export=ALL --array="0-$LH_LAST_TASK" \
  --cpus-per-task=4 --mem=20G --time=10:00:00 --job-name=weather-annual122 \
  --output="$LH_ANNUAL/logs/evaluate_%A_%a.out" --error="$LH_ANNUAL/logs/evaluate_%A_%a.err" \
  "$LH_ANNUAL/source/scripts/experiments/v24_portable_comparison.sh" evaluate "$LH_ANNUAL"

# After every annual task completes:
bash "$LH_ANNUAL/source/scripts/experiments/v24_portable_comparison.sh" report "$LH_ANNUAL"
```

The evaluation budgets are conservative extrapolations from existing 1°
forecasts with spherical diagnostics; record actual walltime and peak memory
on your cluster. Per-initialization scores support paired date-block uncertainty;
these fixed-checkpoint intervals do not measure training-seed variability.

## 7. Development-fitted smoothing for each finalist

Use the portable smoothing launcher for each selected checkpoint, including M1
references. It evaluates both that model and public GC on the same dates. Give
each model its own directory and 2022-fitted filters. The exact commands below
show the lambda-0 final raw model; substitute the selected checkpoint/config/ID
for the other finalists listed in the annual manifest.

```bash
export LH_SMOOTH="$LH_OUTPUT/smoothing_gcm_l000_raw"
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_smoothing.py prepare \
  --package-root "$LH_PACKAGE" --experiment "$LH_SMOOTH" \
  --checkpoint "$LH_GCM/runs/gcm_m1matched_s22_l000/checkpoints/checkpoint_step00002000.pkl" \
  --run-config "$LH_GCM/configs/gcm_m1matched_s22_l000_main.json" --model-id gcm_l000_raw \
  --baseline-checkpoint "$LH_WEIGHTS/public_graphcast.npz" --stats-dir "$LH_PACKAGE/inputs/stats" \
  --development-data "$LH_DEV_DATA" --annual-data "$LH_ANNUAL_DATA" \
  --development-manifest portable/development32.json --annual-manifest portable/annual122.json
```

Run calibration tasks 0–3 on GPUs, then `fit` on a CPU, then annual tasks 0–7
on GPUs, then `report` on a CPU. Each GPU shard uses 4 CPUs, 20 GiB RAM and a
150-minute reference budget. Here is the complete dependency chain; edit the
CPU partition/account for your cluster:

```bash
LH_CPU_SITE=(--account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION)
LH_SMOOTH_WRAPPER="$LH_SMOOTH/source/scripts/experiments/v24_portable_smoothing.sh"
CAL_JOB=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL --array=0-3 \
  --cpus-per-task=4 --mem=20G --time=02:30:00 \
  --output="$LH_SMOOTH/logs/calibrate_%A_%a.out" --error="$LH_SMOOTH/logs/calibrate_%A_%a.err" \
  "$LH_SMOOTH_WRAPPER" calibrate "$LH_SMOOTH")
FIT_JOB=$(sbatch --parsable "${LH_CPU_SITE[@]}" --export=ALL \
  --dependency="afterok:${CAL_JOB%%;*}" --kill-on-invalid-dep=yes \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$LH_SMOOTH/logs/fit_%j.out" --error="$LH_SMOOTH/logs/fit_%j.err" \
  "$LH_SMOOTH_WRAPPER" fit "$LH_SMOOTH")
FILTER_JOB=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL --array=0-7 \
  --dependency="afterok:${FIT_JOB%%;*}" --kill-on-invalid-dep=yes \
  --cpus-per-task=4 --mem=20G --time=02:30:00 \
  --output="$LH_SMOOTH/logs/evaluate_%A_%a.out" --error="$LH_SMOOTH/logs/evaluate_%A_%a.err" \
  "$LH_SMOOTH_WRAPPER" evaluate "$LH_SMOOTH")
sbatch "${LH_CPU_SITE[@]}" --export=ALL \
  --dependency="afterok:${FILTER_JOB%%;*}" --kill-on-invalid-dep=yes \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$LH_SMOOTH/logs/report_%j.out" --error="$LH_SMOOTH/logs/report_%j.err" \
  "$LH_SMOOTH_WRAPPER" report "$LH_SMOOTH"
```

Fit per-channel/lead-group Gaussian and regularized four-band profiles on 2022,
with the established Gaussian candidates, knots 1/10/40/179 and eta 0.01. Apply
filtering only to scored output; feed raw predictions back. Do not clip forecasts
or truth. Retain positivity and finite drift diagnostics.

For the full strength curve, after that model's fitted-smoothing report completes:

```bash
export LH_SWEEP="$LH_OUTPUT/smoothing_strength_gcm_l000_raw"
JAX_PLATFORMS=cpu python scripts/experiments/run_v24_portable_smoothing.py prepare-sweep \
  --package-root "$LH_PACKAGE" --experiment "$LH_SWEEP" --parent-root "$LH_SMOOTH"
STRENGTH_JOB=$(sbatch --parsable "${LH_GPU_SITE[@]}" --export=ALL --array=0-7 \
  --cpus-per-task=4 --mem=20G --time=03:00:00 \
  --output="$LH_SWEEP/logs/evaluate_%A_%a.out" --error="$LH_SWEEP/logs/evaluate_%A_%a.err" \
  "$LH_SWEEP/source/scripts/experiments/v24_portable_smoothing.sh" sweep-evaluate "$LH_SWEEP")
sbatch "${LH_CPU_SITE[@]}" --export=ALL \
  --dependency="afterok:${STRENGTH_JOB%%;*}" --kill-on-invalid-dep=yes \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output="$LH_SWEEP/logs/report_%j.out" --error="$LH_SWEEP/logs/report_%j.err" \
  "$LH_SWEEP/source/scripts/experiments/v24_portable_smoothing.sh" sweep-report "$LH_SWEEP"
```

Strength tasks use a 20-GiB/3-hour reference budget, from completed 17–18-date
shards with similar filtering. The curves include fixed Gaussian widths
0/.25/.5/.75/1/1.5/2/3/4/8° and alpha 0:.1:1 blends of the two frozen fitted
profiles: 30 distinct points per model, with a shared raw point. Report all
prespecified points; do not select a new winner on 2023.

## Return these results

Send the campaign manifests, `checks/`, configs, train metrics, final raw/SWA
checkpoint identities, comparison reports/figures, per-start scores, pooled
spectra, fitted filters and smoothing reports. Include the environment record,
truth-data version and `sacct` elapsed/MaxRSS records. Preserve trained weights
for follow-up evaluations. Keep training completion separate from evidence of
scientific improvement; no new result is claimed by this handoff.
