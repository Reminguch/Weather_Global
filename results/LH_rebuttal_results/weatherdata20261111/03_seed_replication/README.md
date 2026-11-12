# M1 seed replication and memory controls

Five independent GraphCast–Mamba M1 carry/carry runs use training seeds 22–26, ERA5 from 2015–2021, 500 warm-up updates, and 2,000 main-stage updates. SWA averages the main-stage checkpoints at 500, 1,000, 1,500, and 2,000. Each forecast starts with zero temporal state and carries it through the rollout. The local runs reconstruct the published recipe from Appendix A.1 and the repository configs. The original Della configs and frozen training environment were unavailable, and the local seed 22 is a fresh retraining. [Published paper](https://arxiv.org/html/2610.28512v1) · [Training manifest](provenance/seed_training_manifest.json) · [Resolved configs](provenance/resolved_configs/)

The common test cohort contains 122 starts at 00 UTC every third day from January 1 to December 30, 2023, with 40 six-hour leads on the native 1° grid. Local evaluation uses independent NCAR replacements for stale ERA5 pressure fields at three valid times, affecting nine start/lead cells. The archived paper values use the historical evaluation. [Initialization manifest](provenance/initializations_2023_stride3.json) · [Truth audit](provenance/native_truth_audit.json)

## Five-seed SWA results

| Exact GraphCast loss reduction | Local five-seed mean ± sample SD | Local range | Published seed 22 |
|---|---:|---:|---:|
| 10-day rollout | 14.17% ± 0.43 pp | 13.69%–14.84% | 14.42% |
| 240-hour lead | 21.03% ± 0.75 pp | 20.08%–22.15% | 21.66% |

Each seed uses the unmodified public GraphCast-small trajectory from its own evaluation. The seed summary provides date-block intervals for each fixed model. The curve band is one sample standard deviation across the five training seeds. [Seed summary CSV](seed_summary.csv) · [Seed summary JSON](seed_summary.json) · [Four-panel figure](forecast_skill_seed_replication_paper.pdf) · [Reduction curves](forecast_skill_seed_replication_bd_std.pdf) · [Curve data and field definitions](forecast_skill_seed_replication_data/README.md)

## Fig. 6 memory controls

The local seed 22 carry/reset arm uses the same raw main-step 2,000 weights as carry/carry and resets the memory at each forecast step. The reset/reset arm uses its own 500-update warm-up and 2,000-update main run with memory reset during training and forecasting. The figure compares these controls and the five raw M1 carry models with GraphCast fine-tuned for 2,500 total updates. All lines use the public GraphCast trajectory from the local raw seed 22 carry run as their common denominator. The matched reset training control has one local training seed. [Figure](fig6_controls/gcft_vs_m1_2500_total_fig6_controls.pdf) · [Curve data](fig6_controls/gcft_vs_m1_2500_total_fig6_controls.json) · [Reset manifest](provenance/fig6_reset_manifest.json)

The compressed score table contains one row per model, initialization, and lead for five SWAs and the eight Fig. 6 comparison arms. It retains each run's recorded public baseline loss and model loss. Pool losses before taking ratios. The M1 rows also include cosine-latitude-weighted 2 m temperature MSE in K², which can be pooled before taking the square root. The fine-tuned GraphCast evaluator exports exact GraphCast loss. [Per-start scores](per_start_scores.csv.gz) · [Checkpoint and evaluation digests](provenance/checkpoint_identities.json) · [Copied source digests](provenance/copied_sources.json)

## Additional checkpoint comparisons

- The raw-versus-SWA comparison covers all five seeds at main updates 500, 1,000, 1,500, and 2,000 plus final SWA. Each seed uses the public baseline from its raw step 2,000 evaluation. [Figure](extras/raw_vs_swa/m1_raw_vs_final_swa_lead_time.pdf) · [Curve data](extras/raw_vs_swa/m1_raw_vs_final_swa_lead_time.json)
- The seed 22 continuation evaluates raw main checkpoints at 2,250, 2,500, 2,750, and 3,000 alongside raw step 2,000 and SWA. Extending the cosine schedule to 3,000 raised the first resumed spatial learning rate from approximately `1e-5` to `3.32e-5`. [Summary CSV](extras/seed22_3k_continuation/summary.csv) · [Training provenance](extras/seed22_3k_continuation/training_provenance.json)
