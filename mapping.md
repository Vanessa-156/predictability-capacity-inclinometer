# Script ↔ paper mapping

One row per released module: filename · reference point in the manuscript (main-text §/table
or supplementary material) · purpose. All modules are flat in the repository root; the
`Category` column records the group they belong to in the internal project layout.

## A. Pipeline modules (the implementation vehicle)

| Script | Category | Reference | Purpose |
|---|---|---|---|
| `Ablation_Experiment.py` | `main` | §3, §4.2, table 5, S5 | Spatio-temporal graph network (TCN + GCN + vertical GCN + cross-node attention + physics gate), ablation variants V0–V8, data pipeline (`load_raw_data_with_meta`, `build_node_features`, `build_adjacency`, `prepare_data`, `output_space_floor`, `global_metrics`) |
| `Comparison_Experiment.py` | `main` | §4.4.1, table 5, S26 | External and classical baselines under one shared training budget |
| `PCA_Experiment.py` | `main` | S16, S21 | PCA-capacity control for the output representation |
| `Spatial_LOOCV.py` | `main` | §4.2.2, S6 | Leave-one-section-out protocol with node-field masking |
| `Stress_Test_Experiment.py` | `main` | §4.3 | Stress conditions over the predictability capacity |
| `_gen_seeds.py` | `main` | S5 | Seed list generation |
| `_robust_io.py` | `main` | — | Retrying write layer used by every module |

## B. Analysis scripts

| Script | Category | Reference | Purpose |
|---|---|---|---|
| `_k_scan_probe.py` | `interval` | S20; K = 7/13/21 PCA scan | Output-representation capacity scan (§4.4.2) |
| `_loocv_mask_effect_check.py` | `loopv` | S6 | Mask-effect control for the leave-one-section-out protocol |
| `_loocv_section_gap_control.py` | `loopv` | S6 | Section-gap / difficulty control |
| `_loocv_writeup_analysis.py` | `loopv` | S6 | Pooled leave-one-section-out write-up |
| `_multitarget_vicd.py` | `vicd` | S3 | Swapped-target records |
| `_p02_rollorig.py` | `capacity` | S17 | Rolling-origin walk-forward, six folds |
| `_p02_timeblock_inference.py` | `capacity` | §4.4.1, S14 | Time-block bootstrap and variance decomposition |
| `_p03_config_capacity.py` | `capacity` | S21 | Fixed-K configuration/separation experiment |
| `_p04_increment_caliber.py` | `capacity` | S13, S20 | Increment–cumulative dual calibre |
| `_p05_stage_input.py` | `capacity` | S18 | Excavation-depth input rerun |
| `_p06_deep_neg.py` | `capacity` | S27 | Deep-node negative magnitudes |
| `_p06_no_deep.py` | `capacity` | S27 | Retrain without deep nodes |
| `_p06_selection_check.py` | `capacity` | S16 | Stage-1 selection leakage check |
| `_p06_stage2_nestedcv.py` | `capacity` | S16 | Stage-2 nested cross-validation |
| `_p07_block_conformal.py` | `interval` | S15 | Block-conformal calibration |
| `_p08_learning_curve.py` | `interval` | S24 | Learning-curve / model-size scan |
| `_p09_skill_states.py` | `interval` | S28 | Unit-granularity skill states |
| `_p10_noise_floor.py` | `interval` | S25 | Instrument noise floor |
| `_p11_swap_test.py` | `vicd` | S3-7.1 | Fisher swap-target test |
| `_p11_water_inject.py` | `vicd` | S3-7.2 | Water-injection ablation control |
| `_p12_p15_baseline_budget.py` | `interval` | S26 | Modern-baseline unified budget |
| `_p16_vicd_denoise_check.py` | `vicd` | S19 | Denoising-calibre sensitivity |
| `_p17_denoise_target_check.py` | `capacity` | S19 | Denoised-target calibre |
| `_p17_rollorig_zero.py` | `capacity` | S17 | Cross-fold zero-change baseline |
| `_persist_baseline.py` | `interval` | S15 | Zero-change and last-increment speed baselines |
| `_uq_verify.py` | `interval` | S15 | Deep-ensemble uncertainty verification |
| `_vicd_layer2_criteria.py` | `vicd` | S3 | Layer-2 calibration |
| `_vicd_null_distribution.py` | `vicd` | S3 | §3.1 layer-1 null distribution, calibration and three tests |
| `_vicd_phase_recheck.py` | `vicd` | S-supp | Phase-wise water recheck |

Total: 6 pipeline modules + 29 analysis scripts.

## C. Example-data and driver scripts (not part of the paper's results)

| Script | Purpose |
|---|---|
| `make_example_data.py` | Synthetic monitoring data with the schema of the real files |
| `make_example_aux.py` | Auxiliary example inputs (cross-site/cross-scenario, bookkeeping tables) |
| `make_example_artifacts.py` | Runs `Ablation_Experiment` at reduced scale to produce example artefacts |
| `make_example_comparison.py` | Runs `Comparison_Experiment` at reduced scale |
| `make_example_stress.py` | Runs `Stress_Test_Experiment` at reduced scale |
| `make_example_pca.py` | Runs `PCA_Experiment` at reduced scale |
| `make_example_loocv.py` | Runs `Spatial_LOOCV` at reduced scale |
| `run_all.py` | Runs every released script and reports pass/fail |
