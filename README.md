# Reproduction package: representation-conditional predictability ceiling of inclinometer monitoring

Version **v1.1.0** (2026-11-17). This package accompanies the manuscript

> Xu X., Jin D., He J., Sun X. *Calibrating the representation-conditional predictability ceiling of multi-section inclinometer monitoring: a protocol demonstrated on a deep excavation.* Submitted to *Tunnelling and Underground Space Technology*.

It contains the derived tables, the distributed data files and the convention register behind every number in the main text and the Supplementary Information (SI). Raw monitoring records are **not** distributed; the package contains only de-identified derived quantities.

## 1. Layout

```text
reproduction/
  README.md
  MANIFEST.csv          # inventory: path, category, rows, columns, status, description
  CITATION.cff
  LICENSE
  *.py                  # analysis code, example generators and run_all.py
  requirements.txt
  mapping.md            # script-to-section mapping
  example_data/         # synthetic monitoring data (fictional values)
  Tan_Digitized/        # digitized Tan series (CC BY 4.0) and its probe
  Liugou_Processed/     # synthetic example of the auxiliary cross-scenario input
  ML数据/               # synthetic anchor-force examples
  data/                 # eight files distributed with the submission
  tables/               # per-seed, per-window, per-node, per-configuration results
  logs/                 # convergence logs and the main-text cross-check
  legacy/               # superseded V5-protocol comparisons, kept for traceability
  figures_source/       # data behind Fig. 4, Fig. 5, Fig. S5 and the deep-node figure
  gaps/                 # S7 declared-gap index
```

The analysis code sits at the repository root, with `requirements.txt`, `mapping.md` and `run_all.py`. It contains the VICD criterion, the floor–reach decomposition, the interval recalibration, the leave-one-section-out protocol and the training pipeline, plus the synthetic example data generator (`make_example_data.py`) and its auxiliary generators. `Tan_Digitized/fig19_digitized.csv` is the actual digitized Tan series, reused under CC BY 4.0; all other example values are fictional.

### Code quick start

```bash
python make_example_data.py        # synthetic monitoring data -> example_data/
python make_example_aux.py         # auxiliary example inputs -> Tan_Digitized/, Liugou_Processed/
python make_example_artifacts.py   # ablation artefacts (reduced budget)
python make_example_comparison.py  # external-baseline artefacts
python make_example_stress.py      # stress-condition artefacts
python make_example_pca.py         # PCA-capacity artefacts
python make_example_loocv.py       # leave-one-section-out artefacts
python run_all.py                  # run the 30 core analysis scripts and report pass/fail
```

All generator steps use a reduced scale so the archive stays small; they demonstrate the pipeline and do not reproduce the reported numbers, which require the private monitoring data. `run_all.py` sets `EXAMPLE_MODE=1`, which turns the hard numeric anchors into warnings for the synthetic data.

## 2. Distributed data (`data/`)

| File | Rows | Content |
|---|---:|---|
| `SuppData_VICD_records_EN.csv` | 68 | Per-record VICD inputs: four statistics, 95% surrogate thresholds, $n$, $T_{\mathrm{obs}}$, $T_{\mathrm{thr}}$, binding indicator, Stage-1 verdict. |
| `SuppData_derived_target_EN.csv` | 4,260 | De-identified derived target: section- and layer-mean displacement against a step index (5 sections × 4 depth layers × 213 steps). No dates or coordinates. |
| `SuppData_stress_results.csv` | 70 | Stress-test cells: family, intensity, horizon, RMSE and SD for the fixed configuration and the baseline. |
| `SuppData_model_results.csv` | 7,541 | Long-format model results: experiment, variant, horizon, seed, metric, value, SD, $n$, detail, source file. |
| `SuppData_cross_site_results.csv` | 4,207 | Cross-site results: site, series, phase, horizon, metric, value, CI bounds, $n$, detail, source file. |
| `SuppData_S6_water_pairwise_EN.csv` | 216 | Tan water-level pairs: $n$, M1–M4 and MLEV, thresholds, $T_{\mathrm{obs}}$, $T_{0.95}$, verdict. |
| `SuppData_S6_liugou_records_EN.csv` | 48 | Liugou records (6 candidates × 8 targets): statistics, thresholds, $T_{\mathrm{obs}}$, $T_{0.95}$, verdict. |
| `MANIFEST.csv` | 8 | Inventory of the data files, with row and column counts and status. |

## 3. Tables (`tables/`)

The tables keep the identifiers used in the manuscript and SI. The most important entry points are:

| File | Used in | Content |
|---|---|---|
| `T1_long.csv` | Table 2, S5.3 | Per-model, per-horizon, per-seed comparison on both scales. |
| `T2_ablation_long.csv` | Table S6, S4.1 | Per-variant, per-horizon, per-seed ablation results. |
| `T5b_decomposition_ablation.csv` | S4.4 | Floor–reach decomposition per variant and horizon. |
| `T_stress_raw.csv` | §4.3, S5.1.4 | Noise, missing-node and depth-sparsification cells. |
| `T_p02_block_bootstrap.csv` | S5.4.5 | Paired block-bootstrap results. |
| `T_p02_variance_components.csv` | S5.4.4 | Floor, dispersion and seed variance shares. |
| `T_p02_walkforward.csv` | S4.5 | Rolling-origin folds. |
| `T_p03_config_summary.csv` | S5.4.7 | Winner matrix over the 14 configurations. |
| `T_p05_vicd_stage.csv` | S3.1.10 | VICD, PMI and LASSO admission results on the same candidate set. |
| `T_p05_vicd_phase.csv` | S3.3 | Phase-wise joint statistic for the water-level variable. |
| `T_p06_deep_neg.csv`, `T_p06_nodeep.csv` | S5.7 | Deep-node negatives and the with/without-deep-node comparison. |
| `T_p07_conformal.csv` | S5.5 | Conformal coverage by horizon and nominal level. |
| `T_p19_skill_states.csv` | S6.2, S6.6 | Tan and Liugou pooled-ridge results by state. |
| `T_p19_skill_regression.csv` | S6.5 | $S$–skill regression. |
| `T_p19_digitization_unc.csv` | S6.1 | Digitization calibration and grid uncertainty. |
| `pca_floor_by_k.csv` | §4.2, Table 3 | Floor and ceiling by $K$. |
| `S7_pipeline_register.csv` | S7.1 | Per-pipeline conventions behind each reported RMSE. |

The remaining files are per-seed, per-configuration, per-horizon or per-depth detail. `legacy/` holds superseded V5-protocol comparisons and must not be mixed with the main tables; `logs/` holds the epoch-level convergence records.

## 4. Conventions that must not be mixed

- **Scales.** Cumulative and increment scales are reported side by side; they are not interchangeable. The main tables use the last-step scale at $H=10$.
- **Seed and window inference.** Seed-level paired $t$-tests treat the initialization as the unit; the circular block bootstrap treats the test window as the unit. The block bootstrap over-rejects at the observed effect sizes and is reported separately.
- **Pipelines.** 1.5687 (ablation/comparison) and 1.5700 (stress clean control) are the same configuration through different pipelines; 1.8199 and 1.7475 are the density-separation tier evaluated on two node sets. See `tables/S7_pipeline_register.csv`.
- **Training regimes.** The locked protocol uses four gradient steps per epoch; the standard protocol uses 32. Values from the two regimes are not interchangeable.
- **Skill averaging.** Main columns average the last-step RMSE over seeds and then apply the skill formula; the per-seed order gives −0.3119 for the same cell.

## 5. Reproducing a number

1. Locate the claim in the main text or SI; the SI subsection names the table or sample size.
2. Open the corresponding file in `tables/` or `data/`; column names are lower-case with units where relevant.
3. Check the pipeline in `tables/S7_pipeline_register.csv` before comparing across experiments.
4. Use `MANIFEST.csv` to confirm whether a file is distributed, superseded or held in the project source archive.

## 6. Data that are not distributed

Raw inclinometer, settlement, water-level and anchor-force records are project data and are not included. A small number of SI pointers refer to items held in the project source archive (station coordinates, the 734-row IQR screening table, the 55,380-row quality detail, the realized seed list, the 29-project table and a few pairwise tables). These are marked `source-archive` in `MANIFEST.csv`; the manuscript does not depend on any single one of them for a conclusion.

## 7. Running with your own data

Machine-specific paths are neutralized to four environment variables resolved in `_env.py`:

| Variable | Meaning | Default |
|---|---|---|
| `DATA_ROOT` | root of the monitoring data | `<repo>/example_data` |
| `AUX_ROOT` | root of the saved experiment artefacts | `<repo>/example_artifacts` |
| `WORK_ROOT` | repository work root | `<repo>` |
| `PYENV_ROOT` | Python environment prefix (docstring examples only) | empty |

Expected layout under `DATA_ROOT`:

```text
ceshi14/MoXin/data/CX{2,3,8,13,15}.xlsx      # time, point, depth, cumulative displacement
ceshi14/MoXin/data/斜测点坐标.xlsx            # point, X, Y
ceshi14/shuiwei/时间统一+插值/SW{1,4,7}.xlsx  # time, water-level depth, cumulative change
ceshi14/date/raw/地质参数表.xlsx              # thickness, cohesion, friction, modulus, unit weight, Poisson, K0
shuju/沉降/{6,18,31}.xlsx                     # time, cumulative settlement
```

Anchor-force files (`ML*.xlsx`) are read from `WORK_ROOT/ML数据/`.

## 8. Dependencies

See `requirements.txt`: Python 3.10+, PyTorch (CUDA optional), NumPy, SciPy, Pandas, scikit-learn, Matplotlib, seaborn, PyWavelets, openpyxl, LightGBM and thop. All scripts fall back to CPU when CUDA is unavailable.

## 9. License and citation

- Code and scripts: MIT (`LICENSE`).
- Derived data and tables: CC BY 4.0.
- Citation metadata: `CITATION.cff`; cite the article and this package (v1.1.0).

Contact: Xiaodan Sun, College of Aerospace and Civil Engineering, Harbin Engineering University, `sunxiaodan@hrbeu.edu.cn`.
