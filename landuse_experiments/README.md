# Land-use forecasting experiments

Tests of ways to improve the Land Use module's change forecast, starting from an
intern's review (`Intern_Tasks/guinea_land_use_review/FOM40_CEILING_ANALYSIS.pdf`).
The production code in `Land Use-Deforestation/landuse_tool/` is not changed
here. A method moves there only after it passes the acceptance bar below.

## Notebooks, in order

Notebook 00 and the documents are in this folder; the notebooks are grouped in folders by stage.

| # | Notebook | Question | Status |
|---|---|---|---|
| 00 | `00_summary.ipynb` | Summary for review, kept up to date with every result; the comparison with other tools (section 4) | Done for MOLUSCE, CLUE-S, PLUS, FLUS; updated with 09 and 10 |
| 01 | `1_intern_improvement/01_modis_baselines_and_history.ipynb` | Do the intern's baselines and history-feature gains reproduce? | Done: reproduces (14 of 15 rows within tolerance; the unselected 26.49% row does not, in either environment) |
| 02 | `1_intern_improvement/02_satellite_features.ipynb` | Download and check MODIS NDVI/EVI and Landsat predictors | Done: identical coverage and scenes |
| 03 | `1_intern_improvement/03_modis_satellite_models_and_ensemble.ipynb` | Do the satellite models and the 28.06% ensemble reproduce? | Done: reproduces. In the intern's versions the ensemble starts from the intern's member and scores 28.25% (+0.19); ours 28.59% (+0.53) |
| 04 | `1_intern_improvement/04_modis_2020_2024_test.ipynb` | Do the gains hold on 2020→2024, never used for selection? | Done: +3.9 to +5.2 over the original method (land-cover-only rounds) |
| 05 | `2_improvement_and_benchmark/05_esri_data_and_baseline.ipynb` | Reproduce the deployed ESRI baseline (FoM 13.80%) | Done: 13.80% reproduced exactly; **Guinea-only baseline 17.32%** (see below) |
| 06 | `2_improvement_and_benchmark/06_esri_port_experiments.ipynb` | Carry the methods that held up over to ESRI | Done: corrected deployed method 19.1%; the improvement approach +9.5, but it fails the flip-back check (its gain is predicted label flicker) |
| 07 | `2_improvement_and_benchmark/07_esri_hybrid_and_seasonal.ipynb` | Hybrid (returns vs new change) and seasonal MODIS features on ESRI | Done: hybrid fails the flip-back check; seasonal features help on 2023→2024, not on 2024→2025, also when standardised within each year |
| 08 | `2_improvement_and_benchmark/08_esri_contagion.ipynb` | Contagion features, and logistic regression from the tool comparison | Done: contagion improves (+1.0 / +0.9); averaged with a logistic regression it is the chosen model, 19.9% on 2024→2025; the 0.5 weight is not critical (0.25–0.75: 19.7–20.1%) |
| 09 | `2_improvement_and_benchmark/09_esri_neural_network.ipynb` | FLUS's idea: does a neural network improve the averaged model? | Done: first copy of FLUS's network not adopted; FLUS varies ±0.14 between runs; FLUS's own maps averaged with the improved method pass the rule (20.5%); **FLUS's network reproduced in code (`flus_ann.py`), averaged with the improved method: 20.3%, passes the rule**; its one-third weight is not critical (1/4–1/2: 20.3–20.4%) |
| 10 | `2_improvement_and_benchmark/10_benchmark_projected_amount.ipynb` | The tool comparison when every tool gets the amount of change projected from 2023→2024 | Done: improved 13.1%, deployed 12.2%, FLUS 10.1%, MOLUSCE LR 3.6%, CLUE-S 1.6%, PLUS 1.1% |
| 11 | `2_improvement_and_benchmark/11_methodology_improvement.ipynb` | Every improvement step in order, where each idea came from (the intern's review, the literature, MOLUSCE, FLUS) and what the rule decided | Done; back-test on 2022→2023: improved 15.9% against 15.2% for the deployed method (+0.8, flip-back check passed) |
| 12 | `3_predictors/12_predictors.ipynb` | The twelve predictors: sources, literature support, drop-one test, Guinea-specific candidates (mining, fire) | Done: NDVI carries most (−4 points without it), then rainfall, biomass, population; all twelve kept; mining and fire not chosen; 7 labels in methodology §6.2 to correct |

Notebooks 01–04 run on the MODIS 500 m maps at 5-year steps, as the intern
did. **Their FoM values are not comparable with the deployed module's 13.80%**
(ESRI 10 m, annual, observed amount of change supplied).

Each notebook loads saved runs by default and finishes in seconds. Set a
`RERUN` switch to `True` to recompute that step; the rerun goes to a new
time-stamped folder.

## Intern's Review Findings 

| Idea from the intern's review | Gain over the original-style method | Holds on 2020→2024? | Port to ESRI? |
|---|---:|---|---|
| Land-cover history features (previous class, recent change nearby, multi-scale class mix) | +2.1 | Yes (+2.0) | **First** |
| Seasonal MODIS NDVI/EVI with neighbourhood context | +1.5 over the same-protocol control | Not tested on 2024 | Second |
| Averaging probabilities across models | +0.7 over the best single model | Not tested on 2024 | Third |
| Decision thresholds tuned on validation | part of +5.9 | Loses about a quarter of its gain | Not needed: the deployed module is given the amount of change |
| Dry-season Landsat alone | none over its control | — | No |
| Fire history, regional caps, removing isolated predictions | negative in the intern's review | — | No |
| Neighbour-majority rule (no model) | +2.7 (2015→2020), 0.0 (2020→2024) | No | Run as a baseline check |

## ESRI baseline (notebook 05)

| | FoM, Expander-only |
|---|---:|
| Published in the methodology document (§7.1) | 13.80% (reproduced exactly) |
| **Same method on Guinea only** (re-allocated with the deployed functions) | **17.32%** |

Two corrections to the published validation, both found in notebook 05:

1. **Scored area.** The published score covers Guinea's bounding box: 7.06 M pixels,
   43% of them in neighbouring countries. The boundary mask described in §4.3 was not
   applied when scoring.
2. **Years.** The hold-out labelled "2023→2024" is really **2024→2025**. The run
   fetched every ESRI year to 2025 and withheld the latest. Training never saw 2025,
   so the result is still a valid one-year test, but its labels are off by one year.

All ESRI experiments use a 250 m grid, Guinea-only pixels, and the observed amount of
each transition (as the deployed validation does).

## ESRI experiments (notebooks 06–08)

Test year 2024→2025, Guinea only, observed quantity of change, deployed allocator.
Settings are chosen on 2023→2024.

**Score: the Figure of Merit over all change**, as in every published comparison of
land-change models (Pontius et al. 2008), with random placement of the same amount of each
transition as the floor (`esri_methods.random_like`; Pontius et al. 2004).

**Plus one check, for the methodology's appendix:** on ESRI, 82% of one year's change puts a
pixel back into a class it held before, mostly Trees ↔ Rangeland flicker (25–42% of each year's
changes revert the next year; tree canopy cannot regrow in one year). So an addition must not
find less genuinely new change than the deployed method: change into a class not held in any
ESRI map since 2017, the whole record (`esri_methods.change_kind_scores`; returns are the
"alternation" of Pontius et al. 2017).

| 2024→2025 (test) | FoM | × random | Flip-back check |
|---|---:|---:|---|
| Deployed method, corrected inputs (mean of 3 seeds; ±0.02) | 19.07% | 2.2 | reference |
| Rule: neighbourhood share only | 18.14% | 2.1 | |
| Deployed method + history features (improvement approach) | 28.07% | 3.2 | fails |
| History models, averaged (improvement approach) | 28.60% | 3.3 | fails |
| Hybrid, split from the previous interval | 26.05% | 3.0 | fails |
| Deployed + seasonal MODIS features (seed 42) | 18.42% | 2.1 | passes; no gain on the test year |
| Deployed + seasonal, standardised within each year (seed 42) | 18.41% | 2.1 | passes; no gain on the test year |
| Deployed + contagion features (mean of 3 seeds) | 19.93% | 2.3 | passes |
| Logistic regression | 19.36% | 2.2 | passes |
| **Deployed + contagion, averaged with logistic regression** (mean of 3 seeds; chosen) | **19.87%** | **2.3** | passes |
| **Improved method + FLUS's network reproduced in code** (mean of 3 seeds; notebook 09; adopted 24 October) | **20.33%** | **2.3** | passes |
| Random placement | 8.71% | 1.0 | |

Full tables: `outputs/summary/06_summary.csv`, `07_results.csv`, `08_models.csv`.

- **The improvement approach does not carry over.** It gains 9.5 points, but 98% of its correct
  predictions are pixels going back to an earlier class.
- **The deployed method's weakness is regional** (notebook 06): it beats random placement at
  250 m and 1 km, not from 2 km on. Contagion features (recent new change nearby; Rosa et al.
  2013) narrow that gap.
- **Seasonal features** (`esri_seasonal.py`: MOD13Q1 250 m, 12 months per year, 2018–2024, about
  1 hour download, no account, 790 MB) help on 2023→2024 but not on 2024→2025, probably because
  NDVI levels shift from year to year (notebook 07 section 5). Standardising them within each
  year does not help (18.41%).
- **Recommendation:** deployed method with corrected inputs + contagion features, averaged with a
  logistic regression (no new data); since 24 October also with the copy of FLUS's network
  (notebook 09: 20.33% on 2024→2025). Chosen on 2023→2024 (18.08% against 15.58%), +0.80 on
  2024→2025. Other averaging weights change the 2024→2025 score by 0.22 points at most (notebooks
  08–09). Back-test on 2022→2023: +0.8 over the deployed method (notebook 11, section 5). Confirm
  on 2025→2026 when ESRI publishes 2026.

Issues in the deployed run's inputs (none inflated the document's score): 7 of 12 predictor
labels in document §6.2 are attached to the wrong layer; NDVI and precipitation were
probably taken from 2026 and soil moisture from 2025; training included neighbouring
countries with all predictors set to 0.

## Comparison with other tools

Results and what they show: notebook 00, section 4, and notebook 10. How the comparison was done
(data, each tool's settings, checks, limits, what can be said to clients):
`2_improvement_and_benchmark/BENCHMARK_DETAILS.md`. `2_improvement_and_benchmark/BENCHMARK_GUIDE.md` explains how to run MOLUSCE
(QGIS), PLUS and FLUS on the same inputs (`python scripts/benchmark_export.py` writes them to
`outputs/benchmark/`). CLUE-S was run with the R package lulcc (`scripts/benchmark_lulcc.R`; open it in
RStudio and click Source to rerun; its log is `outputs/benchmark/output_lulcc/run_log.txt`), and
checked against the authors' original program (`clue_s.export_original()`) and a Python version of
the original code (`clue_s.py`). Notebook 09 scores every forecast with the
validation set the literature recommends (`benchmark_scoring.py`): Figure of Merit, producer's
and user's accuracy (Pontius et al. 2008), quantity and allocation disagreement (Pontius &
Millones 2011), random placement (Pontius et al. 2004), Figure of Merit at 1 and 5 km, and the
AUC of each model's suitability maps (Pontius & Si 2014). Kappa only for compatibility.

| 2024→2025, Guinea, 250 m | No change | MOLUSCE ANN | MOLUSCE LR | CLUE-S | PLUS | FLUS | Deployed | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Changes forecast (507,539 observed) | 0 | 3,043 | 65,794 | 249,520 | 580,219 | 928,829 | 500,920 | 500,920 |
| **Figure of Merit %** | 0.0 | 0.1 | 3.6 | 14.8 | 12.6 | 18.1 | 19.0 | **20.4** |
| Producer's accuracy % | 0.0 | 0.1 | 4.0 | 19.1 | 23.9 | **43.1** | 31.6 | 33.5 |
| User's accuracy % | – | 11.4 | 30.5 | **38.9** | 20.9 | 23.6 | 32.0 | 33.9 |
| Quantity / allocation disagreement % | 6.20 / 6.46 | 6.24 / 6.47 | 6.20 / 7.04 | 0.00 / 13.98 | 0.19 / 20.73 | 0.00 / 24.68 | 0.08 / 16.92 | 0.08 / 16.47 |
| FoM % at 1 km / 5 km | 0.0 / 0.0 | 0.2 / 0.4 | 5.6 / 10.0 | 18.6 / 25.5 | 22.7 / **35.5** | **25.2** / 35.1 | 24.4 / 34.8 | **25.2** / 35.3 |
| Kappa (compatibility only) | 0.760 | 0.759 | 0.749 | 0.729 | 0.594 | 0.521 | 0.670 | 0.679 |
| FoM % with the same amount of change | – | 10.3 | 19.8 | 18.3 | 11.5 | 20.2 | 19.0 | **20.4** |
| Suitability AUC | – | 0.500 | 0.727 | 0.705 | 0.575 | 0.738 | 0.726 | **0.748** |

The improved method is the version with the copy of FLUS's network (notebook 09, section 4). As
the tools run it has the highest Figure of Merit; given the same amount of change it is slightly
ahead of FLUS (20.4% against 20.2%; 20.3% against 20.0% over three runs each); FLUS is level at
1 km and PLUS slightly ahead at 5 km. Notebook 00 also shows the table without the deployed method. MOLUSCE ANN's
potential maps are 0 on more than 99% of pixels, so its same-amount score reflects tie-breaking.
Kappa and overall accuracy favour forecasting nothing (Pontius & Millones 2011).

## Folders

```
landuse_experiments/
├── 00_summary.ipynb             summary for review (start here)
├── README.md
├── 1_intern_improvement/        notebooks 01–04: the intern's review reproduced and tested (MODIS)
│   └── local_landuse/           the intern's code (see local_landuse/PROVENANCE.md)
├── 2_improvement_and_benchmark/ notebooks 05–11: the deployed method on ESRI, its improvement,
│                                the comparison with other tools, the steps in order
│   ├── BENCHMARK_DETAILS.md     how the tool comparison was done: data, settings, checks, limits
│   └── BENCHMARK_GUIDE.md       how to run MOLUSCE, PLUS and FLUS by hand on the same inputs
├── 3_predictors/                notebook 12: the predictors, their sources, and what each adds
├── scripts/                     code shared by the notebooks
│   ├── nb_helpers.py            run steps, load results, charts
│   ├── esri_methods.py          the deployed method rebuilt for the ESRI experiments (one switch per change)
│   ├── esri_seasonal.py         MODIS seasonal download and features for the ESRI grid (notebook 07)
│   ├── flus_ann.py              FLUS's neural network reproduced (notebook 09)
│   ├── benchmark_export.py      inputs for testing other tools (see BENCHMARK_GUIDE.md)
│   ├── benchmark_scoring.py     the literature's validation metrics for the tool comparison (notebook 00)
│   ├── clue_s.py                CLUE-S: Python version of Dyna-CLUE 2.7, and input/output for the original program
│   ├── benchmark_lulcc.R        CLUE-S with the R package lulcc (the CLUE-S results)
│   └── benchmark_fom_check.R    our Figure of Merit recomputed with lulcc (check)
├── data/                        inputs: land-cover maps, terrain, satellite downloads   (not in git)
├── outputs/                     full runs, maps, logs                                   (not in git)
│   └── summary/                 the tables each notebook saves                          (in git)
└── requirements.txt
```

Each notebook's first line adds `scripts/` to Python's path, so it runs from its own folder.

`data/` and `outputs/` are regenerated by the notebooks. Land-cover and terrain
rasters come from `Land Use-Deforestation/data/`; satellite downloads come from
Microsoft Planetary Computer (notebook 02, about 1 hour, no account needed).

## Environment

A separate venv, not the repo-root uv project (that one belongs to `studies/`).

The two R scripts need R (tested with 4.6.1) and `install.packages(c("lulcc", "gsubfn"))`. On
this machine R is in `D:/envs/R` and its packages in `D:/envs/R-library` (set `R_LIBS_USER`).

```bash
cd landuse_experiments
python -m venv .venv                                  # Python 3.10 tested
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name guinea-landuse-exp \
    --display-name "Python (guinea landuse experiments)"
```

**Windows with PostgreSQL/PostGIS:** PostGIS sets `PROJ_LIB` and `GDAL_DATA`
machine-wide, which breaks rasterio ("Cannot find proj.db"). `nb_helpers.py`
clears them for the notebooks. For scripts and tests, add the same lines to the
venv:

```bash
cat > .venv/Lib/site-packages/sitecustomize.py <<'EOF'
import os
for _k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_k, None)
EOF
```

Check: `cd 1_intern_improvement && ../.venv/Scripts/python -m pytest local_landuse/tests -q` should print 74 passed.

**Reproduction tolerance.** A rerun counts as reproducing the intern's figure when it
is within **±0.4 FoM points** (RF, fixed-demand and rule rows) or **±0.85** (XGBoost rows
with validation-tuned thresholds). Both numbers are measured, not chosen: notebook 01 §4
reruns the same code in this environment and in the intern's pinned one, and versions
alone moved those kinds of rows by at most 0.33 and 0.85 points. The constants are
`nb_helpers.TOL` and `nb_helpers.TOL_XGB_THRESHOLD`.

**Optional: the intern's exact environment** (Python 3.14 and the pins in
`1_intern_improvement/local_landuse/requirements-intern.txt`), used only for that check. It lives outside the
repo and is not a notebook kernel; the comparison runs were made with it from this folder.

```bash
py -3.10 -m venv D:/envs/uv-tool && D:/envs/uv-tool/Scripts/python -m pip install uv
export UV_CACHE_DIR=D:/envs/uv-cache UV_PYTHON_INSTALL_DIR=D:/envs/uv-python
D:/envs/uv-tool/Scripts/uv python install 3.14
D:/envs/uv-tool/Scripts/uv venv D:/envs/guinea-intern-py314 --python 3.14
D:/envs/uv-tool/Scripts/uv pip install --python D:/envs/guinea-intern-py314/Scripts/python.exe     -r 1_intern_improvement/local_landuse/requirements-intern.txt
# add the same sitecustomize.py as above to D:/envs/guinea-intern-py314/Lib/site-packages/
PYTHONPATH=1_intern_improvement D:/envs/guinea-intern-py314/Scripts/python -m local_landuse.advanced --output-dir outputs/advanced_intern_env_py314
```

## Acceptance bar for moving a method into `landuse_tool`

A change is accepted only when:

- its settings are chosen on one period (2023→2024) and it is then tested once on a later
  one (2024→2025), with the same data and the same score (Pontius et al. 2004);
- on the test period its Figure of Merit beats the deployed method's by more than the
  spread across random seeds (Bouthillier et al. 2021), and it beats random placement of
  the same amount of change (Pontius et al. 2004);
- it passes the flip-back check: its new-change FoM is not lower than the deployed method's
  (notebook 06 section 7);
- its cost is stated: runtime, downloads, Earth Engine quota or licensed imagery.

## Open items

| Item | Status |
|---|---|
| Reproduce and record the deployed ESRI baseline | Done (notebook 05). Guinea-only, corrected inputs: 19.1% (notebook 06) |
| Reproduce the intern's MODIS results and test them on 2020→2024 | Done (notebooks 01–04) |
| Port history features, seasonal vegetation and probability averaging to ESRI | Done (notebooks 06–07): none accepted |
| Contagion features; logistic regression from the tool comparison | Done (notebook 08): contagion averaged with logistic regression chosen |
| Weight sensitivity of the averages | Done (notebooks 08–09): 0.22 points at most on 2024→2025 |
| Back-test on 2022→2023 | Done (notebook 11, section 5): +0.8 over the deployed method, flip-back check passed |
| Re-test the chosen model on 2025→2026 | Waiting for ESRI 2026 |
| Compare with MOLUSCE, CLUE-S, PLUS and FLUS on the same data | Done (notebook 00) |
| Test FLUS's idea (neural network per class) by the same rule | Done (notebook 09): FLUS's own maps pass the rule; reproduced in code, its network averaged with the improved method passes too (20.3%). Added to the improved method (24 October); comparison rerun |
| FLUS run-to-run variation (two more runs) | Done (notebook 09): ±0.12 as it runs, ±0.14 with the same amount |
| Comparison with projected amounts of change | Done (notebook 10) |

## References

- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems* 3.
- Pontius, R.G. Jr., Huffaker, D., Denman, K. (2004). Useful techniques of validation for spatially explicit land-change models. *Ecological Modelling* 179(4), 445–461.
- Pontius, R.G. Jr., Boersma, W., Castella, J.-C., et al. (2008). Comparing the input, output, and validation maps for several models of land change. *Annals of Regional Science* 42(1), 11–37.
- Pontius, R.G. Jr., Millones, M. (2011). Death to Kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. *International Journal of Remote Sensing* 32(15), 4407–4429.
- Pontius, R.G. Jr., Si, K. (2014). The total operating characteristic to measure diagnostic ability for multiple thresholds. *International Journal of Geographical Information Science* 28(3), 570–583.
- Pontius, R.G. Jr., Krithivasan, R., Sauls, L., Yan, Y., Zhang, Y. (2017). Methods to summarize change among land categories across time intervals. *Journal of Land Use Science* 12(4), 218–230.
- Rosa, I.M.D., Purves, D., Souza, C. Jr., Ewers, R.M. (2013). Predictive modelling of contagious deforestation in the Brazilian Amazon. *PLoS ONE* 8(10), e77231.
- Verburg, P.H., Overmars, K.P. (2009). Combining top-down and bottom-up dynamics in land use modeling: exploring the future of abandoned farmlands in Europe with the Dyna-CLUE model. *Landscape Ecology* 24(9), 1167–1181.
