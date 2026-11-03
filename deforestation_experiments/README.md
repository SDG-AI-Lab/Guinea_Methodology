# Deforestation risk experiments

Tests of ways to improve the Deforestation module's calculation on the platform's 100 m grid, against what the platform computes now (`Guinea_Project`, main) and the methodology document (*Methodology_Deforestation_Guinea_v2*), and against other tools. They start from the predictor assessment in `studies/forest_loss_drivers.ipynb` and the dataset checks in `studies/dataset_checks/`. The production code in `Land Use-Deforestation/landuse_tool/` is not changed here: the deployed functions are imported and called unchanged.

An earlier trial of these methods ran on a 250 m grid; it is superseded by this work and nothing here depends on its results.

## What is modelled

**Forest loss** (Hansen, canopy ≥ 30 % in 2000, loss from 2013): how much there is, where it happens and where it will happen next.

Most of Hansen's loss in Guinea is a fallow cycle that grows back, so it is called forest loss rather than deforestation. About half of the 2013–2016 loss is trees again a year later, and nearly all of it after 3–4 years (`studies/dataset_checks/04_loss_after_clearing`). Hansen does not record the regrowth: its gain layer stops in 2012. Hansen is still the best record for Guinea:

* **TMF (JRC)** covers only the moist forest, 3.4 % of Guinea.
* **Esri's yearly land cover** changes 10–15 % of Guinea a year, and a third of that flips back (`01_desk_check`).
* **The 30 % canopy threshold** is the team's forest definition, and the best match with WorldCover's trees in the forest ecoregions, which hold 97 % of the loss (best 25–30 %; `03_canopy_threshold`).

So the models predict where clearing happens, temporary or not. The permanent part cannot be measured yet.

## How a change is judged

Each model is calibrated on the loss of one window, with every predictor measured at the window's start, and tested on the loss of the next window, over the forest standing at its start. Windows: 2014–2016 → 2017–2019, 2017–2019 → 2020–2022, 2019–2021 → 2022–2024.

* **AUC** of the risk map on the test window, over the 30 m forest pixels (each 100 m cell counts with the hectares it lost and kept).
* **Verra VT0007** (v1.0, 2024): the median error (MedAE, ha) between predicted and observed loss in blocks of about 100,000 ha, on the calibration window (fit test) and the next window (prediction test). Its benchmark is distance from non-forest; a model qualifies only if it beats the benchmark on both tests.

A change is kept only if it improves every window by more than the run-to-run noise.

## Notebooks and findings

| # | Notebook | Question | Finding |
|---|---|---|---|
| | **`1_areas/`** | **How the platform counts and reports forest and loss** | |
| 01 | `01_grid_and_shares` | Which cells make the platform's 100 m grid, and which way of putting Hansen's 30 m pixels on it gives the right areas | One value per cell, as the methodology document's notebook counts, gives 39–56 % more loss each year and every hexagon 1.3–1.9 times its loss. Per-cell shares give the 30 m area at any cell size and every hexagon its loss (median ratio 1.00). Calculations, the forecast included, use the shares; hexagons only report them. |
| 02 | `02_reporting_areas` | Do the shares give each region, prefecture, ecoregion and hexagon its 30 m area | Within 0.01 % per region, prefecture and ecoregion, about 1 % per hexagon. Four products still count the half-cell masks (2024 forest 2.5 % low); border cells leave the regional tables 0.4 % short. |
| | **`2_risk_model/`** | **The deployed risk model, its problems, the improved model** | |
| 03 | `03_deployed_baseline` | The deployed model on the 100 m grid against the methodology document and Verra's benchmark | It qualifies under VT0007, which the methodology document cites but does not report. Its half-cell masks leave 10–20 % of the next loss without a risk value, and its hold-out reads yearly predictors after the loss (+0.01 AUC too high). |
| 04 | `04_training_cells` | Which 100 m cells the model should learn from | Training on the shares (cells drawn in proportion to the forest lost and kept) ranks the next loss better in every window and gives every forest cell a risk. A rule's AUC over its own cells misleads; score over the 30 m pixels. |
| 05 | `05_predictors_and_drivers` | What each predictor carries, the drivers behind them, drivers not yet in the model | The loss history carries the model (shifting cultivation, 82 % of the loss); terrain, access and climate add to it as groups. 16 predictors score slightly above 12 in every window. Cropland, regrown forest and crop suitability add nothing. |
| 06 | `06_one_national_model` | One national model or one per ecoregion | Per-ecoregion models fit better but predict worse in every window (216 against 180 ha): one national model. |
| 07 | `07_improved_vs_deployed` | The improved model against the deployed one, and where the platform's projection puts the loss | Better on every score (table below). The projection's highest-risk-first rule misplaces the loss (788–931 ha per block); Verra's allocation gives 122–235 ha. |
| | **`3_benchmark/`** | **The improved model against other tools** | |
| 08 | `08_tools_comparison` | riskmapjnr, forestatrisk, weights of evidence (Dinamica EGO), on the same data, windows and scores | The rank average of the improved model and riskmapjnr leads, then the improved model (table below). |
| 09 | `09_ideas_from_the_tools` | Can the improved model take what the tools and the platform do differently | No: equal-area classes, other window sizes, coordinates and the palm-oil price do not improve the prediction test in every window. |
| 10 | `10_calibration_timing` | How Verra's calibration map is built (option A or B) | Option A, the procedure of VT0007 v1.0 and its software, is adopted (section below). |

## Results

**The improved model against the deployed one** (mean of the three windows, `outputs/summary/07_improved_vs_deployed_mean.csv`):

| Model | AUC (30 m pixels) | Fit MedAE (ha) | Prediction MedAE (ha) |
|---|---|---|---|
| VT0007 benchmark (distance from non-forest) | 0.668 | 353 | 373 |
| Deployed model, half-cell masks (platform) | 0.676 | 157 | 236 |
| Improved model, half-cell masks | 0.688 | 141 | 202 |
| Improved model, trained on the shares | 0.753 | 118 | 186 |

Training on the shares gives most of the AUC gain, because the risk map then covers every forest cell. The predictors give most of the gain in Verra's prediction error. The methodology document reports AUC only (0.564, 0.660 and 0.689 on its three hold-out splits, over cells with one value per cell), which is not comparable with these.

**Against other tools**, every model with the same timing (option A; `outputs/summary/08_tools_mean.csv`):

| Model | AUC | Fit MedAE (ha) | Prediction MedAE (ha) |
|---|---|---|---|
| Rank average: improved model + riskmapjnr | 0.759 | 136 | 173 |
| Improved model | 0.752 | 128 | 180 |
| riskmapjnr (moving window) | 0.750 | 164 | 193 |
| Deployed model (platform) | 0.676 | 157 | 236 |
| forestatrisk (iCAR) | 0.745 | 134 | 245 |
| Weights of evidence (Dinamica EGO's method) | 0.725 | 231 | 277 |
| VT0007 benchmark | 0.668 | 353 | 373 |

Every model beats the benchmark on both tests in every window. No model is best in every window, and Verra's errors move by 10–30 ha with the draw of training cells alone, which is why the improved model's row differs slightly from the table above. riskmapjnr as designed (option B): AUC 0.750, fit 22 ha, prediction 142 ha.

## How the calibration map is built: option A (`10_calibration_timing`)

Verra's allocation gives each risk class the loss rate it had in the calibration window. The map those rates are learned on can be built in two ways. The choice changes which model has the lowest Verra errors; it does not change any model's AUC, nor the areas.

| Option | Calibration map | Supported by | Lowest errors | Prediction / fit MedAE (ha) |
|---|---|---|---|---|
| **A. Same timing for every model (adopted)** | Made from what was known at the start of the calibration window | VT0007 v1.0 (§5.1, §5.3.1.1); UDef-ARP, the software written with Verra for it ("covariates appropriate to T1", updated for T2) | Rank average of the improved model and riskmapjnr's local loss rate | 173 / 136 (improved model alone 180 / 128; riskmapjnr 193 / 164) |
| B. riskmapjnr as designed | The calibration window's own loss | Verra's 2021 draft tool, which VT0007 v1.0 replaced; riskmapjnr and its QGIS plugin, deforisk | riskmapjnr | 142 / 22 |

**Why A:**

1. **It is the procedure of the tool in force.** VT0007 v1.0 and UDef-ARP build every model's fitting map from data at the start of the calibration window, then update it for the prediction.
2. **B comes from a superseded draft.** Verra's 2021 draft tool, which carried the same number, had no fit test, so the timing of the map did not matter there.
3. **Under B, v1.0's fit test is close to circular.** riskmapjnr's map is the local loss rate of the calibration window, and the fit test checks it against the loss of that same window. Every cell that lost forest has a high local rate by construction, so the map cannot miss: its fit error drops from 164 to 22 ha per block. The prediction test stays fair (the next window's loss is unseen), but the rates it carries forward were learned on the answer.

**Other scores** (mean absolute error, the 2021 draft's wRMSE, R², share of the loss misplaced; blocks of ~25,000, ~100,000 and ~400,000 ha; 45 comparisons): under A the rank average wins 24, riskmapjnr 19 and the improved model alone 2. The rank average misplaces the least of the loss (8.5 %, against 9.0 % for riskmapjnr and 9.9 % for the improved model). With B allowed, riskmapjnr as designed wins 28, but in 2022–2024 it keeps only Verra's score at ~100,000 ha. No version of the improved model beats it under B (closest 158 ha).

**For the platform:** the rank average is the risk map. riskmapjnr as designed is reported beside it as a sensitivity, since VT0007 v1.0 does not forbid B in so many words and deforisk scores B-built maps with v1.0's tests.

## Decisions for the team

What the results mean for the platform's deforestation module. None of it is on the platform yet.

| Decision | Evidence | To implement on the platform |
|---|---|---|
| Train on the 30 m shares: draw training cells in proportion to their lost and kept forest (ha), and give every forest cell a risk value | `04_training_cells`, `07_improved_vs_deployed` | Replace the half-cell masks (`HansenShares.display_arrays`) in training and scoring |
| Score the validation AUC over the 30 m forest pixels (each cell weighted by the hectares it lost and kept), not over the training rule's own cells | `04_training_cells`, section 2 | Validation AUC |
| Train on the last 3-year window with every predictor read at its start, and map with the predictors at the latest year, as VT0007 v1.0 does (§5.1). The platform now trains on all analysis years with the yearly predictors at their latest year, after the loss they learn from; in a hold-out this overstated the deployed model by about 0.01 AUC | `03_deployed_baseline` | Two predictor stacks in `pipeline.run_from_arrays`: T1 for training, the latest year for the map |
| Use the improved model's 16 predictors (they rank the next loss slightly better than 12 in every window). Do not add cropland, regrown forest or crop suitability: tested, no gain | `05_predictors_and_drivers`, `07_improved_vs_deployed` | Predictor list (`scripts/improved_model.json`) |
| One national model, not one per ecoregion | `06_one_national_model` | – (as now) |
| Keep Verra's 30 geometric classes, not equal-area classes | `09_ideas_from_the_tools` | – |
| Use the rank average of the improved model and riskmapjnr's 5 km local loss rate as the risk map: each forest cell's mean percentile rank of the two (option A). Its values are percentile ranks among Guinea's forest (0–1), not probabilities; every Verra test and AUC here was run on these values | `08_tools_comparison`, `10_calibration_timing` | Risk map; riskmapjnr's local rate from the same shares (`scripts/tool_comparison.py`, `rmj_local_rate`) |
| Add Verra's fit and prediction tests to the platform's validation, with forest and loss in hectares from the shares and option A's calibration map; report riskmapjnr as designed beside it | `03_deployed_baseline`, `10_calibration_timing` | Validation |
| Count forest from the shares, not the half-cell masks, in the forest-cover layer and its hexagons, the forest under the hexagon risk, the prefecture risk summary and the baseline projection (the masks put the 2024 forest 2.5 % low, 3–12 % off per unit) | `02_reporting_areas` | `forest_by_year` and `forest_now` in those products; the zonal tables already use the shares |
| Give the border cells outside every ADM1/ADM2 polygon to the nearest unit, so the regional tables add up to the national figure (now 0.4 % short; tested: exact, shares still within 0.01 % of the 30 m figures) | `02_reporting_areas` | Zone rasterization in `stats.zonal_deforestation` |
| Allocate the baseline projection's loss with Verra's class rates (each class's calibration-window loss rate times each cell's forest, scaled to the projected amount), not to the highest-risk cells first. Highest risk first misplaces it: 788–931 ha per block against 122–235 ha, worse than Verra's benchmark | `07_improved_vs_deployed`, section 4 | `scenario.allocate_future_loss` |
| Drop the palm-oil price: it adds nothing to the prediction test. If kept, read it at the start of each window in Verra's tests, not at a fixed 2018 | `09_ideas_from_the_tools` | `reference_year` of the WFP source |

**Recommended for later, as data rather than predictors** (`05_predictors_and_drivers`): a cropland map that includes shifting cultivation's fallows, and a reliable forest map every year to tell regrowth from standing forest.

**Alignment with the platform:** the platform's products and the half-cell masks were read from `Guinea_Project` main (26–28 October 2026). The risk model here calls the deployed functions from a local copy of `landuse_tool`; main's `drivers.py` (055bf72) gives the same model on the same samples (predictions within 3 × 10⁻¹⁶, same cross-validation and importances); it differs only in memory handling. The platform scripts are to be written against main.

## Server cost

The platform runs every raster analysis on one worker, one task at a time, with the memory budget `WORKER_HEAVY_MEMORY_LIMIT` (set on the server; 4 GB by default in `docker-compose.prod.yml`); its memory guard stops a job at 80 % of it. The deployed 100 m forest run peaks at about 9 GB (`DEPLOYMENT.md`). The server's size is not in the repo, so no cost in money is given.

Each change, measured alone on one window and the full 100 m grid (`scripts/server_cost.py`; experiment code, which keeps values over the forest cells only, so compare with the deployed model's row, not with the platform's 9 GB):

| Step | Peak memory (GB) | Run time (min) |
|---|---|---|
| Deployed model: predictors, train, score | 3.7 | 1.2 |
| Improved model: predictors, train on the shares, score every forest cell | 3.9 | 1.7 |
| riskmapjnr's local loss rate and the rank average | 3.1 | 0.2 |
| Verra's fit and prediction tests | 3.3 | 0.5 |
| Areas per prefecture from the shares, border cells to the nearest unit | 2.0 | 0.2 |

* **The improved model adds about 5 % memory and 45 % time to the model step.** It scores every forest cell, and builds two loss-history predictors (a 5 km moving window and a distance transform). The platform's run is already close to its budget; if its code scales the same way, the change adds about 0.5 GB. Building the predictors over the forest cells only and scoring in chunks keeps it down.
* **The other steps need less memory than the model step,** so they do not raise the peak if the model's arrays are freed first. The rank average's moving window holds two whole-grid float64 arrays (0.4 GB each; float32 halves that). The nearest-unit grid depends only on the boundaries and can be cached.
* **No cost:** counting forest from the shares (the same layer), and reading the hold-out's predictors at the window's start.
* Run times are from a 6-core machine; the random forest uses every core the worker has (`n_jobs=-1`).

## This folder

| Path | What |
|---|---|
| `1_areas/`, `2_risk_model/`, `3_benchmark/` | The notebooks, in order |
| `PLATFORM_UPDATE.md` | For the platform's developers: what changes in `Guinea_Project`, where, why, and how to check it |
| `CLAUDE_GUIDE.md` | Hand-over for a new Claude Code chat that implements `PLATFORM_UPDATE.md` in `Guinea_Project` |
| `scripts/` | The code the notebooks call. Areas: `area_methods.py`, `reporting.py`, `platform_checks.py`. Model: `predictors.py`, `model.py` (training and scoring on the 100 m grid; the improved model's predictors in `improved_model.json`), `verra.py` (VT0007 tests), `holdout_timing.py`, `model_variants.py`, `lean_and_ecoregions.py`. Tools: `tool_comparison.py`, `improve.py`, `calibration_timing.py`, `tools_forestatrisk.py`, `tools_riskmapjnr_rate.py`. New drivers: `candidates.py`. Projection: `projection_allocation.py`. Server cost: `server_cost.py`. Shared: the deployed model's settings and VT0007 classes (`defor_methods.py`), Earth Engine helpers (`defor_data.py`), the tools' environment (`defor_tools.py`) |
| `data/` | Inputs (not committed): predictors from sources coarser than 100 m (rainfall, land temperature, soil moisture, night lights, travel time, burned area), resampled to 100 m when used, and the deforestation notebook's Hansen download (`hansen_100m.tif`); `checks/` small reference tables; `reports/` unit and hexagon tables; `maps/` saved risk maps |
| `outputs/` | `runs/`: each script's saved results, by topic and window (not committed); `summary/`: the tables the notebooks save, prefixed by the notebook's number |
| `.venv_tools/` | GDAL, forestatrisk and riskmapjnr, apart from the main environment (`landuse_experiments/.venv`) |

The 100 m layers (Hansen shares, predictors, Esri land cover, GLAD land cover 2010–2020, GAEZ suitability) are in the shared store `E:\SDG AI Lab\Guinea\data_100m\`, read through `data_100m_tools/grid100.py`. Each notebook adds `../scripts` to Python's path, so it runs from its own folder; saved results load by default, and `RERUN = True` recomputes a step.

## References

* Clark Labs (2024–2025). *UDef-ARP: Unplanned Deforestation Allocated Risk mapping Procedure*. https://github.com/ClarkCGA/UDef-ARP
* Dinerstein, E., Olson, D., Joshi, A., et al. (2017). An ecoregion-based approach to protecting half the terrestrial realm. *BioScience* 67, 534–545.
* Hansen, M.C., Potapov, P.V., Moore, R., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science* 342, 850–853.
* Soares-Filho, B.S., Coutinho Cerqueira, G., Lopes Pennachin, C. (2002). DINAMICA: a stochastic cellular automata model designed to simulate the landscape dynamics in an Amazonian colonization frontier. *Ecological Modelling* 154, 217–235.
* Verra (2021). *VT0007 JNR Risk Mapping Tool, draft v0.1 for public consultation*.
* Verra (2024). *VT0007 Unplanned Deforestation Allocation (UDef-A), v1.0*.
* Vieilledent, G. (2021–2025). *forestatrisk*, *riskmapjnr* and *deforisk*. https://ecology.ghislainv.fr/
