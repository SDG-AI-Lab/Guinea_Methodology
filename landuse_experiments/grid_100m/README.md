# Land-use experiments at 100 m

The platform runs the land-use module at 100 m. This stage tests the land-use calculation and its
improvement on the platform's 100 m grid, against what the platform computes now (`Guinea_Project`,
`main`) and the methodology document (`Docs/Methodology_LandUse_Guinea.docx`), and compares it with
other tools. The first stage, on a 250 m grid, is in `../grid_250m/`; its decisions are re-tested
here, not assumed. Every method choice rests on the scientific literature; where the platform's
backend and the methodology document differ, both are shown and tested.

The platform's code is not changed here: `main`'s `landuse_tool` is copied read-only into
`data/platform_main` (git archive) and called unchanged. The grid and store are shared with the
deforestation work (`data_100m_tools/`).

## How a change is judged

As in the 250 m stage: methods are chosen on validation (trained on 2022 → 2023, forecast
2023 → 2024) and tested once on 2024 → 2025 (trained on 2023 → 2024), with a back-test on
2022 → 2023 (trained on 2021 → 2022). The score is the Figure of Merit over all change (Pontius et al. 2008), with the flip-back
check in notebook 06's appendix; a change must beat the spread across seeds and
random placement. A fourth window, 2020 → 2021 → 2022, was used by no choice and gives the fair
estimate of the final gain (notebooks 05–06; Cawley & Talbot 2010). Platform users choose their own
training years; these windows test the method.

## Notebooks

| # | Notebook | Question | Status |
|---|---|---|---|
| | **`1_data/`** | **The label, the predictors, the areas** | |
| 01 | `01_classes_and_change` | Is the platform's 100 m class a fair label? Do the maps change as much from year to year at 100 m as at 250 m? (`scripts/classes100.py`) | Done: the platform's class is the majority of the 10 m pixels in 94.5 % of cells, 0.4 % of cells are truly mixed; class areas within 0.2 % for the large classes, small ones undercounted by 1–4 % (4–13 % at 250 m); maps change 10–15 % a year on both grids and at 10 m, a quarter to two fifths flips back, so the flip-back check stays; same observed amount in km²; the deployed notebook's 1,000-pixel rule admits 23 transitions at 100 m against 12 at 250 m, the backend trains every transition that occurs; the two maps agree in 86.5 % of cells, so scores are compared within a grid |
| 02 | `02_predictors_and_memory` | The 12 predictors as `main` builds them; time and memory of the platform's own run at 100 m (`scripts/predictors_lu100.py`, `scripts/platform100.py`) | Done: all 12 from local files in about 30 s, no Earth Engine; same-source layers agree with 250 m (Spearman 0.91–1.00), slope and aspect sharper, roads and points of interest differ by OpenStreetMap snapshot; the platform's run takes about 15 min and 3.4 GB per window; holding the 12 predictors together (`assemble_predictor_stack`) takes 5.3 GB, the likeliest cause of the server's out-of-memory stops; `main` runs one Random Forest per transition with no growth modes or neighbourhood, unlike the deployed notebook the 250 m rebuild copied; clouds are kept as a class; the projected demand is about a quarter below the observed change |
| 03 | `03_reporting_areas` | Class areas from the shares against the platform's area tables (map-based areas, the composition endpoint), per region, prefecture, ecoregion and hexagon (`scripts/areas100.py`) | Done: one class per cell is within 0.3 % for trees and rangeland in every unit, but undercounts crops and built area by 1 % (up to −20 % for crops in the south-east forest prefectures) and bare ground by 4 %, and misplaces 2–13 % of each class among H3 hexagons; the platform's published composition adds up to 1.76 times Guinea (summed over the bounding box; only 2024–2025 published); ADM polygons miss border cells (8.8 % of the water), giving them to the nearest unit closes it |
| | **`2_model/`** | **The deployed method, the decisions at 100 m, the improved method** | |
| 04 | `04_deployed_baseline` | `main`'s backend as it runs, against the methodology document; observed and projected amounts; 3 seeds on the three windows; fixes to `main` (1,000-pixel rule, clouds) | Done: the backend scores 12.3–13.7 % as it runs (14.2–17.3 % with the observed amount), the same as `main`'s own validation; the document's method (Expander model with a neighbourhood predictor, 1,000-pixel rule, no clouds) beats it in every window by 0.6–1.4 points, 2–10 times the seed spread, and runs faster (about 11 against 16 min); the growth-mode option is no consistent gain and takes 29 min; clouds and the 1,000-pixel rule change nothing (≤ 0.01) but are literature-supported; the projected amount is 18–27 % too low in every window |
| 05 | `05_decisions_at_100m` | The parts of the document's method one at a time (Expander sample, neighbourhood predictor), with three seeds; the 250 m findings as one-change tests on `main`'s method, with the literature for each: kept (contagion, logistic-regression average, FLUS's network, dated predictors, Guinea-only training) and rejected (seasonal and history features, growth modes and neighbourhood); radius in cells or km; the shares as features and training weights; the 12 predictors | Done: the document's method with three seeds (spread 0.1–0.4); on the validation window the 3 × 3 class shares, a copy of MOLUSCE's logistic regression and their blend all beat it, the blend most (+2.1 points); on test and backtest the blend gains 1.2–1.7 points with both amounts (`scripts/decisions100.py`, `run05.py`). Recent-interval amounts instead of the platform's trend: +2.5 on validation, −1.0 on test, level on backtest, so the trend is kept (`run05_amounts.py`). Second round, contagion and FLUS's network from the 250 m improved method (`run05_second.py`, `run05_fresh.py`): FLUS's network loses on backtest; contagion gains 0.2–0.7 points with every seed in all four windows, but on a fresh 2020 → 2022 window its gain (+0.35) is within the seed spread (0.45), and it takes a run from 14 to 22 min, near the platform's 30-min task limit, so the blend is kept. The parts of the document's method are explained, not run apart (the blend keeps both). Dated predictors and Guinea-only training are already `main`'s behaviour; the radius and the shares as inputs or weights were not run (fine-tuning within the seed noise) |
| 06 | `06_improved_vs_deployed` | The improved method against the deployed one, with the weight checks and the flip-back check | Done: as the platform runs it, 15.1–16.3 % against 12.3–13.7 % in all four windows (+2.4–2.8 points; +2.7–3.7 with the observed amount), 5–16 times the seed spread; on the fresh 2020 window, used by no choice, 13.5 → 16.3 %; flip-back check passed (appendix); equal weights kept (document weight 0.25 within the seed noise, 0.75 worse); about 16 min per run for both, 6–8 GB against 3.7 GB, too close to the server's 9.6 GB memory guard unless the predictors are read in blocks; the platform change in six steps (`scripts/improved100.py`, `run06.py`) |
| | **`3_benchmark/`** | **The improved method against other tools** | |
| 07 | `07_tools_at_100m` | PLUS, CLUE-S, MOLUSCE and FLUS at 100 m with the same data and scores, test window (PLUS and FLUS run by hand, MOLUSCE and CLUE-S from code checked against their programs at 250 m; `scripts/tools100.py`, `run07.py`); the order of the tools at 250 m and 100 m | Done (PLUS and FLUS run by hand; seed means with the spread): our best (the blend of 05) is first with the same amount (20.0 % against 19.6 % for MOLUSCE LR, 18.7 % for FLUS) and as each runs with the observed amount (20.0 % against 17.9 % for FLUS, 17.3 % for the backend); with the platform's projected amount FLUS scores higher (16.5 % against 15.3 %) by changing 2.2 times as many cells, and is behind relative to random (2.0 against 2.5 times); the tools keep their 250 m order except FLUS; the export's 17 empty slope and aspect cells froze MOLUSCE (fixed) |

## Results in brief

**The improved method against what the platform runs now** (notebook 06). Improved: the
document's Expander models with the neighbourhood predictor, averaged as ranks with a
MOLUSCE-style logistic regression (notebook 05). Deployed: `main`'s backend. Figure of Merit %,
mean of three seeds:

| Window | Deployed, as it runs | Improved, as it runs | Gain | Deployed, observed amount | Improved, observed amount | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Backtest 2021 → 2022 → 2023 | 13.73 | 16.10 | +2.37 | 14.18 | 16.85 | +2.68 |
| Validation 2022 → 2023 → 2024 (used to choose) | 12.33 | 15.08 | +2.74 | 14.53 | 18.01 | +3.48 |
| Test 2023 → 2024 → 2025 | 12.73 | 15.26 | +2.53 | 17.31 | 20.01 | +2.71 |
| Fresh 2020 → 2021 → 2022 (used by no choice) | 13.51 | 16.29 | +2.77 | 16.04 | 19.76 | +3.72 |

"As it runs" is the platform's projected amount of change, as users get it; "observed amount"
compares only where change is placed. Every gain is 5–16 times the larger seed spread (at most
0.65), and the improved method is 2.3–3.7 times random placement.

**Against established tools**, test window, same data and scorer (notebook 07):

| Method | As it runs, observed amount | As it runs, projected amount | Same amount |
|---|---:|---:|---:|
| Improved method (code, 3 seeds) | **20.01** | 15.26 | **20.01** |
| Methodology document's method (code, 3 seeds) | 18.60 | 14.08 | 18.60 |
| Deployed backend, `main` (code, 3 seeds) | 17.31 | 12.73 | 17.31 |
| FLUS 2.3 (its program) | 17.91 | **16.52** | 18.72 |
| MOLUSCE LR (its code, 3 seeds) | 4.35 | 4.35 (its own amount) | 19.55 |
| CLUE-S (port of Dyna-CLUE 2.7) | 13.81 | 5.31 | 17.26 |
| PLUS (its program) | 13.16 | 4.75 | 10.83 |
| MOLUSCE ANN (its code) | 0.38 | 0.38 (its own amount) | 7.26 |
| Random placement | | | 8.58 |

FLUS leads with the projected amount only because it changes 2.2 times as many cells as the
platform projects; given exactly FLUS's changes, the improved method places them 2.4–3.5 points
better. For presenting the platform against other tools, the deployed and document rows can be
left out.

**Limits**

* 2025 → 2026 is the independent test, once Esri publishes its 2026 map.
* Every forecast is one year ahead; longer horizons were not scored (nine annual maps).
* A quarter to two fifths of each year's change in Esri's maps goes back the next year (notebook
  01; `studies/dataset_checks`). No method places it, so Figures of Merit near 20 % are what these
  maps allow; 12 of the 13 applications compared by Pontius et al. (2008) score below 50 %.
* The projected amount of change stays 18–27 % low (notebook 04); no rule built from past maps did
  better in every window (notebook 05, section 6).
* Not run at 100 m: the neighbourhood radius in km instead of cells, and the shares as inputs or
  training weights (notebook 05, section 9).

## Decisions for the team

Filled in as the notebooks conclude. None of it is on the platform yet.

**Recommended by the results**

| Decision | Evidence | To implement on the platform |
|---|---|---|
| Report land-cover class areas from the 10 m shares, nationally and per unit (one class per cell undercounts crops and built area by 1 %, bare ground by 4 %, and misplaces 2–13 % of each class among hexagons) | `01_classes_and_change`, `03_reporting_areas` | `store_class_areas` / `class_areas.py`: store the shares with the land-cover map and sum them |
| Sum the national composition inside Guinea's outline (now over the bounding box, 1.76 times Guinea) | `03_reporting_areas` | `class_areas.py` or the published land-cover map: mask to the outline |
| Give border cells outside every ADM1/ADM2 polygon to the nearest unit, so regional tables add up to Guinea (now up to 8.8 % of the water missing) | `03_reporting_areas` | Zone rasterization, as for deforestation |
| Write each predictor to its file as soon as it is made, instead of holding all twelve in memory (5.3 GB at 100 m), and let the improved method's regression read its inputs in blocks (8 GB at peak otherwise, against the server's 9.6 GB guard; two land-use jobs at 100 m were already stopped for lack of memory) | `02_predictors_and_memory`, `06_improved_vs_deployed` | `data_loader.assemble_predictor_stack` / `prepare_predictor_files` |
| Treat clouds (Esri class 10) as no data, as Esri's legend and the methodology document do; model only transitions with at least 1,000 changed pixels, as the document does (Thanh Noi & Kappas 2018). No change in accuracy, fewer unreliable models | `04_deployed_baseline` | `tasks.run_training`: drop class 10 and filter the transitions before `train_transition_models` |
| Run the improved method: the document's Expander models with the neighbourhood predictor (Soares-Filho et al. 2002; Verburg et al. 2002), averaged per transition as ranks with a logistic regression on each cell's 3 × 3 window of classes and predictors (MOLUSCE's; Clemen 1989; Verburg et al. 2004). About a fifth better than the backend as it runs in every window, the fresh one included; about 16 min and up to 8 GB per run | `04`–`06` | `06_improved_vs_deployed`, section 5: six steps; `training.create_transition_dataset(mode="expander")` is already in `main`, the regression and the rank average are new |

**Tested, not adopted**

| Candidate | Evidence so far |
|---|---|
| Contagion inputs (Rosa et al. 2013) and FLUS's network (Liu et al. 2017) added to the blend | `05_decisions_at_100m`, section 7: FLUS's network loses on backtest; contagion is consistently a little better (+0.2–0.7) but within the seed spread on a window no choice had used, at 22 min per run |
| A recent-interval amount of change instead of the platform's trend (Mas et al. 2014) | `05_decisions_at_100m`, section 6: +2.5 on validation, −1.0 on test, level on backtest; the flicker turns direction from year to year |
| Unequal weights in the blend (document's map 0.25 or 0.75) | `06_improved_vs_deployed`, section 3: 0.25 within the seed noise, 0.75 worse |

## Server cost

Peak memory and run time of each change on the full 100 m grid, filled in as the notebooks
conclude (this laptop; mean of the three windows). On the platform each land-use task is stopped after 30 minutes
(`CELERY_TASK_TIME_LIMIT`, `main`).

| Run | Minutes | Peak memory | Notebook |
|---|---|---|---|
| Backend (`main`) as it runs | 16 | 3.7 GB | 04 |
| Backend with the growth-mode option | 29 | 3.7 GB | 04 |
| Methodology document's method | 11 | 3.7 GB | 04 |
| Clouds as no data, 1,000-pixel rule | no change | no change | 04 |
| `main`'s predictor stack held in memory (`assemble_predictor_stack`) | — | 5.3 GB | 02 |
| The improved method, the blend (document's method + regression + rank average) | 14; about 16 with `main`'s demand and allocation | 8.0 GB (the regression) | 05–06 |
| The blend with contagion (not adopted; plus about 1.2 GB of inputs per year) | 22 | 8.0 GB (the regression) | 05 |
| For comparison, the tools: PLUS (LEAS + CARS, observed amount) / MOLUSCE / CLUE-S port / FLUS (network + CA, observed amount) | 415 / 80–100 / 22 / 75 | not recorded / 7.3–8.8 GB / 6.6 GB / not recorded | 07 |

## This folder

| Path | What |
|---|---|
| `1_data/`, `2_model/`, `3_benchmark/` | The notebooks, in order; each adds `../scripts` to Python's path |
| `PLATFORM_UPDATE.md` | For the platform's developers: what changes in the land-use calculation, where in `Guinea_Project`, why, the task list and how to check it |
| `CLAUDE_GUIDE.md` | Hand-over for a new Claude chat that implements `PLATFORM_UPDATE.md` in `Guinea_Project` |
| `scripts/` | `classes100.py` (labels, change, areas), `predictors_lu100.py` (the 12 predictors as `main` builds them, and the files for the platform's code), `platform100.py` (the platform's run with its own code, timed), `deployed100.py` and `run04.py` (notebook 04), `tools_export100.py` (inputs for running MOLUSCE, PLUS and FLUS by hand), `improved100.py` and `run06.py` (notebook 06), `decisions100.py`, `run05.py`, `run05_amounts.py`, `run05_second.py`, `run05_fresh.py` and `fetch_coarse_predictors.py` (notebook 05; the last makes the 2020 rainfall, soil moisture and NDVI with the platform's Earth Engine recipe), `tools100.py` and `run07.py` (notebook 07; with `../scripts/molusce_run.py`, `tools_check250.py`) |
| `data/platform_main/` | `main`'s `landuse_tool`, read-only (its commit in `COMMIT`; not committed here) |
| `outputs/` | The small tables the notebooks save |

Inputs are on this laptop, with no Earth Engine: the Esri class and shares 2017–2025 in the shared
store on `E:\SDG AI Lab\Guinea\data_100m\`, and the coarse predictors in `../data/`. GeoTIFF inputs
for the platform's code and its runs are in `E:\SDG AI Lab\Guinea\data_100m\landuse\`. After the E:
drive dropped out on 2026-10-29, the runs read a copy of those inputs in `data/inputs/` and new runs
are written to `data/runs/` (`platform100.run_dir` finds a run in either place). Run with
`landuse_experiments/.venv`.
