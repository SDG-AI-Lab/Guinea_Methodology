# Land-use experiments at 250 m

The first stage of the land-use work, on a 250 m grid. It reproduces the intern's review
(`Intern_Tasks/guinea_land_use_review/FOM40_CEILING_ANALYSIS.pdf`), rebuilds the method of the
methodology document, improves it and compares it with other tools. The platform runs land use at
100 m; the second stage, on its grid, is in `../grid_100m/` (its own README). Folder layout and the
Python environment: `../README.md`.

## Summary

What was tested at 250 m, what holds, and how it improves the deployed land-use method. Each part points to the notebook with the details. (This was the summary notebook `00_summary`; its tool comparison is now notebook 09a.)

**The task.** The platform forecasts land cover one year ahead from ESRI's annual 10 m maps,
on a 250 m grid. The methodology document reports a Figure of Merit (FoM) of 13.80%. An
earlier review (the *improvement approach*) raised FoM from 20.0% to 28.06% on MODIS maps.
We reproduced it, tested whether it improves the deployed method, and tried other improvements.

**Figure of Merit** (Pontius et al. 2008) = correctly predicted changes ÷ (observed changes ∪ predicted changes). A change counts as correct only if both the place and the new class are right. It is the score used throughout, as in published comparisons of land-change models.

### 1 · The improvement approach reproduces on MODIS, but its 28% is not comparable

Notebooks 01–04 rerun the review's own code. Its figures reproduce: 28.25% for the final
ensemble with the review's package versions, against 28.06% in the write-up. The gains also
hold on a later interval which earlier was not tuned on (2020→2024).

However, the review measures a different task compared with the platform's implementation:

| | Improvement approach (MODIS) | Deployed platform (ESRI) |
|---|---|---|
| Land-cover map | MODIS MCD12Q1, 500 m | ESRI 10 m, on a 250 m grid |
| Forecast step | 5 years | 1 year |
| Change that is scored | All change (33–38% of MODIS changes revert, notebook 01) | All change, plus a check that a gain is not just pixels flipping back (notebook 06) |
| Amount of change | Predicted by the model | Given: the observed amount |

Why its 28% is higher than anything on the platform:

* **The starting points are the same.** On MODIS the original-style method scores 20.0%; on
  ESRI the deployed method, corrected, scores 19.1%. The difference is what the review added.
* **Flips back are easy to predict, and the all-change score counts them.** 33–38% of MODIS
  changes revert (notebook 01). Its history features predict exactly those.
* **Its models choose how much change to forecast.** Forecasting more change finds more of it
  (table below). The platform is given the observed amount.
* **Coarser cells score higher** (Pontius et al. 2004). On our data the deployed method's
  score rises from 18.9% at 250 m to 24.5% at 1 km (notebook 06, section 7). MODIS
  pixels are 500 m.

It is not that MODIS has more change: over 5 years 7.6% of its pixels changed, against 12.7%
of ESRI's in one year. So the MODIS figures show the approach works *on MODIS*. They say nothing
about the platform's 13.80%.

**Why the gain looks different on ESRI.** The same check as on ESRI, on the review's own MODIS
forecasts of 2015→2020: "new" change is change into a class the pixel did not have in the 2001,
2005 or 2010 maps. Rows marked *same amount* forecast exactly 67,954 changed pixels, so only
*where* change goes differs.

| MODIS forecast 2015→2020 | Changed pixels forecast | All-change FoM % | New-change FoM % |
|---|---:|---:|---:|
| Random forest, original features (same amount) | 67,954 | 20.8 | 11.6 |
| + land-cover history features (same amount) | 67,954 | 22.9 | 10.2 |
| Two-stage model, review's pick, same amount | 67,954 | 23.6 | 11.2 |
| Two-stage model, review's pick, model sets the amount | 115,084 | 25.9 | 15.0 |
| + satellite features (typicality XGBoost, same amount) | 67,954 | 25.4 | 13.8 |
| Final ensemble, the 28% (model sets the amount) | 82,945 | 28.2 | 15.1 |

Pixels that really changed: 75,365.

* **History features lower the new-change score on MODIS too** (11.6% → 10.2%), while raising
  the all-change score. On both products they score by predicting pixels that go back to an
  earlier class. On ESRI's annual maps, where flips back are far more common, the effect is
  much stronger (7.4% → 3.5%).
* **Part of the MODIS gain comes from forecasting more change.** The review's models set their
  own amount of change: up to 115,000 changed pixels instead of 68,000. With few correct
  predictions per forecast change, more forecasts find more change. The platform is given the
  observed amount, so this part of the gain cannot carry over.
* **Satellite features are the part that genuinely helps** (11.6% → 13.8% with the same amount
  of change). On ESRI the seasonal features helped on 2023→2024 but not on
  2024→2025 (notebook 07).

### 2 · On ESRI the improvement approach predicts flicker, not land change

Ported to ESRI (notebook 06), the approach gains 9.5 points on the document's metric. But most
of what that metric counts as change on ESRI is pixels flipping between similar classes,
mostly Trees ↔ Rangeland: 25–42% of each year's changes revert the next year, and tree canopy
cannot regrow in one year. The approach scores by predicting those flips.

| 2024→2025 | Figure of Merit (all change) % | Flip-back check: new-change FoM % | Correct predictions that are flips back % |
|---|---:|---:|---:|
| Deployed method, corrected inputs | 19.1 | 7.4 | 87.9 |
| Improvement approach (history models, averaged) | 28.6 | 3.4 | 98.2 |
| Rule: a pixel that just changed goes back | 23.7 | 0.0 | 100.0 |
| Random placement | 8.7 | 2.9 | |

* A rule with no model at all, "a pixel that just changed goes back", beats the deployed method
  on the score (23.7% against 19.1%) and scores exactly 0 on genuinely new change.
* On new change, the improvement approach is barely better than random placement.

**So the score stays the Figure of Merit over all change,** the measure every published
comparison and every tool uses, **with one check** in the methodology's appendix: an addition
must not find less genuinely new change than the deployed method. New change is change into a
class the pixel had not held in any ESRI map since 2017, the whole record, so there is no time
window to tune (notebook 06, section 7).
Returns to an earlier class ("alternation") are a separate component of change (Pontius et al.
2017), and deforestation is a conversion of land use, not a temporary loss (FAO 2020). Every
forecast is also compared with random placement of the same amount of change (Pontius et al. 2004).

### 3 · The deployed method's real score, and three fixes

* **The published 13.80% was scored over Guinea's bounding box**, 43% of it in neighbouring
  countries, and its years are off by one: the hold-out is 2024→2025 (notebook 05).
* **Three input problems** (notebook 06, section 1): 7 of 12 predictor names are attached to the
  wrong layer, so the document's importance ranking is misread; vegetation and rainfall were taken
  from after the forecast start; training included neighbouring countries with blank predictors.
* **Corrected and restricted to Guinea**, the deployed method scores **19.1%**, 2.2 times
  random placement of the same amount of change (8.7%).

A low FoM is normal in this field: in 12 of the 13 applications compared by Pontius et al.
(2008), errors outnumbered correct predictions at the data's own resolution.

**How this compares with published land-change models.** Pontius et al. (2008) scored 13
applications of 9 peer-reviewed land-change models the same way: figure of merit, producer's
accuracy (share of observed change predicted correctly) and user's accuracy (share of predicted
change that is correct), all at the data's own resolution and counting all change.

| Application | Model | Resolution | Years forecast | Figure of merit % | Producer's accuracy % | User's accuracy % |
|---|---|---|---:|---:|---:|---:|
| Perinet, Madagascar | Logistic Regression | 30 m | 43 | 59 | 73 | 75 |
| Costa Rica | CLUE | 15 km | 11 | 49 | 63 | 65 |
| Haidian, China | CLUE-S | 100 m | 9 | 43 | 49 | 65 |
| Honduras | CLUE | 15 km | 19 | 38 | 60 | 49 |
| Kuala Lumpur, Malaysia | CLUE-S | 150 m | 9 | 28 | 35 | 50 |
| Maroua, Cameroon | CLUE-S | 250 m | 12 | 23 | 40 | 32 |
| Cho Don, Vietnam | SAMBA | 32 m | 11 | 21 | 26 | 37 |
| Detroit, USA | LTM | 26 m | 20 | 15 | 25 | 25 |
| Twin Cities, USA | LTM | 30 m | 7 | 11 | 20 | 20 |
| Worcester, USA | Geomod | 30 m | 28 | 9 | 19 | 14 |
| Holland (15 classes) | Environment Explorer | 500 m | 4 | 7 | 10 | 15 |
| Holland (8 classes) | Land Use Scanner | 500 m | 4 | 5 | 19 | 6 |
| Santa Barbara, USA | SLEUTH | 50 m | 12 | 1 | 1 | 7 |
| **Guinea ESRI: deployed method, corrected** | this study | 250 m | 1 | 19.0 | 31.6 | 32.0 |
| **Guinea ESRI: improved method** (contagion + logistic regression + FLUS's network) | this study | 250 m | 1 | 20.4 | 33.5 | 33.9 |

Pontius et al. (2008), Table 2 and Fig. 4; Guinea rows computed the same way (test year 2024→2025).

* **12 of the 13 published applications have a figure of merit below 50%:** more errors than
  correct predictions. Only Perinet, a 43-year forecast of a one-way transition, is above.
* **The deployed method scores 19.0% and the improved method 20.4%,** the middle of the
  published range (between Cho Don's 21% and Detroit's 15%), with a forecast of **one year**
  where the published ones span 4 to 43 years.
* Pontius et al. found that FoM rises with the amount of change observed: all six of their
  applications under 15% had less than 10% net change. Guinea's net change in one year is
  6.2%, yet both our forecasts score above 15%.

### 4 · Comparison with existing tools

Notebook 09a; the results table is in "Comparison with other tools" below.

### 5 · What improved the deployed method

**The rule** (README): settings chosen on 2023→2024, tested once on 2024→2025. An
improvement must beat the deployed method by more than the spread across random seeds (about
±0.05 points; Bouthillier et al. 2021), beat random placement (Pontius et al. 2004), and pass
the flip-back check (section 2 above).

* **Improvement approach (history models) and hybrid** (notebooks 06–07): higher scores, but
  they fail the flip-back check.
* **Seasonal MODIS features** (notebook 07): help on 2023→2024, not on 2024→2025, also when
  standardised within each year. Not adopted.
* **Contagion features** (how much new change happened nearby last year; Rosa et al. 2013;
  notebook 08): improve, with no new data.
* **Logistic regression**, the model of MOLUSCE and CLUE-S (an insight from notebook 09a; notebook
  08): improves. **Averaged with the Random Forest with contagion features, it is the chosen
  model: 19.9% on 2024→2025, +0.8 points over the deployed method, 2.3 times random placement.**
  It was the improved method of the first tool comparison.
* **FLUS's neural network**, reproduced in code (an insight from notebook 09a; notebook 09, section
  4): averaged with that model, each of the three with equal weight, it improves again:
  **20.3% on 2024→2025, +1.3 points over the deployed method**, 2.3 times random placement. **This
  is now the improved method**, used in notebook 09a and notebook 10. The
  idea came from FLUS's 2024→2025 result, so that year is a weaker test for it; 2025→2026 is the
  independent one.
* **Checks of the chosen method:** the averaging weights are fixed, not tuned, and changing them
  moves the 2024→2025 score by 0.22 points at most (notebooks 08 and 09). On an earlier year,
  2022→2023, the method gains +0.8 points over the deployed method and passes the flip-back check
  (notebook 11, section 5).

Every step, in order, with where the idea came from: notebook 11.

### 6 · Recommendations

1. **Report the forecast with the standard set, on Guinea only:** Figure of Merit (all change)
   as the headline, with producer's and user's accuracy, quantity and allocation disagreement,
   and random placement as the floor (Pontius et al. 2004, 2008; Pontius & Millones 2011). The
   published 13.80% was scored over the bounding box.
2. **Fix the three input problems** (predictor names, dates, training area).
3. **Adopt contagion features, averaged with a logistic regression and a copy of FLUS's network:**
   20.3% on 2024→2025 (mean of 3 seeds), +1.3 points over the corrected deployed method, ahead of
   MOLUSCE, CLUE-S, PLUS and FLUS as they run (notebook 09a). No new data.
4. **Do not adopt the improvement approach's history models or the hybrid:** their gain is
   predicted flicker. Seasonal features: not adopted (no gain on the test year, also when standardised).
5. **Keep the flip-back check in the methodology's appendix** for every future change.
6. **Confirm on 2025→2026** once ESRI publishes its 2026 map. Methods and settings are now fixed,
   so the method can be adopted now: the gain also holds on 2022→2023 (+0.8 points) and does not
   depend on the averaging weights.
7. **Judge land-change tools by the Figure of Merit and its components, not kappa or overall
   accuracy** (notebook 09a; Pontius & Millones 2011).
8. **The tool comparison is complete** (notebooks 09, 09a, 10). Keep the two-table layout, and
   present the platform with both its accuracy and what it takes to run each tool.
9. **The copy of FLUS's network is added to the improved method** (notebook 09, section 4; decided
   24 October 2026): it passes the rule (20.3% on 2024→2025, +0.46), runs in code in about 2 minutes
   a year, needs no new data and no FLUS program. The tool comparison (notebook 09a, notebook 10) is
   rerun with it. Every table is shown with and without the deployed method.
10. **Keep the twelve predictors; correct their labels in the methodology document** (notebook 12).
    Leaving each out in turn shows NDVI carries most of the information (about 4 points of Figure of
    Merit), then rainfall, biomass and population. Leaving out the six with no clear effect costs
    accuracy on 2024→2025; mining areas and fire, the two Guinea-specific candidates, are not
    chosen. Seven of the twelve labels in §6.2 name the wrong layer.

## Notebooks, in order

The notebooks are grouped in folders by stage.

| # | Notebook | Question | Status |
|---|---|---|---|
| 01 | `1_intern_improvement/01_modis_baselines_and_history.ipynb` | Do the intern's baselines and history-feature gains reproduce? | Done: reproduces (14 of 15 rows within tolerance; the unselected 26.49% row does not, in either environment) |
| 02 | `1_intern_improvement/02_satellite_features.ipynb` | Download and check MODIS NDVI/EVI and Landsat predictors | Done: identical coverage and scenes |
| 03 | `1_intern_improvement/03_modis_satellite_models_and_ensemble.ipynb` | Do the satellite models and the 28.06% ensemble reproduce? | Done: reproduces. In the intern's versions the ensemble starts from the intern's member and scores 28.25% (+0.19); ours 28.59% (+0.53) |
| 04 | `1_intern_improvement/04_modis_2020_2024_test.ipynb` | Do the gains hold on 2020→2024, never used for selection? | Done: +3.9 to +5.2 over the original method (land-cover-only rounds) |
| 05 | `2_improvement_and_benchmark/05_esri_data_and_baseline.ipynb` | Reproduce the deployed ESRI baseline (FoM 13.80%) | Done: 13.80% reproduced exactly; **Guinea-only baseline 17.32%** (see below) |
| 06 | `2_improvement_and_benchmark/06_esri_port_experiments.ipynb` | Carry the methods that held up over to ESRI | Done: corrected deployed method 19.1%; the improvement approach +9.5, but it fails the flip-back check (its gain is predicted label flicker) |
| 07 | `2_improvement_and_benchmark/07_esri_hybrid_and_seasonal.ipynb` | Hybrid (returns vs new change) and seasonal MODIS features on ESRI | Done: hybrid fails the flip-back check; seasonal features help on 2023→2024, not on 2024→2025, also when standardised within each year |
| 08 | `2_improvement_and_benchmark/08_esri_contagion.ipynb` | Contagion features, and logistic regression from the tool comparison | Done: contagion improves (+1.0 / +0.9); averaged with a logistic regression it is the chosen model, 19.9% on 2024→2025; the 0.5 weight is not critical (0.25–0.75: 19.7–20.1%) |
| 09 | `2_improvement_and_benchmark/09_esri_neural_network.ipynb` | FLUS's idea: does a neural network improve the averaged model? | Done: first copy of FLUS's network not adopted; FLUS varies ±0.14 between runs; FLUS's own maps averaged with the improved method pass the rule (20.5%); **FLUS's network reproduced in code (`flus_ann.py`), averaged with the improved method: 20.3%, passes the rule**; its one-third weight is not critical (1/4–1/2: 20.3–20.4%) |
| 09a | `2_improvement_and_benchmark/09a_tool_comparison.ipynb` | MOLUSCE, CLUE-S, PLUS and FLUS on exactly our inputs, scored with the validation set the literature recommends (formerly section 4 of the summary notebook) | Done: as each tool runs, the improved method has the highest Figure of Merit (20.4%, FLUS 18.1%, deployed 19.0%); with the same amount of change it is slightly ahead of FLUS (20.4% against 20.2%); its suitability maps rank best (AUC 0.748) |
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

Full tables: `../outputs/summary/06_summary.csv`, `07_results.csv`, `08_models.csv`.

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

Results and what they show: notebooks 09a and 10. How the comparison was done
(data, each tool's settings, checks, limits, what can be said to clients):
`2_improvement_and_benchmark/BENCHMARK_DETAILS.md`. `2_improvement_and_benchmark/BENCHMARK_GUIDE.md` explains how to run MOLUSCE
(QGIS), PLUS and FLUS on the same inputs (from `landuse_experiments/`, `python scripts/benchmark_export.py` writes them to
`../outputs/benchmark/`). CLUE-S was run with the R package lulcc (`../scripts/benchmark_lulcc.R`; open it in
RStudio and click Source to rerun; its log is `../outputs/benchmark/output_lulcc/run_log.txt`), and
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
1 km and PLUS slightly ahead at 5 km. Notebook 09a also shows the table without the deployed method. MOLUSCE ANN's
potential maps are 0 on more than 99% of pixels, so its same-amount score reflects tie-breaking.
Kappa and overall accuracy favour forecasting nothing (Pontius & Millones 2011).

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
| Compare with MOLUSCE, CLUE-S, PLUS and FLUS on the same data | Done (notebook 09a) |
| Test FLUS's idea (neural network per class) by the same rule | Done (notebook 09): FLUS's own maps pass the rule; reproduced in code, its network averaged with the improved method passes too (20.3%). Added to the improved method (24 October); comparison rerun |
| FLUS run-to-run variation (two more runs) | Done (notebook 09): ±0.12 as it runs, ±0.14 with the same amount |
| Comparison with projected amounts of change | Done (notebook 10) |

## References

- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems* 3.
- Dietterich, T.G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems*, Lecture Notes in Computer Science 1857, 1–15.
- FAO (2020). Global Forest Resources Assessment 2020: Terms and definitions. Forest Resources Assessment Working Paper 188. Rome.
- Liang, X., Guan, Q., Clarke, K.C., et al. (2021). Understanding the drivers of sustainable land expansion using a patch-generating land use simulation (PLUS) model: a case study in Wuhan, China. *Computers, Environment and Urban Systems* 85, 101569.
- Liu, X., Liang, X., Li, X., et al. (2017). A future land use simulation model (FLUS) for simulating multiple land use scenarios by coupling human and natural effects. *Landscape and Urban Planning* 168, 94–116.
- Moulds, S., Buytaert, W., Mijic, A. (2015). An open and extensible framework for spatially explicit land use change modelling: the lulcc R package. *Geoscientific Model Development* 8, 3215–3229.
- Pontius, R.G. Jr. (2018). Criteria to confirm models that simulate deforestation and carbon disturbance. *Land* 7(3), 105.
- Pontius, R.G. Jr., Boersma, W., Castella, J.-C., et al. (2008). Comparing the input, output, and validation maps for several models of land change. *Annals of Regional Science* 42(1), 11–37.
- Pontius, R.G. Jr., Huffaker, D., Denman, K. (2004). Useful techniques of validation for spatially explicit land-change models. *Ecological Modelling* 179(4), 445–461.
- Pontius, R.G. Jr., Krithivasan, R., Sauls, L., Yan, Y., Zhang, Y. (2017). Methods to summarize change among land categories across time intervals. *Journal of Land Use Science* 12(4), 218–230.
- Pontius, R.G. Jr., Millones, M. (2011). Death to Kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. *International Journal of Remote Sensing* 32(15), 4407–4429.
- Pontius, R.G. Jr., Peethambaram, S., Castella, J.-C. (2011). Comparison of three maps at multiple resolutions: a case study of land change simulation in Cho Don District, Vietnam. *Annals of the Association of American Geographers* 101(1), 45–62.
- Pontius, R.G. Jr., Schneider, L.C. (2001). Land-cover change model validation by an ROC method for the Ipswich watershed, Massachusetts, USA. *Agriculture, Ecosystems & Environment* 85, 239–248.
- Pontius, R.G. Jr., Si, K. (2014). The total operating characteristic to measure diagnostic ability for multiple thresholds. *International Journal of Geographical Information Science* 28(3), 570–583.
- Rosa, I.M.D., Purves, D., Souza, C. Jr., Ewers, R.M. (2013). Predictive modelling of contagious deforestation in the Brazilian Amazon. *PLoS ONE* 8(10), e77231.
- Verburg, P.H., Overmars, K.P. (2009). Combining top-down and bottom-up dynamics in land use modeling: exploring the future of abandoned farmlands in Europe with the Dyna-CLUE model. *Landscape Ecology* 24(9), 1167–1181.
- Verburg, P.H., Soepboer, W., Veldkamp, A., et al. (2002). Modeling the spatial dynamics of regional land use: the CLUE-S model. *Environmental Management* 30(3), 391–405.
