# Comparison with existing tools: how it was done

The results and what they show are in notebook 00, section 4 (as each tool runs, with the same
amount of change, and the flip-back check) and notebook 10 (every tool given the amount projected
from 2023→2024). How the improved method was built, step by step: notebook 11. How to run MOLUSCE,
PLUS and FLUS by hand on the same inputs: `BENCHMARK_GUIDE.md`.

This document holds what those do not: why the tools were run on our data, the data and each
tool's settings, the checks on the tools and the scoring, what running each tool takes, what can
be said to clients, and the limits. The code is in `scripts/`: `benchmark_export.py`,
`benchmark_scoring.py`, `clue_s.py`, `benchmark_lulcc.R` and `benchmark_fom_check.R`.

## 1. Why the tools were run on our data

Does our land-use forecast perform better than established land-change tools? Comparing
published scores from other studies is not possible: they use different places, maps, years and
resolutions, and the Figure of Merit depends strongly on how much change a place has (Pontius et
al. 2008). So the tools were run **on our own data** and scored **the same way**. This follows the
published comparisons of land-change software, which run several tools on one dataset and compare
their maps (Mas et al. 2014; Paegelow et al. 2014).

## 2. The same data for every tool

| | |
|---|---|
| Land cover | ESRI 10 m annual maps, resampled to the platform's 250 m grid; 7 classes (water, trees, flooded vegetation, crops, built area, bare ground, rangeland) |
| Area | Guinea only (4.0 million pixels) |
| Learn from | 2023 → 2024 (the 2024 map and the predictors dated 2024 for tools that learn from one map) |
| Forecast | 2025, one step from the 2024 map |
| Checked against | ESRI's 2025 map. No tool was given it. |
| Predictors | The deployed method's 12: elevation, population density, precipitation, soil moisture, above-ground biomass, NDVI, slope, aspect, distance to roads, distance to waterways, points-of-interest density, protected areas |

The inputs are written by `scripts/benchmark_export.py` to `outputs/benchmark/inputs/`.

## 3. The tools and how each was run

| Tool | How it works | Amount of change it is given | How it was run |
|---|---|---|---|
| **MOLUSCE 5.1** (QGIS plugin, NextGIS) | Neural network (ANN) or logistic regression (LR) learns transition potentials from 2023→2024; cellular automata simulate | None: it extrapolates the past rates, limited by its model | In QGIS, by us, with the plugin's defaults (`BENCHMARK_GUIDE.md`, tool 1) |
| **CLUE-S** (Dyna-CLUE 2.7; Verburg et al. 2002; Verburg & Overmars 2009) | One logistic regression per class says where each class is likely; an iterative allocation meets the amount of each class | The observed 2025 total of each class, as Pontius et al. (2008) gave every model | With the R package **lulcc** 1.0.4 (Moulds et al. 2015), `benchmark_lulcc.R`: the results in the comparison. Checked against the authors' **original program** and our **Python version** of the original code (section 5) |
| **PLUS** (Liang et al. 2021) | Random forest per class (LEAS), then patch-growing cellular automata (CARS) | The observed 2025 total of each class | In its Windows program, by us: LEAS, then CARS with the 2024 map, neighbourhood 3, patch threshold 0.2, expansion coefficient 0.1, seeds 0.1 |
| **FLUS** (GeoSOS-FLUS 2.3; Liu et al. 2017) | Neural network per class, then cellular automata with "self-adaptive inertia" | The observed 2025 total of each class | In its Windows program, by us (`BENCHMARK_GUIDE.md`, tool 5): neural network on 2% of pixels, 12 hidden nodes; cellular automata with every conversion allowed and equal neighbourhood weights |
| **Deployed method** (the platform) | Random forest per transition; the platform's allocator | The observed number of each transition | Rebuilt with three input errors corrected (notebooks 05–06) |
| **Improved method** | Deployed method + contagion features, averaged with a logistic regression and a copy of FLUS's network | The observed number of each transition | Notebooks 08–09; steps in notebook 11 |

Tools not run: Dinamica EGO (download site unreachable), TerrSet Land Change Modeler and
CA–Markov (paid licence). The other models compared by Pontius et al. (2008) are commercial
(Environment Explorer, Land Use Scanner), model only urban growth (SLEUTH, Land Transformation
Model), model one transition (Geomod) or were built for one site (SAMBA).

## 4. How they are compared

The tools differ in how much they are told about the amount of change, so each is compared twice:

1. **As each tool runs.** What a user of that tool would get. This is the main comparison.
2. **With the same amount of change.** Each model's own suitability (probability) maps are used
   to place the observed number of each transition, with the same allocator for every model. Only
   *where* change goes then differs.

A third check needs no amount of change at all: the **AUC** of each model's suitability maps, i.e.
how well it ranks the pixels that really changed above those that did not (Pontius & Schneider
2001; Pontius & Si 2014).

The metrics and their sources are listed in notebook 00, section 4. The Figure of Merit's four
parts (hits, misses, wrong hits, false alarms; Pontius et al. 2008; Varga et al. 2019) are in
`outputs/summary/00_benchmark_detail.csv`. The same choices are recommended by reviews of land-change
model validation (van Vliet et al. 2016), and FLUS and lulcc themselves include a Figure of Merit
tool. The Figure of Merit (all change) was fixed as the headline on 22 October 2026, before PLUS's
simulation and FLUS were run (23 October).

## 5. Checks on the tools and the scoring

* **Our scorer is correct.** The Figure of Merit of four forecasts was recomputed with lulcc's own
  implementation of Pontius et al. (2011) (`benchmark_fom_check.R`): identical to two decimals.
  MOLUSCE's own correctness and kappa values are reproduced exactly (`molusce_metrics`).
* **The CLUE-S results come from published R code, and can be rerun.** `benchmark_lulcc.R` does
  the whole CLUE-S run in R with lulcc: the regressions, the conversion elasticities and the
  allocation. It can be opened in RStudio and run with *Source*, or in R's own window
  (`Rgui.exe`, *File > Source R code*). It writes everything it prints to
  `outputs/benchmark/output_lulcc/run_log.txt`: the R and lulcc versions, the start and end
  times, the demand and the allocated area of each class, a checksum (MD5) of every output file,
  and the full list of R packages loaded. The run is seeded: rerunning it on 23 October 2026 gave
  files with the same checksums as the first run.
* **CLUE-S does not depend on who programmed it.** The original program, lulcc and our Python
  version were given the same inputs: the 2024 map, the 12 predictors, conversion elasticities
  equal to each class's share of unchanged pixels in 2023–2024, every conversion allowed, the
  observed 2025 class totals, and convergence limits of 1% on average and 5% for any class (6%
  for the original program; the manual suggests 0.35% and 3%; see section 8). The original program
  and the Python version used the same regression coefficients; lulcc fitted its own. Results:

  | CLUE-S, 2024→2025 | R package lulcc (in the comparison) | Original program | Python version |
  |---|---:|---:|---:|
  | Changes forecast | 249,520 | 249,489 | 249,490 |
  | Figure of Merit | 14.76% | 14.78% | 14.78% |
  | Producer's / user's accuracy | 19.1% / 38.9% | 19.2% / 39.0% | 19.2% / 39.0% |
  | Same class as the original program | 99.87% of pixels | – | 100.00% of pixels |

  The original program and the Python version give the same map; lulcc, programmed
  independently, differs on 0.13% of pixels.
* **PLUS, FLUS and MOLUSCE** were run in their own programs, which have no batch mode. Their output
  files and screenshots of their validation (MOLUSCE) and finished runs (PLUS, FLUS) are in
  `outputs/benchmark/`.
* **Random variation is small.** Our methods were trained with three random seeds; their scores
  differ by ±0.04 points or less (Bouthillier et al. 2021). FLUS, run three times, varies by ±0.12
  points as it runs and ±0.14 with the same amount of change (notebook 09).
* **The averaging weights are fixed, not tuned.** Other weights change the 2024→2025 score by
  0.22 points at most (notebooks 08 and 09).
* **The reference map itself is noisy.** Esri changes the class of 10–15 % of Guinea every year,
  and a third of that change goes back the following year, mostly between trees and rangeland
  (`studies/dataset_checks/01_desk_check.ipynb`). Every tool is scored against the same map, so the
  ranking is fair, but the scores measure how well Esri's next map is forecast, not only real land
  change.

## 6. What running each tool takes

* **PLUS:** its simulation step crashed on our data repeatedly before it ran, and it stopped 15,008
  pixels off the demand.
* **FLUS:** works only by clicking through its windows; it cannot be scripted or scheduled.
* **CLUE-S:** the authors' program is an older Windows program that needed thousands of iterations
  and a relaxed tolerance to finish on Guinea. The R package lulcc ran out of memory on a 16 GB
  machine until its steps were split up.
* **MOLUSCE:** a QGIS plugin; on one year of change it forecasts almost no change.

The platform runs from code at national scale with every setting recorded, so a forecast can be
repeated each year when a new map is published. A defensible message is: *as accurate as the best
established tools, with the best-ranked suitability maps, and operational at national scale.*

## 7. What can be said to clients

* "On the same data and the standard measure of the land-change literature, our method
  forecasts land change more accurately than MOLUSCE, CLUE-S, PLUS and FLUS" (20.4% against
  18.1% for FLUS, the best of the other tools, as each tool runs).
* "It is tested the strict way: built on 2023→2024, checked once on 2025, a year it never saw."
* "It stays the most accurate when every tool is given only what a user would know about the
  coming year (13.1% against 10.1% for FLUS, the best of the other tools)."
* "Given the same amount of change, it places change at least as well as the best other tool
  (20.3% against 20.0% for FLUS, three runs each)."

Not to be said:

* That it is much better at deciding *where* change goes. With the amount held equal it is 0.3
  points ahead of FLUS, and part of that comes from a copy of FLUS's own network.
* That it is the most accurate at every scale. At 1 km FLUS is level, and at 5 km PLUS is slightly
  ahead.
* That it beats every tool at every scale on equal information: with projected amounts FLUS is
  ahead at 1 km and 5 km.

## 8. Limits

* **One test year, and a back-test.** The improved method's settings are fixed; 2025→2026 will be
  an independent test when ESRI publishes its 2026 map. Meanwhile, on an earlier year (2022→2023)
  it beats the deployed method by 0.8 points (15.9% against 15.2%) and passes the flip-back check
  (notebook 11, section 5). For the FLUS network part, 2024→2025 is a weaker test, as the idea came
  from FLUS's result on that year.
* **PLUS**'s simulation stopped with a deviation of 15,008 pixels from the demand (it stops when its
  deviation no longer falls): Water 17,051 pixels against 24,555, Bare ground 1,734 against 648.
  FLUS met the demand within 93 pixels.
* **CLUE-S** convergence: Bare ground (648 pixels, 0.02% of Guinea) could not be brought within
  the manual's 3% in 20,000 iterations, and its 4% alone pushes the average over the 7 classes
  above the manual's 0.35%. So the CLUE-S runs used 1% on average and 5% for any class. The
  original program stops by itself after 20,000 iterations; at 5% it stopped there with Bare
  ground 5.7% off and wrote no map, so it was run with 6%. These limits concern 30–40 pixels of
  Bare ground.
* **MOLUSCE ANN** produced potential maps that are 0 almost everywhere, so its same-amount score
  carries no information.
* Each tool was run with its defaults or the settings in its manual. A specialist could tune any
  of them further; so could we ours.

## References

* Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks.
  *Proceedings of Machine Learning and Systems* 3.
* Liang, X., Guan, Q., Clarke, K.C., et al. (2021). Understanding the drivers of sustainable land
  expansion using a patch-generating land use simulation (PLUS) model: a case study in Wuhan,
  China. *Computers, Environment and Urban Systems* 85, 101569.
* Liu, X., Liang, X., Li, X., et al. (2017). A future land use simulation model (FLUS) for
  simulating multiple land use scenarios by coupling human and natural effects. *Landscape and
  Urban Planning* 168, 94–116.
* Mas, J.-F., Kolb, M., Paegelow, M., Camacho Olmedo, M.T., Houet, T. (2014). Inductive
  pattern-based land use/cover change models: a comparison of four software packages.
  *Environmental Modelling & Software* 51, 94–111.
* Moulds, S., Buytaert, W., Mijic, A. (2015). An open and extensible framework for spatially
  explicit land use change modelling: the lulcc R package. *Geoscientific Model Development* 8,
  3215–3229.
* Paegelow, M., Camacho Olmedo, M.T., Mas, J.-F., Houet, T. (2014). Benchmarking of LUCC
  modelling tools by various validation techniques and error analysis. *Cybergeo: European
  Journal of Geography*, 701.
* Pontius, R.G. Jr., Boersma, W., Castella, J.-C., et al. (2008). Comparing the input, output,
  and validation maps for several models of land change. *Annals of Regional Science* 42(1), 11–37.
* Pontius, R.G. Jr., Peethambaram, S., Castella, J.-C. (2011). Comparison of three maps at multiple
  resolutions: a case study of land change simulation in Cho Don District, Vietnam. *Annals of the
  Association of American Geographers* 101(1), 45–62.
* Pontius, R.G. Jr., Schneider, L.C. (2001). Land-cover change model validation by an ROC method
  for the Ipswich watershed, Massachusetts, USA. *Agriculture, Ecosystems & Environment* 85, 239–248.
* Pontius, R.G. Jr., Si, K. (2014). The total operating characteristic to measure diagnostic
  ability for multiple thresholds. *International Journal of Geographical Information Science*
  28(3), 570–583.
* van Vliet, J., Bregt, A.K., Brown, D.G., van Delden, H., Heckbert, S., Verburg, P.H. (2016). A
  review of current calibration and validation practices in land-change modeling.
  *Environmental Modelling & Software* 82, 174–182.
* Varga, O.G., Pontius, R.G. Jr., Singh, S.K., Szabó, S. (2019). Intensity Analysis and the Figure
  of Merit's components for assessment of a Cellular Automata–Markov simulation model.
  *Ecological Indicators* 101, 933–942.
* Verburg, P.H., Overmars, K.P. (2009). Combining top-down and bottom-up dynamics in land use
  modeling: exploring the future of abandoned farmlands in Europe with the Dyna-CLUE model.
  *Landscape Ecology* 24(9), 1167–1181.
* Verburg, P.H., Soepboer, W., Veldkamp, A., Limpiada, R., Espaldon, V., Mastura, S.S.A. (2002).
  Modeling the spatial dynamics of regional land use: the CLUE-S model. *Environmental
  Management* 30(3), 391–405.
