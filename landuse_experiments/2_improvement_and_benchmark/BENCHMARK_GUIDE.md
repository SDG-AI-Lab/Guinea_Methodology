# Testing other land-change tools on our data

Goal: run established land-change tools on **exactly the data our notebooks use**, then
score their 2025 forecasts with the same scorer as the deployed method.

## The test

| | |
|---|---|
| Learn from | ESRI land cover **2023 → 2024** |
| Forecast | **2025**, one step from the 2024 map |
| Scored against | the observed ESRI 2025 map (`reference/`) |
| Area | Guinea only (everything outside is nodata) |
| Grid | the deployed 250 m grid, EPSG:4326, 3443 × 2444 px |

Rules, so the comparison is fair:

1. **Never give a tool the 2025 map as an input.** It is only for checking at the end.
2. **Do not reproject or resample anything.** Keep the grid as delivered, so the
   forecast lines up pixel for pixel with the reference.
3. One forecast step only (2024 → 2025).

## The files (`outputs/benchmark/`)

Recreate them any time with `python scripts/benchmark_export.py`.

| File | What |
|---|---|
| `inputs/landcover_2023.tif`, `inputs/landcover_2024.tif` | Land cover, classes 1–7, 0 = nodata |
| `inputs/factors_2023/` | The deployed method's 12 predictors, with vegetation, rain and soil moisture from 2023: use for **training** |
| `inputs/factors_2024/` | The same, dated 2024: use for **forecasting** |
| `inputs/factors_2023_cube.tif`, `inputs/factors_2024_cube.tif` | The same 12 predictors as one 12-band raster (for Dinamica EGO) |
| `reference/landcover_2025_reference.tif` | Observed 2025. **Checking only** |
| `legend_and_demand.csv` | Class codes and names, and the number of pixels per class in 2023, 2024 and 2025 |

Classes: 1 Water, 2 Trees, 3 Flooded vegetation, 4 Crops, 5 Built area, 6 Bare ground,
7 Rangeland. Predictors: elevation, population density, precipitation, soil moisture,
above-ground biomass, NDVI, slope, aspect, distance to roads, distance to waterways,
points-of-interest density, protected areas. Predictor nodata is −9999.

The QGIS window titles below may differ slightly between plugin versions. The steps are the same.

## Tool 1: MOLUSCE (QGIS plugin, free)

MOLUSCE is NextGIS's land-use change simulation plugin. It learns transition potentials
with an artificial neural network (ANN), logistic regression (LR), weights of evidence
(WoE) or multi-criteria evaluation, then simulates with cellular automata.

1. **Install:** QGIS → *Plugins → Manage and Install Plugins* → search "MOLUSCE" → install.
2. **Inputs tab:** initial land cover `landcover_2023.tif` (2023), final land cover
   `landcover_2024.tif` (2024), spatial variables: the 12 files in `factors_2023/`.
   Click *Check geometry*.
3. **Evaluating correlation tab (optional):** this only checks whether two predictors carry
   the same information (for example elevation and rainfall). It does not change the model.
   Pearson's correlation suits continuous predictors; Cramer's V and joint information
   uncertainty are for categorical maps. Pearson is fine here.
4. **Area changes tab:** compute the change map and transition matrix. "Raster units" (pixel
   counts) is correct: the grid is in degrees, so square metres would be wrong anyway.
5. **Transition potential modelling tab:** ANN (multi-layer perceptron), and in a second run
   logistic regression. Neighbourhood 1 px, random sampling, 50,000 samples, other settings
   at their defaults. Train.
6. **Cellular automata simulation tab:**
   * Tick **"separate spatial variables version for simulation"** and add the 12 files in
     `factors_2024/`. This is the forecast's own year of predictors.
   * Tick the output **"transition potential maps"** (and "certainty function" if offered) and
     save them in a folder named after the method, e.g. `output_molusce/potentials_ann/`.
     These are outputs of the model, not inputs.
   * 1 iteration. Save the forecast as `molusce_ann_2025.tif` (or `molusce_lr_2025.tif`).
7. *(Optional)* Validation tab against `reference/landcover_2025_reference.tif`.

**What the first run showed (2026-10-22):** MOLUSCE's simulator changes a pixel only where
the model's single most likely class differs from the current one. In one year 87% of pixels
stay the same, so the ANN forecast only 2,291 changes (507,539 happened). Its kappa (0.76)
equals that of forecasting no change at all. The saved transition potential maps let us
place the observed amount of change with MOLUSCE's own potentials, exactly as the deployed
method does: a like-for-like test of where each model puts change.

## Tool 2: PLUS (free, Windows)

PLUS (Liang et al. 2021) is widely used for land-use simulation. It finds what drives each
class's expansion with a random forest (LEAS), then grows patches with cellular
automata (CARS). Download it from the HPSCIL group's GitHub
(`Patch-generating_Land_Use_Simulation_Model`). If the download link there points to Baidu
and is blocked, look for the executable inside the repository's own files or under *Releases*;
HPSCIL also publishes a parallel version (mcPLUS). If none is reachable, skip PLUS: MOLUSCE and
Dinamica EGO cover the same kind of model.

1. **Extract land expansion:** first map `landcover_2023.tif`, second map
   `landcover_2024.tif` → saves an expansion map.
2. **LEAS:** expansion map from step 1, plus the 12 files in `factors_2023/`. Default
   random-forest settings, about 20% sampling. It writes one development-probability map per class.
3. **CARS:**
   * Start map: `landcover_2024.tif`
   * Probabilities: from step 2
   * **Demand:** the `pixels_2025_observed` column of `legend_and_demand.csv`.
     This gives PLUS the observed 2025 amount of each class, as the deployed
     validation does.
   * Transition matrix: all 1 (every change allowed). Neighbourhood weights and patch
     settings: defaults.
   * Save the result as `plus_2025.tif`.
4. Keep the **development-probability maps from LEAS** (step 2), one per class. As with MOLUSCE's
   potentials, they let us place the observed amount of change with PLUS's own model.

settings configuration:
LEAF: random forest regression (RFR), number of regression tree 20, sampling rate 0.01, mTry 12, operating parameters thread 1. uniform sampling is unchecked.
---=== Please wait ===---
Sampling Number: 4310
 Result -> min max: -6,1656
 Result -> min max: 0,177.11
 Result -> min max: 0,4288.19
 Result -> min max: 0,0.470276
 Result -> min max: 0,155.06
 Result -> min max: -1984,9306
 Result -> min max: 0,54.7955
 Result -> min max: 0,359.82
 Result -> min max: 0,167.8
 Result -> min max: 0,188.367
 Result -> min max: 0,1
 Result -> min max: 0,1
Find and exclude invalid data: 0
Valid data : 4310
Positive:13
Negative:4297
Add some Backgroud Points for Type 1
Positive:1464
Negative:2846
Add some Backgroud Points for Type 2
Positive:37
Negative:4273
Add some Backgroud Points for Type 3
Positive:190
Negative:4120
Add some Backgroud Points for Type 4
Positive:33
Negative:4277
Add some Backgroud Points for Type 5
Positive:3
Negative:4307
Add some Backgroud Points for Type 6
Positive:2570
Negative:1740
Add some Backgroud Points for Type 7

Train data, please wait...
save accuracy record...

Trees: 20 NVar(mtry):12
Type 0
make df-trees success = 1
RMSE =  0.0217025
OOB RMSE =  0.060958
Run time: 0.266
Contribution:, 0.118805,0.0358178,0.248813,0.0600185,0.0358178,0.157746,0.0976157,0.0424596,0.0359444,0.148227,0.0187357,

Trees: 20 NVar(mtry):12
Type 1
make df-trees success = 1
RMSE =  0.16357
OOB RMSE =  0.464749
Run time: 0.843
Contribution:, 0.0894948,0.0996177,0.138481,0.123383,0.0725361,0.152964,0.0529298,0.0437738,0.0733538,0.0791423,0.0718629,0.0024608

Trees: 20 NVar(mtry):12
Type 2
make df-trees success = 1
RMSE =  0.0291806
OOB RMSE =  0.0806222
Run time: 0.125
Contribution:, 0.592481,0.0663273,0.0504095,0.0241424,0.0322737,0.0155788,0.0361764,0.0390094,0.0942562,0.0301266,0.0192186,

Trees: 20 NVar(mtry):12
Type 3
make df-trees success = 1
RMSE =  0.066378
OOB RMSE =  0.189741
Run time: 0.5
Contribution:, 0.0499781,0.103891,0.049341,0.124071,0.142592,0.344123,0.0437983,0.0147329,0.0339078,0.0438445,0.0489039,0.00081689

Trees: 20 NVar(mtry):12
Type 4
make df-trees success = 1
RMSE =  0.0316044
OOB RMSE =  0.0912239
Run time: 0.422
Contribution:, 0.0543306,0.317104,0.0813979,0.0576074,0.0170909,0.0210086,0.0259604,0.0382017,0.112443,0.0290517,0.245133,0.00067033

Trees: 20 NVar(mtry):12
Type 5
make df-trees success = 1
RMSE =  0.00628038
OOB RMSE =  0.0180639
Run time: 0.078
Contribution:, 0,-0.0029926,0.00288086,0.00715395,0.908643,0.0580202,0,0.0188216,0,0,0.00747324,

Trees: 20 NVar(mtry):12
Type 6
make df-trees success = 1
RMSE =  0.176563
OOB RMSE =  0.504921
Run time: 0.766
Contribution:, 0.104243,0.0876729,0.132305,0.166189,0.0812033,0.103515,0.0634545,0.0463774,0.0723966,0.0843561,0.0538798,0.0044074

Predict, please wait...
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_1.tif-band1
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_2.tif-band2
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_3.tif-band3
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_4.tif-band4
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_5.tif-band5
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_6.tif-band6
Output file: D:/Guinea_Methodology/landuse_experiments/outputs/benchmark/output_plus/leas_result_band_7.tif-band7
 total time for one block: 78.109 s

---=== Finish! ===---

CARS: neighborhood size 3, thread 1, land use patern landuse_2023, development potential are the maps from LEAS, conversion constraint emptied. patch generation threshold 0.2, expansion coefficient 0.1, percentage of seeds 01. 
you may check the tutorial guide from the github: https://github.com/HPSCIL/Patch-generating_Land_Use_Simulation_Model/blob/master/Tutorials_Eng/Tutorial%20A%EF%BC%9A%20User%7FManual%20of%20the%20PLUS%20model%20%2020250531.pdf

CARS processing crashes, will try in another computer or troubleshoot.

### Fixing the CARS crash

Four settings above differ from what CARS needs:

1. **Land-use pattern: `landcover_2024.tif`, not 2023.** The forecast starts from 2024.
2. **Probability of development: `output_plus/leas_probability_7band.tif`.** LEAS wrote seven
   separate one-band files (`leas_result_band_1..7.tif`); CARS expects one file with one band per
   class. The 7-band file stacks them in class order.
3. **Cost (conversion) matrix: fill all 7 × 7 cells with 1.** Left empty, CARS has nothing to read.
4. **Future land demand: the `pixels_2025_observed` column** of `legend_and_demand.csv`, one
   number per class. The only CARS crash reported on PLUS's GitHub
   ([issue #1](https://github.com/HPSCIL/Patch-generating_Land_Use_Simulation_Model/issues/1))
   went away when the demand was filled in.

Keep the rest as it was (neighbourhood 3, patch threshold 0.2, expansion coefficient 0.1, default
seed percentage). If it still crashes, note the last message in the CARS window.

Optional, for a fairer PLUS: rerun LEAS with a sampling rate of about 0.2 instead of 0.01. At 0.01
it trained on 4,310 points, with only 3 to 37 positives for the small classes.

**Status (2026-10-23): CARS ran.** It saved `output_plus/output_flusSimulation_1.tif` (the output path
typed was `output_flus.tif`; CARS adds `Simulation_1`). Settings: the 2024 map, the 7 LEAS bands,
neighbourhood 3, patch generation threshold 0.2, expansion coefficient 0.1, percentage of seeds 0.1,
demand = `pixels_2025_observed`. It stopped when its deviation from the demand no longer fell
(15,008 pixels: Water 17,051 against 24,555, Bare ground 1,734 against 648). Screenshot:
`output_plus/CARS_processing.png`.

## Tool 3: Dinamica EGO -- not working

Skipped: the download site was unreachable (2026-10-22).

## Tool 4: CLUE-S
https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/RLDNDX
download tool in: https://dataverse.nl/file.xhtml?fileId=408332&version=1.0
but i couldnt download it, im not sure why and how.

**Status (2026-10-22): run.** The file downloads through Dataverse's API
(`https://dataverse.nl/api/access/datafile/408332`, a 13 KB `.7z`; Windows' own `tar -xf`
opens it). It is the C++ source of Dyna-CLUE 2.7 (Verburg & Overmars 2009, CC0), written for
Borland C++Builder, so there is no program to run and it cannot be built with today's compilers.

Its allocation routine was therefore ported line by line to Python (`clue_s.py`), and run on the
same inputs as the other tools:

* one logistic regression per class, on the 2024 map and the 2024 predictors;
* conversion elasticity of each class = the share of its pixels that stayed the same in 2023→2024;
* every conversion allowed; no neighbourhood option (standard CLUE-S);
* demand = the observed 2025 total of each class, as Pontius et al. (2008) gave every model.

An independent check is possible with the R package `lulcc` (Moulds et al. 2015), which has its own
CLUE-S implementation (`CluesModel`). It needs R installed.

**Update (2026-10-23): the original program found and run.** DataverseNL also holds the authors'
compiled program ("CLUE Full Version", `clues-1.exe`, doi:10.34894/CXAUYO; its Borland runtime
DLLs come with the "Demo version of CLUE", doi:10.34894/ZDGY5V; manual doi:10.34894/FD1COL). It runs
in batch mode (`clues <demand file> <region file>`, manual p. 30). `clue_s.export_original()`
writes our inputs in its formats, with the same regressions as the port, and `clue_s.read_original()`
reads its map back. lulcc is run with `Rscript scripts/benchmark_lulcc.R` (R 4.6.1 in `D:/envs/R`, packages
in `D:/envs/R-library`).

**Used in the comparison (2026-10-23): the lulcc run**, so the CLUE-S column comes entirely from
published R code (the regressions too). The original program and the Python version are kept as
checks. How to rerun it and the run log: see the top of `scripts/benchmark_lulcc.R`.

## Tool 5: GeoSOS-FLUS (free, Windows)

FLUS (Liu et al. 2017) is the predecessor of PLUS from the same group: a neural network learns where
each class is found, then cellular automata with "self-adaptive inertia" allocate the demand.
The program and its manual are already in `outputs/benchmark/tools/GeoSOS-FLUS V2.3/`
(downloaded from http://www.geosimulation.cn/FLUS.html). Start `UGB-FLUS V2.3_boxed.exe`.
Save everything in `outputs/benchmark/output_flus/`.

1. **FLUS Model → ANN-based Probability-of-occurrence Estimation**
   * Land Use Data: `inputs/landcover_2024.tif`. *Set NoData Value*: set the value **0** to "NoData Value".
   * ANN Training: Random Sampling; sampling rate **20** (thousandths, so 2% of the pixels);
     hidden layer **12** (the manual's value).
   * Save path: `output_flus/Probability-of-occurrence.tif`, Single Accuracy.
   * Driving Data: the 12 files in **`inputs/factors_2024_flus/`** (already scaled to 0–1), and
     choose **"No Normalization"**.
   * Run. Keep the text in the right-hand box (training accuracy).
2. **FLUS Model → Self-adaptive inertia and competition mechanism CA**
   * If `config_color.log` or `config_mp.log` exist next to the program, delete them first
     (manual, section 4: they hold another study area's settings).
   * Land Use Data: `inputs/landcover_2024.tif`; set **0** as NoData.
   * Probability Data: `output_flus/Probability-of-occurrence.tif` from step 1.
   * Save Simulation Result: `output_flus/flus_2025.tif`.
   * No restricted area.
   * Land Use Demand, "Future Pixel Number": the `pixels_2025_observed` column of
     `legend_and_demand.csv` (class 1 to 7, in that order).
   * Cost Matrix: **1 in every cell**.
   * Weight of Neighborhood: **1 for every class** (no expert tuning).
   * Iterations 300, neighbourhood 3, acceleration factor and threads at their defaults.
   * Run from the "Show" tab and wait until it stops and saves.
3. Send back `Probability-of-occurrence.tif`, `flus_2025.tif`, and the log files FLUS wrote
   (`ANNoutput.log`, `output.log`, `logFileSimulation.log`).

**Status (2026-10-23): run.** ANN training RMSE 0.21 (`output_flus/ann_rmse.png`); the CA met the
demand after 9 iterations, Water within 93 pixels (`output_flus/simulation_result.png`).

## Extra runs for the next steps (FLUS and PLUS)

Same settings as above unless stated. Each run goes in its own subfolder. Keep a screenshot of
each finished window.

**A. FLUS run-to-run variation (next step 2): two more full runs.** Repeat steps 1 and 2 of tool 5
twice, unchanged; the network samples its training pixels at random, so each run differs.
Save in `output_flus/run2/` and `output_flus/run3/`, with the same file names
(`Probability-of-occurrence.tif`, `flus_2025.tif`).

**B. FLUS on the 2023 map (so that averaging FLUS's maps with our method can be chosen on
2023→2024, notebook 09).** Step 1 only: Land Use Data `inputs/landcover_2023.tif`, Driving Data the
12 files in **`inputs/factors_2023_flus/`**. Save `output_flus/2023/Probability-of-occurrence.tif`.

**C. Projected amounts (next step 3, notebook 10).** Rerun only the simulation step, with the
`pixels_2025_projected` column of `legend_and_demand.csv` as the demand, in classes 1 to 7:

| Water | Trees | Flooded | Crops | Built | Bare | Rangeland |
|---:|---:|---:|---:|---:|---:|---:|
| 21,048 | 2,028,750 | 29,975 | 58,178 | 40,427 | 1,048 | 1,831,250 |

* **FLUS:** step 2 of tool 5, Probability Data = the existing `output_flus/Probability-of-occurrence.tif`.
  Save `output_flus/projected/flus_2025_projected.tif`.
* **PLUS:** CARS as before (the same 7 LEAS bands and settings), Future Amounts = the table above.
  Save as `output_plus/projected/plus_2025_projected.tif`. CARS adds `Simulation_1` to the name
  typed, so type `plus_2025_projected` and rename the file afterwards if needed.

Notebook 10 adds the PLUS and FLUS columns when these files exist.

**Status (2026-10-23): A, B and C all run.** PLUS saved its projected map as
`output_plus/projected/plus_2025_projectedSimulation_1.tif` (62 pixels off the demand); FLUS's
projected run met the demand except Bare ground (850 against 1,048).

## What to send back

For each run:

1. The forecast map (`*_2025.tif`), and for MOLUSCE the transition potential maps
2. The method and any settings you changed
3. The runtime, and any errors or warnings

## How the forecasts will be scored

Every forecast gets the validation set the literature recommends (`benchmark_scoring.py`,
notebook 00):

| What is compared | Metric | Source |
|---|---|---|
| Each tool as it runs | **Figure of Merit (all change)**, the headline; producer's and user's accuracy | Pontius et al. 2008 |
| | Quantity disagreement and allocation disagreement | Pontius & Millones 2011 |
| | Times better than random placement of the same amount of change | Pontius et al. 2004 |
| | Figure of Merit at 1 km and 5 km | Pontius et al. 2004, 2008 |
| Where change goes, with the same amount for every tool | Figure of Merit, placing the observed amount of each transition with the tool's own suitability maps and the deployed allocator | like-for-like |
| Suitability maps, whatever the amount | AUC per transition, weighted by observed change | Pontius & Si 2014 |
| Compatibility only | Overall accuracy and kappa, as MOLUSCE reports them; not used to rank | Pontius & Millones 2011 |

**Difference in quantity:**

* **Deployed method:** gets the observed number of each transition.
* **PLUS and CLUE-S:** get the observed total of each class.
* **MOLUSCE:** extrapolates the 2023→2024 rates, capped by its model (see above).

The like-for-like row removes these differences.

## Other tools

The 13 applications compared by Pontius et al. (2008) used 9 models. Status of each for this test:

| Model | Status |
|---|---|
| CLUE, CLUE-S | Run: R package lulcc (in the comparison); the authors' original program and `clue_s.py` as checks |
| Logistic regression | Run (MOLUSCE LR, and inside the deployed method, notebook 08) |
| Geomod | Inside TerrSet (paid); models one transition only |
| Land Transformation Model (LTM), SLEUTH | Urban growth only, not multi-class land cover |
| Environment Explorer, Land Use Scanner | Commercial, built for the Netherlands |
| SAMBA | Agent-based, built for one Vietnamese district |

Widely used tools since then:

| Tool | Status |
|---|---|
| MOLUSCE (QGIS) | Run |
| PLUS | Run (LEAS and CARS) |
| Dinamica EGO | Site unreachable |
| FLUS / GeoSOS-FLUS | Run (tool 5) |
| TerrSet Land Change Modeler, CA–Markov | Paid licence |
| `lulcc` (R) | Run: its CLUE-S is the CLUE-S column (tool 4) |
