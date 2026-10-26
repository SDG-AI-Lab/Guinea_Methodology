# Historical satellite features for five-year change forecasting

**The 40% target was not reached.** The final validation-selected class-specific
ensemble scores **27.97% FoM** on 2015→2020, with **91.10% overall accuracy**.
The strongest individual forecast row is **28.00%**, but its model did not win
the combined validation comparison. All results use the same 992,833 pixels.

Open [satellite_feature_experiments.ipynb](satellite_feature_experiments.ipynb)
for the tables, maps and saved-model verification. Individual-model results are
in `outputs/satellite_summary_20260910/summary.json`; final ensemble selection
and results are in `outputs/source_ensemble_20260910/report.json`.

| Configuration | Spatial validation FoM | Forecast FoM | Forecast accuracy | Predicted change |
|---|---:|---:|---:|---:|
| Original-style transition RF, local reproduction | Different sampling | 20.02% | 90.28% | 7.28% |
| Original-style binary RF, shared samples/blocks | Different selection protocol | 20.29% | 90.60% | 6.84% |
| Initial multiclass RF, shared samples/blocks | Different selection protocol | 20.91% | 90.72% | 6.84% |
| Same-protocol RF control | 30.40% | 26.93% | 90.19% | 9.80% |
| Selected seasonal MODIS model | 35.75% | 27.50% | 91.08% | 8.40% |
| Selected MODIS + neighborhood model | 36.06% | 28.00% | 90.63% | 9.40% |
| Selected Landsat model | 34.38% | 26.16% | 90.70% | 8.64% |
| Combined observations, selected individual | 36.24% | 27.87% | 91.00% | 8.70% |
| Class-specific ensemble, final selection | 36.37% | **27.97%** | **91.10%** | **8.54%** |

The selected ensemble correctly predicts 34,292 changes, misses 37,840,
predicts 47,254 false changes, and assigns 3,233 wrong destinations. Observed
change is 7.59%. Compared with the same-protocol RF control, FoM improves by
1.03 percentage points and accuracy by 0.91 points. It remains 12.03 FoM points
short of the target. Better spatial validation did not transfer fully to the
later period. No final-year threshold tuning was used to close that gap.

Against the original-style local transition RF, the ensemble improves FoM by
7.95 points; against the shared-sample/block binary RF, it improves by 7.68
points. The former is a local reproduction with valid-pixel and single-conversion
safeguards, not an untouched execution of the cloud backend. The controlled
binary baseline incorporates the earlier agreed changes to training samples.
Both remain in each run's `comparison_with_original.csv` and `.json`.

The selected individual combined model scores 25.68% with the common fixed
demands, versus 23.73% for the control. The class-specific ensemble combines
threshold-selected forecasts and makes no fixed-demand claim.

This round adds historical satellite observations to the existing local models.
The target remains **destination-sensitive Change FoM ≥40% on the complete
2015→2020 evaluation area**. Model selection uses 2010→2015 spatial validation.
Validation improvements alone do not establish that the target was reached.

## Evaluation protocol

- Keep the original 100×100-pixel blocks: 95 training blocks and 41 validation
  blocks, with the same primary 67,938 training pixels.
- Fit 2010→2015 in training blocks; choose model and change thresholds using
  2010→2015 in validation blocks. Predict 2015→2020 with the fitted model and
  choices frozen. This differs from the preceding round's earlier-period
  temporal validation and must be labeled separately.
- Evaluate the same 992,833 pixels and original class codes. Missing satellite
  observations become explicit missing-value/support features, never a smaller
  evaluation mask. There are 75,365 observed changes (7.59%).
- Expanded candidates retain the primary pixels and add pixels only within
  existing training blocks, up to 300,000. Pooled candidates also use the prior
  2005→2010 training period. These are larger-data experiments, not algorithm-only
  comparisons with identical rows.
- Save a paired fixed-demand prediction for each model using the same primary
  training-derived transition quantities and common allocator. The other map
  uses validation-selected change thresholds.

2020 labels are loaded after every candidate's forecast is saved. The target was
examined in previous work, so these are retrospective comparisons, not a pristine
test. Agreement with supplied map labels is not independent field validation.

## Implemented approaches

**Longer land-cover history.** Two previous intervals, return-to-older-class
indicators, observed persistence duration, and directional historical transition
densities in 7-, 21-, and 51-pixel neighborhoods. The completed nine-candidate
history-only round selected pooled joint XGBoost: spatial validation FoM 31.53%,
but forecast FoM only 25.99%. Longer history alone did not improve transfer.
Its control RF scored 30.40% on spatial validation and 26.93% on the forecast.
Results: `outputs/temporal_memory_20260909/report.json`.

**Seasonal MODIS vegetation.** Twelve Terra MOD13A1.061 composites per year/tile,
NDVI and EVI, for 2005, 2010 and 2015. Every composite starts and ends within
its requested year. Native quality flags exclude poor retrievals, high aerosols,
cloud contamination, snow and shadow before nearest-neighbor reprojection.
Features include individual seasonal observations, missingness, annual and
wet/dry-season statistics, a fitted annual harmonic, previous observations and
five-year trends. Four tiles × twelve composites were acquired for each year.

| Observation year | At least one usable NDVI period | At least six usable periods |
|---|---:|---:|
| 2005 | 99.91% | 96.02% |
| 2010 | 99.91% | 94.90% |
| 2015 | 99.89% | 94.81% |

**Vegetation neighborhood context.** Starting-year NDVI/EVI annual, wet-season
and dry-season means in 3×3 and 15×15 windows, neighborhood variability, local
contrast, support, and neighborhood trends. Neighbor calculations exclude missing
observations. They use no future land-cover or future satellite measurements.

**Dry-season Landsat.** January–April Landsat 7 (2005), Landsat 5 (2010) and Landsat 8 (2015),
up to two low-cloud scenes per path/row. Features are red, near-infrared and two
shortwave-infrared reflectances, NDVI, NDMI and NBR, scene variability, clear
observation count, older observations, trends, and neighborhood statistics.
The archive returned no Landsat 5 scenes for Guinea in 2005, so acquisition
falls back to same-year Landsat 7. For Landsat 5/8, native QA is converted to a
clear-area fraction; cells need ≥99% clear support. Reflectance is averaged to
the analysis grid separately, so that method is not an exact native-pixel
QA-masked reflectance average. For Landsat 7, native gaps and clouds are masked
**before** reflectance averaging and cells need ≥50% observed clear area. No
reflectance values are interpolated into scan-line gaps. Sensor and aggregation
differences remain possible sources of temporal transfer error. The full
land-cover evaluation mask is unchanged for every year and method.

**Combined observations.** A separate runner combines seasonal MODIS,
neighborhood vegetation features and Landsat, keeping their missing-value flags.

**Class-specific ensemble.** After the 28 individual candidates, a deterministic
validation-only search assigns an already fitted member to each starting class.
Each swap must improve full validation FoM (accuracy breaks ties); it does not
maximize an unweighted mean of class scores. Sources with fewer than 1,000
validation pixels or 100 observed changes keep the global winner. Choices are
saved before ensemble forecast maps or 2020 observations are read.

RF and XGBoost candidates predict the next class directly, use separate
change/destination stages, or predict stay/return/other-destination actions.
Exact hyperparameters, feature names, row hashes and policies are saved with
each model and in its run manifest. All feature observations precede or belong
to the forecast's starting year; these are retrospective reprocessed products,
not a claim that these exact product versions were published in that year.

## Run locally

Run commands from this directory with the repository's existing environment.
Acquisition uses public Microsoft Planetary Computer assets without Earth
Engine credentials. Downloaded scenes are cached; rerunning acquisition resumes
from completed scenes. Training output directories must be new/empty.

```bash
../../.venv/bin/python -m local_landuse.modis_phenology --output-dir outputs/historical_modis --years 2010 2015 2005
../../.venv/bin/python -m local_landuse.landsat_features --output-dir outputs/historical_landsat --years 2010 2015 2005
../../.venv/bin/python -m local_landuse.phenology_models --phenology-dir outputs/historical_modis --output-dir outputs/my_phenology_run
../../.venv/bin/python -m local_landuse.vegetation_context --phenology-dir outputs/historical_modis --output-dir outputs/my_context_run
../../.venv/bin/python -m local_landuse.spectral_models --spectral-dir outputs/historical_landsat --output-dir outputs/my_spectral_run
../../.venv/bin/python -m local_landuse.vegetation_context --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --output-dir outputs/my_fused_run
../../.venv/bin/python -m local_landuse.temporal_summary --runs outputs/my_phenology_run outputs/my_context_run outputs/my_spectral_run outputs/my_fused_run --output-dir outputs/my_satellite_summary
../../.venv/bin/python -m local_landuse.source_ensemble --runs outputs/my_phenology_run outputs/my_context_run outputs/my_spectral_run outputs/my_fused_run --output-dir outputs/my_source_ensemble
../../.venv/bin/python -m pytest tests -q
```

Training and prediction work offline once the source composites are cached.
Saved `.joblib` bundles contain the fitted model, frozen decision policy, exact
feature names and training provenance. Source catalogs and composite hashes
remain in each acquisition directory. Large generated artifacts are git-ignored.

Landsat acquisition completed all requested scenes: 34/34 in 2005 (93.18%
coverage), 35/35 in 2010 (97.21%), and 38/38 in 2015 (99.22%). Satellite coverage
describes available predictors; it never defines the evaluation footprint.

Verification: **34 tests pass**, including future-label invariance for the
temporal runner and source ensemble. All seven notebook code cells executed.
The notebook loads the selected individual model in a fresh kernel and exactly
reproduces 5,000 saved forecast rows, reconstructs the complete ensemble map,
and checks the RF control against the preceding spatial-protocol run. The
evaluation mask and primary sample also exactly match the original controlled
comparison's temporal evaluation and training sample.

Source documentation:
[Planetary Computer STAC access](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/),
[MOD13A1 bands and QA](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A1),
[Landsat Collection 2 quality bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands),
[Landsat 7 scan-line gaps](https://www.usgs.gov/landsat-missions/landsat-7).
