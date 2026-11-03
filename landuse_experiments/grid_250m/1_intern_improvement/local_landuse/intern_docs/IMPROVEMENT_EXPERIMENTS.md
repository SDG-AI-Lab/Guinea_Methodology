# Local land-use improvement experiments

For the next round targeting 40% FoM, see
[ADVANCED_FOM_EXPERIMENTS.md](ADVANCED_FOM_EXPERIMENTS.md) and
[advanced_fom_experiments.ipynb](advanced_fom_experiments.ipynb).

Open **[improvement_experiments.ipynb](improvement_experiments.ipynb)** for the
executed comparison. The saved run is `outputs/improvements_20260909`.
Seven model/feature configurations and 24 allocation settings were evaluated
locally. Earlier experiments and source scripts were preserved.

## Results on 2015→2020

All rows use the same 992,833 valid evaluation pixels. Observed map-label change
is 75,365 pixels (7.59%). FoM measures correctly predicted change divided by the
union of observed and predicted change; a wrong destination counts as an error.

| Configuration | Change FoM ↑ | Accuracy ↑ | Pixels predicted to change |
|---|---:|---:|---:|
| No change | 0.00% | 92.41% | 0.00% |
| Previous multiclass RF baseline | 20.91% | 90.72% | 6.84% |
| RF + history **(validation-selected model)** | 22.89% | 91.10% | 6.84% |
| RF + history + larger neighborhoods | 23.61% | 91.23% | 6.84% |
| Regularized RF + all features | 23.41% | 91.19% | 6.84% |
| Flexible RF + all features | 22.38% | 90.99% | 6.84% |
| Shallow XGBoost + all features | 24.03% | 91.31% | 6.84% |
| Deeper XGBoost + all features | 23.70% | 91.25% | 6.84% |
| Selected RF + separately tuned allocation | 24.16% | 90.36% | 8.53% |

The history RF gains **1.98 percentage points of FoM** over the baseline.
Shallow XGBoost gains **3.12 points (14.9% relative)**, but was not the winner
on earlier validation data. It is a promising follow-up candidate, not a model
chosen independently of these final results. Its gain over the RF with the same
101 features is only **0.42 points**: most of the total gain comes from features.
There is an improvement, but no demonstrated large breakthrough or statistical
significance across periods/seeds.

The allocation variant was selected on validation data: multiply historical
transition demand by 1.25 and require a destination score of at least 0.10.
It predicts 84,648 changes instead of 67,954. Compared with the history RF using
unchanged allocation, correctly located changes with the right destination rise
from 26,229 to 30,502, while false changes rise from 39,216 to 50,884.
Higher FoM here trades against overall accuracy. A score threshold is not a
calibrated confidence guarantee.

## What was held constant and what changed

All candidates share the **same 67,938 training rows, 95 training blocks,
41 held-out blocks, 100×100-pixel block scheme, seed 42, class vocabulary,
terrain inputs, transition support and sample-based demand**. The default
allocator gives every candidate exactly the same per-transition quantities.
The new baseline's training indices and entire final prediction raster were
verified equal to the previous controlled multiclass experiment.

The feature sequence makes the changes explicit:

1. **Baseline, 29 features:** elevation/slope/aspect, starting class indicators,
   and 3×3 class fractions; RF 100 trees, depth 20, split minimum 5, leaf minimum 2.
2. **History, 47 features:** also previous class, whether the pixel changed,
   historical availability/interval, nearby previous change rate, and distance
   to previous change. RF settings stay identical.
3. **Multiscale history, 101 features:** also 7×7 and 15×15 class fractions,
   their historical trends, and surrounding change rates. RF settings stay identical.
4. **Alternative learners on those same 101 features:** RF 250 trees with either
   depth 12/leaf 10/split 10 or unlimited depth/leaf 2/split 5 (both max features
   0.7); XGBoost 180 trees/depth 3 or 150 trees/depth 6, learning rate 0.07,
   minimum child weight 5, row/column subsampling 0.8, L2 regularization 5.

RF uses ordinary multiclass classification; XGBoost uses `multi:softprob` and
multiclass log loss. Both supply destination probabilities to the common
allocator. **FoM selects configurations on validation maps; neither learner
directly optimizes FoM during fitting.** Class weights remain equal.

These are extensions to the controlled local comparison, not an exact rerun of
the API implementation. See [CONTROLLED_COMPARISON.md](CONTROLLED_COMPARISON.md)
for the deliberate shared-sampling and local-allocation differences.

## Selection and temporal safeguards

- Initially fit on **2005→2010** in training blocks, using history from 2001→2005.
- Validate forecasts of **2010→2015** in held-out blocks, using history from
  2005→2010. Select model and then allocation settings by validation FoM.
- Write `frozen_selection.json` before final prediction/evaluation.
- Refit each predeclared candidate on **2010→2015**, using 2005→2010 history.
- Predict **2015→2020**, using 2010→2015 history; only then load 2020 labels.

The selected model's validation FoM was 20.58%, versus baseline 18.81%.
The selected allocation raised validation FoM to 22.05%.
All forecast intervals are five years. The initial historical lag is four years,
while later lags are five: neighborhood trends are annualized and interval length
is a feature, but this does not remove every effect of unequal lag durations.

Final evaluation covers the full valid footprint, including training-block
locations at a later date. The block scheme has no extra geographic buffer.
2020 was inspected in earlier experiments, so these are **retrospective
comparisons**, not results from a previously untouched test set. No final metric
was used to select this run's winner, and no independent field reference was used.
The bundled 2024 map was not used; a future evaluation there needs explicit
handling of its four-year forecast horizon.

## Label audit and unavailable predictors

Among pixels that changed class from 2005→2010, 37.89% returned to their 2005
class in 2015. For changes from 2010→2015, 32.99% returned by 2020. These audit
counts use all valid land-cover pixels, including some without valid terrain;
their denominator differs from the model evaluation mask. Reversals can reflect
real land-cover cycles or classification instability, so **no labels were edited**.

Local notebook descriptions identify MODIS/IGBP inputs; raster bands are
`LC_Type1` on a roughly 500 m grid. The exact collection version is not recorded.
The [MODIS Collection 6 guide](https://modis.ornl.gov/documentation/guides/MCD12_User_Guide_V6.pdf)
cautions against interpreting annual class differences directly as land-cover
change. Source QA layers or independent imagery are needed to assess errors.

The local population layer is dated 2020; roads, POIs, protected areas, biomass
and soil moisture lack sufficiently verified historical dates for these fits.
They were excluded. Better dated predictors and independently checked labels
remain hypotheses for further improvement, not gains demonstrated in this run.

## Reproduce and inspect

From `Land Use-Deforestation`, using the workspace Python environment:

```bash
../../.venv/bin/python -m pip install -r requirements-local.txt
../../.venv/bin/python -m local_landuse.improvements --output-dir outputs/my_improvement_run
../../.venv/bin/python -m pytest tests -q
```

Use a new/empty output directory. After dependencies are installed, fitting and
prediction require no network or remote geospatial service. Notebook rerunning
loads saved results by default; set `RETRAIN=True` to generate a new run.

The run stores model specifications, input hashes, training indices, validation
scores, the frozen choice, allocation ledgers, trained models, prediction
GeoTIFFs, final metrics and error maps. Tests cover baseline-feature and allocator
equivalence, threshold behavior, validation-only selection, immutable labels,
and invariance of selection/predictions when final labels are replaced.
