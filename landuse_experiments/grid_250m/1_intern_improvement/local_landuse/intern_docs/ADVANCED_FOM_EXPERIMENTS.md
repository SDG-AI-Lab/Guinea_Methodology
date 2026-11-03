# Further experiments targeting 40% Change FoM

**The target was not reached.** The model chosen by earlier validation achieves
**26.24%** destination-sensitive FoM on 2015→2020, leaving **13.76 percentage
points** to 40%. The highest final score among the 16 candidates is **26.49%**,
but that candidate did not win earlier validation.

Open [advanced_fom_experiments.ipynb](advanced_fom_experiments.ipynb) for the
executed tables, maps, target chart and error diagnostics. The authoritative
combined results are in `outputs/fom40_summary_20260909/summary.json`.

## Results and trade-offs

| Configuration | Final Change FoM | Final accuracy | Pixels predicted to change |
|---|---:|---:|---:|
| Earlier multiclass RF baseline | 20.91% | 90.72% | 6.84% |
| Earlier XGBoost + history/neighborhoods | 24.03% | 91.31% | 6.84% |
| History RF + selected change thresholds | 25.60% | 89.63% | 10.31% |
| Two-stage RF, **validation-selected** | **26.24%** | **88.92%** | **11.84%** |
| Weighted XGBoost + patch features, best final row | 26.49% | 89.27% | 11.27% |

The same 992,833 valid pixels are evaluated throughout; observed map-label
change is 7.59%. Classes, map resolution and evaluation masks were not changed.
These results remain retrospective agreement with supplied maps, not independent
field validation. 2020 was inspected in previous experiments.

The selected RF has two stages: predict change versus persistence on the shared
training pool, then predict destination on the changed rows of that pool.
Each forest uses 150 trees, maximum depth 16, minimum split size 5, minimum leaf
size 2, `max_features="sqrt"`, seed 42 and four workers. It uses 132 features.
Its validation FoM is 25.84%; final FoM on the previously held-out blocks alone
is 27.76%. The full-footprint final score includes training-block locations at
a later date, and remains the main comparison metric.

The selected model correctly predicts **39,154 changes**, misses **31,614**,
predicts **73,824 false changes**, and assigns **4,597 wrong destinations** to
pixels that really changed. Correct-destination change precision is about 33.3%
and recall about 52.0%. Higher FoM has come with lower overall accuracy.

As a diagnostic, even fixing every wrong destination at the currently predicted
change locations would yield only **29.33% FoM**. This uses final labels and is
an oracle calculation, not a model result. Reaching 40% requires better detection
of **where change occurs**, including fewer false alarms; destination assignment
alone cannot close the gap at these locations.

## What was implemented

- **Three decision scores:** strongest alternative-destination probability,
  probability of any change, and alternative-destination probability divided
  by persistence probability. Each gets a 65-value threshold search on validation.
- **Source-specific thresholds:** a guarded coordinate search improves the same
  full-mask validation FoM. Sources need at least 1,000 validation pixels and
  100 observed changes; rarer sources retain the global threshold. This can
  overfit one validation period, so later-period results determine transfer.
- **Changed-pixel weighting:** XGBoost variants give changed rows four times
  the fitting weight. A known-cost inverse adjustment is applied to probabilities;
  this is not a claim of empirical probability calibration.
- **Two-stage RF and XGBoost:** a change detector plus a conditional destination
  classifier; the latter is fitted only on changed rows.
- **Source-specific XGBoost:** source experts use their source subset of the
  shared pool, with a global model fallback when support is insufficient.
- **Landscape features:** patch area, distance to patch edges and land-cover
  classes, terrain neighborhood statistics, and relative map coordinates,
  alongside the previous history and multiscale features. All use information
  available in the starting/past maps, never future destination labels.
- **Relative-action models:** classify stay, return to previous class, or change
  to a specified other class. Predicted actions map back to the original land-cover
  codes; no classes are merged for evaluation.
- **An ensemble:** average the joint geometry XGBoost and two-stage XGBoost
  destination probabilities, then select its decision rule on validation.
- **More training pixels:** two separately labeled candidates expand the primary
  67,938 rows to 300,000 within the same training blocks. The original rows are
  retained and no held-out block is added. This did not produce a large gain.

Every candidate also produces a **fixed-demand map** using the previous common
allocator and the same training-derived transition quantities. Those maps still
change 6.84% of pixels. The selected two-stage RF gets 23.73% FoM under that
constraint; the best new fixed-demand row is the ensemble at 24.24%, only 0.21
points above the previous 24.03% result. Most of this round's headline gain comes
from changing the decision policy and quantity rather than a large score-model gain.

## Selection protocol

Fit 2005→2010 in the 95 training blocks, validate 2010→2015 in the 41 held-out
blocks, and choose model/thresholds by validation FoM. Then refit 2010→2015 and
predict 2015→2020 with those choices frozen. The model and policy selection rule
across both candidate batches is maximum earlier validation FoM, with validation
accuracy then name as deterministic tie-breakers. Final metrics do not select it.

The original 100×100-pixel block scheme and shared primary sample are retained.
Historical lag is four years for initial fitting and five years thereafter;
rates are annualized and lag duration is a feature. No extra geographic buffer
is added between blocks. The exact candidate settings are saved in each run's
`candidates.json`; each policy is saved in `frozen_selection.json`.

## Reproduction and verification

From `Land Use-Deforestation`:

```bash
../../.venv/bin/python -m local_landuse.advanced --output-dir outputs/my_advanced_run
../../.venv/bin/python -m local_landuse.actions --output-dir outputs/my_action_run
../../.venv/bin/python -m local_landuse.advanced_summary --runs outputs/my_advanced_run outputs/my_action_run --output-dir outputs/my_summary
../../.venv/bin/python -m pytest tests -q
```

Training directories must be new/empty. Dependencies are already listed in
`requirements-local.txt`; no new dependency or network service was needed for
this round. RF and XGBoost parameters follow their
[official RF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [official XGBoost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).

During verification, explicit all-one RF sample weights were found to change
bootstrap draws. The code now passes no weights for that unweighted case. The
three affected RF candidates were rerun in `outputs/rf_corrected_fom40_20260909`;
the combined report supersedes their initial rows. The corrected history control
reproduces the previous history RF fixed-demand raster exactly. Future runs
already include the fix and do not need a correction run.

The initial module entry point also serialized custom model classes under
`__main__`. Entry points now use canonical module identities. Portable copies
of all 15 fitted models (the ensemble uses its members) are saved under
`outputs/fom40_summary_20260909/models`, with decision policies and source-run
metadata. The first-run artifacts remain preserved for audit.

Tests cover FoM arithmetic including wrong destinations, valid two-stage/action
probabilities, bootstrap equivalence, missing-neighbor features, threshold
selection, expanded-sample block boundaries, and invariance of model selection
and forecasts when final labels are replaced. Saved model loading was checked
in a separate Python process.

## What the remaining gap calls for

The experiments do not show that 40% is impossible. They show that these local
maps, predictors and tested algorithms have not achieved it. The next substantive
work is better historical evidence about change locations: independently checked
change labels/source QA, dated vegetation and disturbance histories, and dated
access/agricultural/mining covariates. These are data-acquisition experiments,
not gains demonstrated here. The undated local covariates remain excluded.

Avoid selecting settings against the 2020 result to force the target. Keep a
future evaluation independent; using the bundled 2024 map requires explicit
handling of the four-year 2020→2024 horizon. No 2024 experiment was run here.
