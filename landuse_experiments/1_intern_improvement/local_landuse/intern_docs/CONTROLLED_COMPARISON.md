# Controlled RF comparison: shared spatial split and allocation

For subsequent history-feature, RF-tuning and XGBoost experiments, see
[IMPROVEMENT_EXPERIMENTS.md](IMPROVEMENT_EXPERIMENTS.md) and
[improvement_experiments.ipynb](improvement_experiments.ipynb).

Open **`controlled_model_comparison.ipynb`**, select the workspace
`.venv/bin/python` kernel, and Run All. Saved local results are included in
`outputs/controlled_2010_2015_2020`; set `RETRAIN=True` to create another run.
No network, Earth Engine or Azure services are required.

This experiment isolates **separate binary transition forests versus a single
multiclass destination forest**. The multiclass forest supplies suitability
scores to the **same allocation function**; its argmax does not select the
final map class in this experiment.

## What is held constant

- The same local source maps: calibrate 2010→2015; forecast 2015→2020.
- The same predictor grid and feature columns: elevation, slope, aspect,
  one-hot initial class and 3×3 initial-map class fractions. All class fractions
  are recomputed from the starting map of the relevant period.
- One uniform pool of 100,000 valid calibration pixels, sampled without replacement.
- One **100×100-pixel block** assignment, using the original code's
  `(row // 100) * 1000 + column // 100` IDs and
  `GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)`.
- Exactly the same training rows/order and held-out sample rows for every forest.
  The first run has **67,938 training pixels in 95 blocks**, and **32,062 test
  pixels in 41 blocks**. Thirty percent refers to blocks, not necessarily pixels.
- Equal weights; 100 trees, depth 20, minimum split size 5, minimum leaf size 2,
  seed 42 and four workers. The same row count/order/seed gives matching bootstrap
  draws across models. There is no tuning or additional random five-fold CV.
- The same supported transition list, estimated historical demands, demand order,
  eligibility mask, deterministic allocation and evaluation masks.

Only the targets/model formulation differ. Each binary forest learns
`(source == A and destination == B)` versus all other outcomes. The multiclass
forest learns the destination code directly; for A→B suitability it supplies
`P(destination=B)` on source-A pixels. Persistence remains a multiclass target.

## Relationship to the original implementation

We adopt the original block split scheme and RF defaults. As agreed, the shared
row pool replaces the original per-transition balanced A→B/A→A samples: a single
multiclass model cannot retain all those separate samples and splits exactly.
The binary target consequently uses all other outcomes as negatives. Both sides
use the existing **local** common allocator, which retains the validity and
single-conversion safeguards added in the first local implementation.

This is not a byte-for-byte reproduction of the old API notebook. That notebook's
ten-layer predictor stack is unavailable locally; this experiment holds the
local feature set constant between its two sides. The supplied rasters are
roughly 500 m, so a 100-pixel block has a different ground size from the original
250 m API configuration. Legacy source scripts and prior outputs remain intact.

## Demand and evaluation

Historical transition counts are taken **only from shared training rows**.
For each forecast footprint, demands are expanded as:

`training transition count × eligible forecast pixel count / training sample size`

The calibration and forecast durations are both five years. Identical frozen
demand dictionaries go to both models' allocation calls. A supported transition
needs at least ten sampled positives and negatives and an estimated full-grid
calibration demand of at least 200 pixels. The same support is imposed on both
methods; low-count exclusions are recorded in `transition_training.csv`.

The spatial test predicts 2015 on **all eligible pixels in held-out blocks**,
including pixels not selected for the 100,000-row sample. Scores on just the
32,062 held-out sample rows are also recorded. Neither held-out destination
labels nor 2020 labels inform model fitting, transition selection or demand.

The temporal test applies the frozen models to the full valid 2015 footprint
and evaluates the resulting 2020 map. This includes locations whose blocks were
used for calibration. A separate view reports performance only in the previously
held-out blocks; it evaluates the same full-footprint allocation, not a new map.

Both models allocate the same transition counts. Consequently the total number
of predicted changed pixels is equal on the full prediction footprint. On a
subset evaluation mask, counts can differ because the chosen locations differ.

## First local results

| Evaluation | Binary accuracy | Multiclass accuracy | Binary change FoM | Multiclass change FoM |
|---|---:|---:|---:|---:|
| Held-out blocks, 2010→2015 | 90.44% | 90.57% | 21.72% | 22.35% |
| Full footprint, 2015→2020 | 90.60% | 90.72% | 20.29% | 20.91% |

Both allocate 21,871 changed pixels in the spatial test and 67,954 in the temporal
test. The gain is small and has not been established as statistically significant
across seeds/regions/periods. These are map-label agreement scores, not independent
field validation. Both models have lower overall accuracy than persistence on
this data, but identify some changes that persistence cannot identify.

## Run and audit

From `Land Use-Deforestation`:

```bash
../../.venv/bin/python -m local_landuse --experiment controlled --output-dir outputs/my_controlled_run
../../.venv/bin/python -m pytest tests/test_local_landuse.py tests/test_controlled_landuse.py -q
```

The CLI defaults to this controlled experiment. To reproduce the earlier direct
multiclass experiments, explicitly pass `--experiment direct` and optionally
`--sampling legacy_balanced`; see `LOCAL_COMPARISON.md`.

The run directory contains `report.json`, all-scope `comparison.csv`, trained
models, actual sampled/train/test row indices, block IDs, and training diagnostics.
Each scope subdirectory contains predictions, an evaluation mask, an error map,
metrics, per-class scores, shared demands, and both allocation ledgers.

Tests verify identical/disjoint split identities, score-column interpretation,
identical allocation ledgers, independence from future labels, and independence
of spatial predictions/demand from held-out calibration destination labels.
