# Local land-cover model comparison

**Latest experiment:** use `controlled_model_comparison.ipynb` and
[CONTROLLED_COMPARISON.md](CONTROLLED_COMPARISON.md) for shared spatial blocks
and identical allocation. This document describes the earlier direct-class
experiments, retained for comparison.

Open `local_model_comparison.ipynb` and select the workspace's `.venv/bin/python`
as its Python kernel. Run the cells from top to bottom. The notebook displays
the saved first run if present; set `RETRAIN = True` for a new timestamped run.
No Earth Engine credentials, Azure account, API, database, or backend is used.

## Models and experiment

Train on **2010 → 2015**, then predict **2015 → 2020** and evaluate against the
2020 map only after prediction. Equal five-year intervals are required. The same
geography is used in calibration and evaluation: this tests temporal prediction,
not generalization to unseen regions. Evaluation is agreement with supplied
land-cover maps, not independent field validation.

The direct experiment's **shared** mode draws one uniform sample of up to **100,000
eligible calibration pixels**. Every forest receives **all the same rows, in
the same order, with equal weights**. No per-transition resampling, filtering,
class balancing or sample weighting is applied. The RF seed and row count are
also identical, giving matching per-tree bootstrap draws. Actual pixel indices
are saved in `shared_training_indices.npy`; report and per-model diagnostics
record their SHA-256 digest. These are row-major indices in the original raster;
`row, column = divmod(index, report['grid']['width'])` recovers coordinates.

1. **Multiclass RF:** predicts the destination class, including persistence.
   Argmax selects the next class; there are no imposed change quotas.
2. **Transition RF + allocation:** each A→B forest predicts whether that exact
   transition occurs. **All other outcomes are negative**, including A→A, A→C,
   and pixels whose starting class is not A. This differs from the original
   A→B versus A→A target and is necessary to use identical rows in every forest.
   Each demand is estimated as `sampled_transition_count * eligible_count /
   sample_size`, then scaled by forecast duration / calibration duration.
   No unsampled transition counts are used for this demand. Transitions with
   estimated full-footprint counts below 200, or fewer than 10 sampled positives
   or negatives, are skipped and recorded. Larger demands are allocated first
   among remaining original-source pixels. Invalid pixels are excluded and a
   pixel cannot change twice in one forecast. The targets, number of forests
   and allocation algorithm still differ between methods; this experiment
   controls samples, not every modeling choice.
3. **No change:** every valid 2015 pixel keeps its class. Both learned approaches
   need comparison with this baseline because persistence dominates these maps.

Both forests use 100 trees, depth 20, minimum split size 5, minimum leaf size 2,
seed 42, and four CPU workers. There is no tuning against the test year.

Both receive the same features: local terrain bands (elevation, slope, aspect),
one-hot starting-year land cover, and 3×3 starting-map neighborhood class
fractions. Neighborhoods are recomputed for 2010 and 2015 separately, excluding
nodata neighbors. The vocabulary is fixed from calibration maps; unseen test
classes are reported. A rare destination can be absent from the multiclass
uniform sample, also reported. In shared mode the current-class indicators vary
within every model's training data, letting each binary forest distinguish its
source class from the others. In legacy mode they are constant within each
source-specific binary model.

The supplied maps have `LC_Type1` band descriptions and share a grid of about
500 m (0.00449 degrees), not the API notebook's 250 m processing grid. Class
codes are preserved without inventing a class-name legend. Values 0 and 255 are
excluded as background/fill in this local experiment. Change counts refer to
pixels, not hectares; latitude-varying cell areas are not treated as constant.

The local population layer is dated 2020; soil moisture and biomass files lack
sufficient temporal metadata for this test. They are deliberately excluded.
Only supply additional rasters via `--static-predictor` if they are valid at
both calibration and forecast start dates. A future extension can accept paired
year-specific predictor stacks; do not substitute future observations.

## Run from a terminal

From the `gis_review_project` workspace root, the environment is already set up:

```bash
cd 'Guinea_Project/Land Use-Deforestation'
../../.venv/bin/python -m local_landuse --experiment direct --sampling shared --output-dir outputs/my_shared_comparison
```

For a fresh installation with Python 3.14, from the workspace root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r 'Guinea_Project/Land Use-Deforestation/requirements-local.txt'
```

The output directory must be new or empty, so previous results cannot be silently
overwritten. Use `--help` for years, sample limits, workers, and other settings.
The runner reads arrays into memory and caps grid size at five million cells.
The bundled country grid has about two million cells; expect several hundred MB
to over 1 GB of RAM and a run lasting minutes, depending on hardware.

### Earlier independently sampled experiment

The original run remains in `outputs/local_comparison_2010_2015_2020`; the new
same-sample run is in `outputs/shared_samples_2010_2015_2020`. To reproduce the
earlier approach, pass `--experiment direct --sampling legacy_balanced` with a new output directory.
That mode gives the multiclass forest 100,000 uniform samples, while each binary
forest independently balances A→B against A→A (up to 50,000 each). It uses full
valid calibration counts for demand. It therefore does **not** meet the shared
sample constraint, and is retained only for comparison with the previous result.
Neither local baseline mode is a byte-for-byte rerun of the legacy notebook.

## Outputs

- `comparison.csv`: overall accuracy, macro F1, change Figure of Merit (FoM),
  observed/predicted change counts, and quantity/allocation disagreement.
- `per_class.csv`: destination-specific precision, recall, F1 and pixel counts.
- `comparison.png`: maps of correct changes, missed changes and false alarms.
- Three predicted GeoTIFFs, `evaluation_mask.tif`, and an uncalibrated
  `multiclass_confidence.tif`. Prediction maps cover the eligible forecast grid;
  all reported metrics use the same evaluation mask for every model.
- `report.json`: input hashes, grid, settings, features, timing, sample counts,
  confusion matrices, unseen classes, shared sample digest and interpretation notes.
- `shared_training_indices.npy`: exact training-pixel indices, in training order
  (shared mode only).
- `transition_training.csv` and `allocation.csv`: skipped models, demand and
  unfulfilled demand. `*_model*.joblib`: fitted models plus feature schema and
  forecast horizon. Load joblib artifacts only from trusted sources.

Higher accuracy, macro F1 and FoM are better. Lower disagreement is better.
FoM counts a change as correct only if its destination is also correct. When
there is no observed or predicted change, this implementation reports FoM=0.
Do not choose a winner based only on overall accuracy, and do not treat these
untuned defaults as the best achievable performance of either method.

## Tests

From `Land Use-Deforestation`:

```bash
../../.venv/bin/python -m pytest tests/test_local_landuse.py -q
```

Tests cover next-state learning, deterministic sampling, nodata/grid handling,
allocation duration and competing conversions, change metrics, and an end-to-end
check that changing future labels cannot alter either model's predictions.
Additional tests intercept every RF fit to verify identical feature rows and
binary labels, and verify shared demand ignores unsampled transition labels.
