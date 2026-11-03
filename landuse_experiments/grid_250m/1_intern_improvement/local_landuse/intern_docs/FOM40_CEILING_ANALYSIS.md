# Why the 40% target has not been reached, and what bounds it

**The 40% target was not reached.** The best forecast on the complete 2015→2020
evaluation area is now **28.06% FoM** at **91.25% overall accuracy**, from a
probability-averaging ensemble added this round — the highest on both measures so
far, and the only change this round that improved the forecast. Two new feature
families, one alternative selection protocol and a 15-year annual vegetation
history were also added; those produced negative or negligible results,
documented below.

All numbers use the same 992,833 evaluation pixels, the same original class
codes, and the same 75,365 observed changes (7.59%).

## The target is bounded by false alarms, not by missed change

Because `hits + wrong destination + missed = observed change = 75,365` is fixed,

```
FoM = hits / (75,365 + false alarms)
```

so the target implies a hard requirement on precision:

| Allowed false alarms | Correctly located *and* labelled changes needed for 40% |
|---:|---:|
| 0 | 30,412 |
| 10,000 | 34,412 |
| 20,000 | 38,412 |
| 45,376 (current best) | 48,296 |

The best ensemble already produces **33,874 hits** — more than the 30,412 that
40% would require at zero false alarms. It produces **45,376 false alarms**
alongside them. Every remaining point of FoM has to come from separating change
from non-change more sharply, not from finding more change. This is exactly the
mechanism by which the probability-averaging ensemble gained: against its own
strongest member it cut false alarms from 52,109 to 45,376 (**-12.9%**) while
losing only 862 hits.

## Decision-threshold tuning is exhausted

The runner already searches three decision rules (`odds`, `joint`, `risk`) over
65 thresholds each, plus coordinate-optimised per-starting-class thresholds,
scored on the same full-mask FoM. Taking the **best of all 198 policies** for the
strongest fused model on the *spatial validation* period — an oracle choice, not
an honest one — reaches **35.28% FoM** (15,109 hits, 16,363 false alarms). The
validation period is the easier one. No threshold in that search reaches 40%,
so the remaining gap cannot be closed by allocation or thresholding.

## Part of the target is transient product noise

The land-cover maps flip labels back and forth on a substantial minority of
pixels, and those flips are counted as observed change.

| Measurement | Value |
|---|---:|
| Pixels that changed 2010→2015 and change again 2015→2020 | 37.4% |
| Of those, share returning to their **2010** class | 88.2% |
| 2015→2020 changes that revert to the 2015 class by 2024 | 28.9% |
| Pixels changing in 2+ of the five intervals (2001–2024) | 12.6% |
| Pixels never changing across all five intervals | 72.5% |

A trivial rule — *pixels that changed in the previous interval revert to their
older class* — scores **19.96% FoM** on 2015→2020, essentially matching the
original-style transition RF at 20.02%. This is not evidence that the original
implementation is trivial; it indicates that a large share of what the metric
counts as predictable change is label reversion rather than physical change.

## What was added this round

**Class-membership atypicality** ([class_typicality.py](local_landuse/class_typicality.py)).
For each starting year, per-class diagonal Gaussian descriptions of seasonal
NDVI/EVI are estimated **only from pixels whose label was already unchanged over
the previous five years**. Each pixel then receives its own-class log-likelihood,
posterior, margin against the best competing class, the identity of that
competing class, neighbourhood versions of those, and their five-year change.
No outcome label and no observation after the starting year is read; a test
overwrites the 2020 map and asserts the features are bit-identical.

These features carry real, independent signal. Measured directly on 2015:

| Single feature | AUC for change 2015→2020 |
|---|---:|
| neighbourhood own-class posterior | 0.753 |
| own-class posterior margin | 0.744 |
| own-class posterior | 0.740 |
| own-class log-likelihood | 0.720 |

The proposed competing class alone matches the observed 2020 destination for
**59.7%** of actual changes.

On their own they did **not** improve the forecast, because gradient-boosted
trees were already recovering equivalent information from the raw seasonal
features plus the starting class. Matched pairs, identical protocol and rows:

| Model | Spatial validation FoM | With typicality | Forecast FoM |
|---|---:|---:|---:|
| `joint_xgb` | 35.28% | 35.33% | 27.13% |
| `deep_xgb` | 35.48% | 35.54% | 27.10% |
| `two_stage_rf` | 32.50% | 32.61% | 26.40% |
| `expanded_xgb` | 36.24% | 36.34% | **27.25%** (selected) |

Validation improved by 0.05–0.11 points; the selected model's forecast fell from
27.87% to **27.25%**. Re-running the class-specific ensemble over five runs
including this one gave **27.87%**, below the four-run ensemble's 27.97%.
They are not wasted, however: three of the four members the probability-averaging
ensemble chose are typicality models, so the features earn their place as a
source of disagreement between members rather than as a better single model.

**Fifteen-year annual vegetation history** (`transfer_models --feature-set annual`).
Annual MODIS 2001–2015 instead of the 2005/2010/2015 endpoints, giving per-pixel
interannual slope, variability, robust trend and the departure of the latest year
from its own history. On the endpoint temporal protocol it beat its 3-date
counterpart on **9 of 11 candidates**: selected `expanded_xgb` **27.44%** versus
26.02%, best individual **27.76%** versus 27.18%. The gain is real but small, and
the protocol as a whole still trails the spatial one.

**Probability averaging across fitted models**
([probability_ensemble.py](local_landuse/probability_ensemble.py)). This is the
only change this round that improved the forecast. The existing ensemble picks
one model per starting class, which keeps a single model's ranking within each
class. This module instead rebuilds the probability surface of every saved model
across five runs (36 members), then chooses members greedily **with replacement**
on the held-out spatial blocks, each step required to improve full validation FoM.
Membership and decision policy are written to `frozen_selection.json` before any
2020 label is read.

Selection converged on four equally weighted members — `expanded_typicality_xgb`,
`typicality_deep_xgb`, `expanded_typicality_rf` and `vegetation_context_expanded_xgb` —
at **36.69%** validation FoM, the highest recorded, and **28.06%** on the forecast:

| | Best single member | Probability ensemble | Change |
|---|---:|---:|---:|
| Forecast FoM | 27.25% | **28.06%** | +0.81 |
| Overall accuracy | 90.66% | **91.25%** | +0.59 |
| Correctly located changes | 34,736 | 33,874 | −862 |
| False alarms | 52,109 | **45,376** | **−6,733** |
| Predicted change | 9.09% | 8.28% | — |

The gain arrives through precision, as the identity above requires: a 12.9% cut
in false alarms for a 2.5% cut in hits.

**MODIS burned area** ([fire_history.py](local_landuse/fire_history.py),
[fire_models.py](local_landuse/fire_models.py)). Fifteen years of MCD64A1,
2001–2015, twelve monthly granules per tile and year, each beginning and ending
inside its own year. Every year reached **100% mapped coverage**; between 13.2%
and 19.0% of land burns annually. Features are annual burned share, burn counts
and shares over 3-, 5- and 10-year horizons, years since the last fire, mean and
spread of burn day, a recent-versus-earlier frequency trend, and neighbourhood
versions of these. Unmapped cells keep support columns instead of becoming
unburned zeros.

Fire is genuinely informative, but only *conditionally on the starting class*,
which is why its marginal AUC is near chance (best single feature 0.561). The
direction of the effect reverses between classes, so the effects cancel in the
margin:

| Burns in the previous 10 years | class 9 savanna | class 10 grassland | class 8 woody savanna |
|---|---:|---:|---:|
| 0 | 6.5% | 9.1% | 6.3% |
| 1–3 | 4.6% | 13.3% | 9.3% |
| 4–7 | 2.9% | 14.7% | 32.4% |
| 8–10 | 2.5% | 17.4% | 100% |

Frequently burned savanna is *stable* savanna; frequently burned grassland and
woody savanna are unstable. Tree models condition on the starting class and can
represent this, and the fire candidates did reach the highest validation scores
recorded — `expanded_fire_xgb` at 36.38%, and a probability ensemble including
fire at **36.76%**, the best validation figure in the project. Neither
transferred. The fire run's selected model scores **27.49%** on the forecast, and
the fire-inclusive probability ensemble **27.77%**, both below the 28.06% reached
without fire. Fire is therefore recorded as a negative result for the forecast,
and the best model remains the fire-free ensemble.

**Endpoint temporal selection** (`transfer_models --feature-set endpoints`).
The preceding round selected models on 2010→2015 spatial blocks and lost 8 points
transferring to 2015→2020. This run instead fits an earlier period and selects on
a later one, so that selection is itself a temporal-transfer decision, and adds
CatBoost candidates and label-free covariate-shift weighting. It did not close
the gap: the selected `adapt_rf` scores **26.02%**, the best individual candidate
**27.18%**, both below the spatial protocol's 28.00%.

## Spatial and regional post-processing cannot raise precision

Diagnostics on the saved best forecast map (`prob_ensemble_20260910`), added
2026-09-13. No model was refitted; the 2020 labels were already inspected in
earlier rounds, and nothing here was used for selection.

The identity gives an exact criterion for any filter that turns predicted
changes back into "no change". Removing a set with `h` hits and `f` false alarms
raises FoM only if `h / f < FoM` — fewer than **0.28 hits per false alarm**,
i.e. a set with precision below about **22%**. Every spatially defined set
tested is far above that.

**Isolated predictions are the most precise, not the least.** Among predicted
changes, grouped by the number of predicted-change neighbours in a 3×3 window:

| Neighbours | Predicted | Precision |
|---:|---:|---:|
| 0 | 6,775 | **50.6%** |
| 1 | 12,816 | 43.1% |
| 2–4 | 45,428 | 39.5–40.7% |
| 5–8 | 17,186 | 36.9–39.5% |

Pruning isolated predictions therefore lowers FoM: removing those with 0, ≤1, ≤2
neighbours gives 25.90%, 22.54%, 17.93%. The same holds when neighbours must share
the predicted destination (23.67% at 0). Probability smoothing, which makes
predictions more clustered, pushes toward the lower-precision rows. Observed change
is strongly clustered (86.1% of changes have a changed neighbour, against 46.8% at
random), but false alarms are clustered just as much as hits.

**False alarms are regional over-prediction.** At 25 km, blocks that predict more
change than they observe hold 80.8% of false alarms. Block-level predicted change
correlates 0.87 with observed change and 0.96 with false alarms. But capping each
block's quantity does not help, even with an oracle:

| Cap on each block's predicted change | 10 km | 25 km |
|---|---:|---:|
| none (current) | 28.06% | 28.06% |
| **oracle**: observed 2015→2020 count | 25.31% | 25.64% |
| **oracle**: 1.5 × observed | 27.83% | 27.95% |
| honest: 2 × the block's 2010→2015 rate | 28.04% | 28.06% |

Removal inside a block here is proportional, not score-ranked, because only the
decided map is saved. A score-ranked cap removes the lowest-scoring pixels, which is
what the per-class thresholds already do. A regional prior also brings no new
information: the model's own block-level change rate correlates **0.906** with
2015→2020, slightly better than 2010→2015 persistence at **0.897**.

Transitions confirm where the churn is. Four transitions carry 62.5% of false
alarms: 9→10 (23.1%), 14→9 (13.5%), 10→9 (13.0%) and 8→9 (12.9%, at 25.1%
precision). Transitions among the savanna classes 8/9/10 hold 51.9% of false
alarms and 49.3% of observed change.

## A model-free rule beats the original implementation

Added 2026-09-16 ([trivial_rules.py](local_landuse/trivial_rules.py)). Each rule below
fits nothing and reads land-cover labels up to 2015 only. All are scored on the
identical 992,833-pixel mask, and the original-style baselines are re-scored from
their saved maps in the same run.

| Rule | Forecast FoM | Predicted change | Hits | False alarms | Accuracy |
|---|---:|---:|---:|---:|---:|
| Original-style transition RF, local reproduction | 20.02% | — | — | — | 90.28% |
| Original-style binary RF, shared samples/blocks | 20.29% | — | — | — | 90.60% |
| Initial multiclass RF, shared samples/blocks | 20.91% | — | — | — | 90.72% |
| **Majority class of the eight 2015 neighbours** | **22.70%** | 68,322 | 26,046 | 39,380 | 91.07% |
| Majority class of the 2015 3×3 window | 21.10% | 52,385 | 21,851 | 28,213 | 91.77% |
| Revert to the 2010 class if changed 2010→2015 | 19.96% | 73,236 | 24,185 | 45,828 | 90.23% |
| Revert to the 2005 class if changed 2010→2015 | 17.34% | 44,975 | 17,418 | 25,061 | 91.64% |
| Revert to the modal 2001–2015 class if changed 2010→2015 | 17.19% | 42,572 | 16,983 | 23,437 | 91.76% |
| Previous changers take the eight-neighbour majority | 16.61% | 31,134 | 14,935 | 14,576 | 92.45% |
| Previous changers move to their class's most likely destination | 15.18% | 73,175 | 18,392 | 45,767 | 89.65% |
| Revert to the modal class if changed twice | 2.00% | 5,130 | 1,567 | 2,908 | 92.27% |
| No change | 0.00% | 0 | 0 | 0 | 92.41% |
| Probability-averaging ensemble, for scale | 28.06% | 82,205 | 33,874 | 45,376 | 91.25% |

**Assigning every pixel the most common class among its eight 2015 neighbours scores
22.70%**, above all three original-style baselines and above the reversion rule's
19.96%. It uses one map, no training, no covariates and no history. Letting the pixel
vote for its own class as well gives 21.10%.

The mechanism is the same churn the models learn: at 500 m the product's class
boundaries shift between years, so a pixel whose neighbourhood already disagrees with
its own label is the one most likely to be relabelled next. Note that "no change"
scores 0.00% FoM at 92.41% accuracy, the highest accuracy in the table — which is why
accuracy is not the operative measure here.

Two things this does not show. It is not evidence that the models are worthless: the
ensemble's 28.06% remains 5.36 points above the best model-free rule, and 8.04 points
above the original-style baseline. And it is not a candidate forecast — a rule that
simply spreads existing boundaries has no mechanism for anticipating new change. It
is evidence about what the metric rewards on this product.

## A temporally cleaned reference lowers every score

Added 2026-09-13 ([hmm_reference.py](local_landuse/hmm_reference.py)). For coarse
land-cover products, the established remedy for label churn is to smooth each
pixel's label sequence with a hidden Markov model before measuring change
(Abercrombie & Friedl 2016, *IEEE TGRS* 54:703). It was applied to the
2001/2005/2010/2015/2020/2024 stack to test whether a cleaner reference moves the
forecast past 28%.

**Protocol.** Starting-class frequencies, a single transition matrix and one label
accuracy are fitted by Baum–Welch on **2001–2015 labels only**; a test overwrites
the later years and asserts the parameters are identical. Label errors are assumed
to fall on other classes in proportion to observed swaps. Viterbi then smooths the
whole sequence, so 2020 and 2024 inform only the reference. No model or forecast map
was changed. The main reference keeps pixels whose raw 2015 label survives smoothing
(971,467 of 992,833), so the models' input and the reference start agree, and
scores them against the smoothed 2020 label. The full-mask variant keeps every pixel,
with raw 2015 as the start. Snapshots are 4–5 years apart, far coarser than the
annual sequences the method was designed for.

The fitted label accuracy is 0.972. Smoothing removes **99.7%** of the 2015→2020
changes that revert by 2024, as intended, but also 33.5% of the changes that
persist, and it halves reference change from 75,365 to 35,712 pixels.

| Configuration | Raw labels | Cleaned, stable start | Cleaned, full mask |
|---|---:|---:|---:|
| Original-style transition RF, local reproduction | 20.02% | 12.17% | 18.98% |
| Original-style binary RF, shared samples/blocks | 20.29% | 12.58% | 19.48% |
| Initial multiclass RF, shared samples/blocks | 20.91% | 12.89% | 20.18% |
| Best single member (`expanded_typicality_xgb`) | 27.25% | 15.21% | 26.23% |
| **Probability-averaging ensemble** | **28.06%** | **15.32%** | **27.14%** |
| Reversion rule (diagnostic) | 19.96% | 1.91% | 20.74% |

**Why every score falls.** The cleaning removes the models' hits, not their false
alarms. Of the ensemble's 33,874 raw hits, 15,732 lie on pixels whose 2015 label is
itself judged noise, and 4,906 more become false alarms; 13,190 remain. Of its
45,376 false alarms, 45,219 remain false alarms. Much of what any forecast gets right
on raw labels is anticipating churn, which the reversion rule isolates: it collapses
from 19.96% to 1.91%.

**The gain over the original survives:** +3.15 points on the stable-start reference
(26% relative, against 40% on raw labels) and +8.16 points on the full mask. The model
ranking is unchanged wherever the reference keeps a meaningful amount of change.

**The cleaned score depends mostly on the assumed label accuracy**, which weakens it
as an evaluation basis:

| Assumed label accuracy | Reference change, stable start | Ensemble | Original transition RF |
|---:|---:|---:|---:|
| 0.972 (fitted) | 35,712 | 15.32% | 12.17% |
| 0.95 | 32,453 | 14.08% | 11.25% |
| 0.90 | 17,031 | 8.57% | 7.26% |
| 0.85 | 8,691 | 3.90% | 3.88% |
| 0.75 | 3,077 | 1.52% | 1.59% |

On the full mask at an assumed accuracy of 0.75–0.85, the reversion rule scores
highest (32.2–36.4%), because correcting a noisy 2015 label then counts as change.
That variant rewards predicting churn, so it is not a sound target either.

**Conclusion.** A temporally consistent reference is a defensible robustness check,
and it confirms the improvement over the original implementation. It is not a route
past 28%: against the cleaned reference the best forecast scores 15.32%, further from
40% than on raw labels.

## Comparison with the original implementation

Required in every report. Same evaluation mask, same starting and target maps,
verified by hash and by re-scoring each saved baseline map.

| Configuration | Forecast FoM | Accuracy | Predicted change |
|---|---:|---:|---:|
| Original-style transition RF, local reproduction | 20.02% | 90.28% | 7.28% |
| Original-style binary RF, shared samples/blocks | 20.29% | 90.60% | 6.84% |
| Initial multiclass RF, shared samples/blocks | 20.91% | 90.72% | 6.84% |
| Best model-free rule: eight-neighbour majority | 22.70% | 91.07% | 6.88% |
| Same-protocol RF control | 26.93% | 90.19% | 9.80% |
| Endpoint temporal selection, selected | 26.02% | 87.95% | 13.42% |
| Annual vegetation history, selected | 27.44% | 88.74% | 12.61% |
| Fire history, selected | 27.49% | 90.67% | 9.28% |
| Probability ensemble including fire | 27.77% | 91.29% | 8.12% |
| Class typicality, selected | 27.25% | 90.66% | 9.09% |
| Five-run class-specific ensemble | 27.87% | 90.84% | 8.98% |
| Four-run class-specific ensemble | 27.97% | 91.10% | 8.54% |
| Best individual forecast, MODIS + neighborhood | 28.00% | 90.63% | 9.40% |
| **Probability-averaging ensemble (best overall)** | **28.06%** | **91.25%** | **8.28%** |

Against the original-style local transition RF the best result improves FoM by
**8.04 points**; against the shared-sample binary RF by **7.77 points**. It
remains **11.94 points short of 40%**. The original-style row is a local
reproduction with valid-pixel and single-conversion safeguards, not an untouched
execution of the cloud backend.

## Assessment

Across this and the preceding rounds, 28 satellite-feature candidates, 11
endpoint-transfer candidates, 8 typicality candidates, longer land-cover
histories, larger training samples, covariate-shift weighting, CatBoost, and
class-specific ensembling, fifteen years of MODIS burned area and probability
averaging have moved the forecast from 20.0% to 28.1%. Every feature family added after the first satellite round has
changed the forecast by less than one point, in both directions, while validation
scores kept rising — the signature of a validation ceiling rather than a
modelling gap. Only combining models, which attacks precision directly, has
produced a further gain, and it produced 0.06 points over the previous best.

Two independent data sources have now been added whose features rank highest on
validation and lowest on transfer — class typicality and fire history. In both
cases the validation score reached a project record while the forecast fell. That
is the clearest available evidence that the remaining gap is not a missing
covariate.

Reaching 40% would require roughly **78% precision over ~44,000 predicted
pixels**. The strongest available ranking reaches about 59% precision over its
top 10,000 pixels, and precision falls from there. On this data — 500 m maps at
five-year steps, with ~29% of the target reverting within the next interval — 40%
does not appear reachable without either a different evaluation definition
(scoring change without destination is untested; removing reverting change by HMM
smoothing lowers every score, see above)
or a land-cover product with lower label churn. That is a statement about what
the evidence supports, not a recommendation to stop.

## Reproduce

```bash
../../.venv/bin/python -m local_landuse.fire_history --output-dir outputs/historical_fire --years 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015
../../.venv/bin/python -m local_landuse.fire_models --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --fire-dir outputs/historical_fire --output-dir outputs/my_fire_run
../../.venv/bin/python -m local_landuse.probability_ensemble --run outputs/phenology_models_20260910=phenology --run outputs/vegetation_context_20260910=context --run outputs/spectral_models_20260910=spectral --run outputs/fused_satellite_20260910=fused --run outputs/typicality_20260910=typicality --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --output-dir outputs/my_probability_ensemble
../../.venv/bin/python -m local_landuse.class_typicality --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --output-dir outputs/my_typicality_run
../../.venv/bin/python -m local_landuse.transfer_models --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --feature-set endpoints --output-dir outputs/my_endpoint_run
../../.venv/bin/python -m local_landuse.transfer_models --phenology-dir outputs/historical_modis --spectral-dir outputs/historical_landsat --feature-set annual --output-dir outputs/my_annual_run
../../.venv/bin/python -m local_landuse.source_ensemble --runs outputs/phenology_models_20260910 outputs/vegetation_context_20260910 outputs/spectral_models_20260910 outputs/fused_satellite_20260910 outputs/typicality_20260910 --output-dir outputs/my_ensemble
../../.venv/bin/python -m local_landuse.baseline_comparison --runs outputs/my_typicality_run outputs/my_endpoint_run outputs/my_ensemble
../../.venv/bin/python -m local_landuse.hmm_reference --output-dir outputs/my_hmm_reference
../../.venv/bin/python -m local_landuse.trivial_rules --output-dir outputs/my_trivial_rules
../../.venv/bin/python -m pytest tests -q
```

Verification: **74 tests pass**. Every run's `comparison_with_original.csv` and
`.json` re-score the original baseline maps on the identical mask and fail if a
saved FoM no longer reproduces. Reports: `outputs/trivial_rules_20260916/report.json`,
`outputs/hmm_reference_20260913/report.json`,
`outputs/prob_ensemble_20260910/report.json`,
`outputs/fire_models_20260910/report.json`,
`outputs/prob_ensemble_fire_20260910/report.json`,
`outputs/typicality_20260910/report.json`,
`outputs/transfer_endpoints_20260910/report.json`,
`outputs/transfer_annual_20260910/report.json`,
`outputs/source_ensemble_20260910b/report.json`.
