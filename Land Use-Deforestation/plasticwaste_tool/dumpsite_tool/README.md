# dumpsite_tool — importable package (DPWMS component 2)

These are **copies** of the numbered research scripts in the parent folder,
renamed so Python can import them. The numbered originals stay where they are
and are what `dumpsite_detection_workflow.ipynb` uses; nothing about the
notebooks changes.

| module | copied from | what it holds |
|---|---|---|
| `inference.py` | `05_guinea_inference.py` | tile fetchers (Esri + Google Static Maps), blank-placeholder detection, tile grids, scoring |
| `feature_probe.py` | `02c_feature_probe.py` | feature extraction, spatial blocking, bootstrap CIs |
| `features.py` | `02d_benchmark_features.py` | ImageNet / colour representations, paired bootstrap |
| `point_model.py` | `08_point_model.py` | build samples, fetch, evaluate, contact sheets, archive |
| `footprint.py` | `09_national_footprint.py` | national scan footprint (GHSL SMOD), random sampling |
| `train_head.py` | `02e_train_probe_head.py` | freeze a deployable probe head |
| `scan.py` | *(not copied — written here)* | **the deployment path**: two-stage national scan |

Regenerate the copied modules with `scratchpad/build_package.py`. Do not
hand-edit them until the numbered originals are retired, or the two will
drift apart. `scan.py`, this README and `models/` are NOT generated —
the generator never touches them.

---

## Import torch FIRST — this is not optional on Windows

```python
import torch          # <- must come first
import plasticwaste_tool.dumpsite_tool.point_model as pm
```

Reversing that order fails with:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine
failed. Error loading "...torch\lib\c10.dll" or one of its dependencies.
```

**Why:** torch and the GEE/GDAL stack both ship their own native Windows DLLs,
and whichever initialises first wins. `plasticwaste_tool/__init__.py` eagerly
imports `api_connector`, which pulls in GDAL — so by the time Python reaches
this subpackage, the race is already lost.

This subpackage's `__init__.py` imports torch defensively, but **that cannot
help** when the parent package has already been executed as part of the same
import statement. The fix has to be at the caller.

**For the backend:** put `import torch` at the top of the Celery task module,
before any `plasticwaste_tool` import. One line, and it prevents an error that
looks like a broken torch installation rather than an ordering problem.

The alternative — importing torch at the top of `plasticwaste_tool/__init__.py`
— would fix it globally but forces a torch dependency on FDI-only deployments
that do not need it. That is a call for whoever owns the shared module, so it
has been left alone.

---

## The backend should not import this package

The stable interface is:

```python
plasticwaste_tool.detection.classify_dumpsite_candidates()
```

which already exists, already returns the documented schema (empty while
`config.DUMPSITE_MODEL_CONFIG["enabled"]` is `False`), and is what
`docs/GEDH_Integration_Deployment_Plan.md` names as the integration point.

`dumpsite_tool` is the implementation behind that function. Keeping the backend
pointed at `detection.py` means the developers' surface stays one function wide
and does not move when this package changes.

---

## Current state

Both heads are frozen and the deployment interface is wired. The backend can
call it today; it returns an empty typed layer until
`config.DUMPSITE_MODEL_CONFIG["enabled"]` is flipped.

### The frozen heads

Three heads ship. Two are loaded by `scan.py`; the third is a documented
experiment that is deliberately not wired to anything (see below).

| | stage 1 — screen | stage 2 — confirm |
|---|---|---|
| file | `models/probe_screen.joblib` | `models/probe.joblib` |
| imagery | Esri z17, **free** | Google Static Maps z19, **billable** |
| footprint | 250 m cells | 80 m windows |
| trained on | 595 cells, 55 positive | 651 point-centred samples, 111 positive |
| regularisation | `C=0.001` | `C=0.01`, chosen by blocked inner CV |
| blocked-CV ROC-AUC | 0.856 | 0.911 |
| average precision | — (judged on recall) | 0.696 |
| loaded by `scan.py` | yes | yes (default) |

The third, `models/probe_urban.joblib`, is the hard-negative retraining
experiment: same architecture, 853 samples with 202 urban windows added,
C=0.001. It is frozen so the experiment is reproducible, it carries
blind-grid metrics in its own `_meta.json`, and **nothing loads it** — it did
not beat the shipped head (ΔROC-AUC −0.004) and is kept for provenance only.

Stage 1 is deliberately generous and deliberately inaccurate: at threshold 0.30
it keeps 31.6% of cells and retains 85.5% of true sites. It is the settlement
detector described above, used on purpose — its only job is to discard empty
country on free imagery so stage 2 never pays to look there.

### Reading the recall figure honestly

End-to-end recall is **stage 1's 85.5% multiplied by stage 2's recall at the
tier being displayed** — not the tier's recall alone. At the default `high`
tier that is 0.855 × 0.324 ≈ **28%** of sites found. High confidence buys
precision (86% at eval prevalence) by giving up recall; `all` finds far more
and is wrong more often. That trade is the point of publishing tiers rather
than one cut-off, and any recall number that reaches a document must say which
tier it refers to and that stage 1 is already folded in.

### Cost of a national scan

Stage 1 is free. Stage 2, on the settled footprint (GHSL SMOD ≥ 21 plus a 1 km
fringe, 89,336 cells), is roughly **276,000–404,000 billable requests, about
$530–790** — the range is how clustered the kept cells turn out to be, since
adjacent cells share windows and those are fetched once. Tighter and wider
footprints are costed in `footprint.py budget`.

### The blind-grid test was run, and the point-centred figures did not transfer

Point-centred crops and a blind grid are different tasks, and the gap between
them is large. Measured on 422 hand-labelled windows over a 1.9 x 0.9 km
Conakry AOI (`manual_qa/runs/cascade_demo_conakry/`), none used in training:

| task | ROC-AUC | average precision |
|---|---|---|
| crop classification ("is this 80 m crop a dumpsite?") | 0.911 | 0.696 |
| blind-grid detection ("where are the dumpsites here?") | 0.682 | 0.476 |

Operating points on the blind grid, against a 26.1% base rate:

| threshold | flagged | true | precision | recall |
|---|---|---|---|---|
| 0.90 (`high`) | 77 | 45 | 0.584 | 0.409 |
| 0.70 (`medium`) | 140 | 61 | 0.436 | 0.555 |
| 0.50 (`all`) | 200 | 72 | 0.360 | 0.655 |

**Cause.** 43% of positives are Conakry against 9% of negatives, so the model
partly learned "is this dense urban?" instead of "is this waste?". The frozen
head scores the Conakry negatives it was *trained* on at 0.216 but unseen urban
tiles above 0.99: with 2048 features and n=651 it memorised the 50 urban
negatives it saw and never formed a general concept of "urban without waste".

**Cross-validation could not have caught this.** Spatial blocking stops a tile
leaking into the adjacent fold; it does nothing about a dataset-wide
correlation between terrain type and label, and every fold shared the same
mostly-rural negative distribution. This is covariate shift, visible only on a
blind grid.

**What still holds.** ROC-AUC 0.911 / AP 0.696 are not withdrawn. They answer
"is this crop a dump or random Guinean land?" and remain the correct basis for
the Esri-vs-Google comparison. They do not answer "where are the dumps in this
city?", which is what a national sweep asks.

### Two attempted remedies, both measured, neither adopted

Both were run on the spatially separated east half of the review AOI (220
held-out windows), which is why the baseline reads 0.692/0.485 here rather than
the 0.682/0.476 measured over all 422.

**1. Hard negatives.** 202 hand-labelled urban windows (49 positive) were added
to training and frozen as `models/probe_urban.joblib`.

| | ROC-AUC | average precision | prec@0.90 | recall@0.90 | flagged@0.90 |
|---|---|---|---|---|---|
| `probe.joblib` (shipped) | 0.692 | 0.485 | 0.561 | 0.377 | 41 |
| `probe_urban.joblib` | 0.688 | 0.504 | **0.667** | **0.131** | 12 |

ΔROC-AUC −0.004. The apparent precision gain is not an improvement: recall fell
from 0.377 to 0.131 and flagged windows from 41 to 12. The model became more
conservative, not more discriminating, and ROC-AUC — which is threshold-free —
did not move. **`probe_urban.joblib` is retained for provenance and is NOT the
model of record**; it does not overwrite `probe.joblib` and nothing loads it by
default.

**2. Changing the training mixture.** A model trained on urban windows alone
(n=202, dropping the 540 mostly-rural samples) leads in absolute terms —
ROC-AUC 0.747, AP 0.555 — but a paired block bootstrap over 34 spatial blocks
puts the difference at ΔROC-AUC +0.055, 95% CI [−0.048, +0.143], P=0.277, and
ΔAP +0.069, 95% CI [−0.093, +0.208], P=0.385. Both intervals include zero. It
is additionally confounded: the two halves are adjacent parts of one 1.9 km
area, so an unknown share of the lead is local adaptation rather than a better
mixture. Documented, not adopted.

### What this means for deployment

The engineering is finished and tested. The model is a **screening aid**, not a
national inventory:

* At the default `high` tier, blind-grid precision is 0.584 against a 0.261
  base rate — roughly 2.2× better than chance. That is a usable prioritisation
  signal and nothing more.
* End-to-end recall at `high` is stage 1's 85.5% multiplied by stage 2's
  blind-grid 40.9% ≈ **35% of sites present**, and lower for sites that are
  small, canopy-covered or walled. Any recall figure that reaches a document
  must name its tier and say that stage 1 is folded in.
* The 26.1% base rate is enriched: the AOI was chosen because it contained
  known dumps, so precision in a randomly chosen area would be lower.

**The outstanding requirement before a national layer is published is a second
blind grid in a town outside Conakry** — roughly 150 metered requests plus
labelling time, and the only way to learn whether the Conakry figure
generalises. Neither hard negatives nor mixture changes moved the number; of
the remaining candidates, label quality is both cheaper to address and
currently unmeasured, and should be attempted first.

Present every feature as "AI-detected candidate, not field-verified", default
the display to `high`, and pair the map with a per-prefecture ranked worklist —
precision at the top of a ranking is considerably higher than precision across
a whole layer.
