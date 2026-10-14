"""
The deployed FDI threshold rule, evaluated as if it were a classifier.

WHY THIS MODULE EXISTS AND WHAT IT IS FOR

Guinea ships a mean+3sigma FDI anomaly rule. Every FDI patch reported for
Conakry and for the river corridors came from it. But the rule has never been
evaluated against labelled data, for the simple reason that no labelled marine
debris exists for Guinea -- which is exactly why it was built as an
unsupervised anomaly rule in the first place.

MARIDA closes that gap. It lets the rule be run, unmodified, on pixels whose
true class is known, and answers two questions that matter more than accuracy:

  1. When the rule fires, WHAT does it fire on? If most detections are
     Sargassum, foam or wakes, then every patch count this module reports is
     a count of floating material, not of plastic, and the methodology must
     say so in those words.

  2. Does the rule find debris at all? A rule that never fires on true debris
     would make the Guinea layer an elaborate way of reporting nothing.

THE THRESHOLD IS DERIVED THE SAME WAY, NOT TUNED

`de_africa_threshold` reproduces the deployed rule: mean + 3 sigma of FDI over
the WATER population of the scene. It is not fitted to MARIDA and no value is
chosen to make the comparison flattering. Fitting it would answer a different
and much less useful question -- how well the rule COULD do -- when what is
deployed is how well it DOES.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import features as F
from .data import (CLASSES, FLOATING_CONFUSERS, NON_WATER, TARGET,
                   WATER_CLASSES)

N_SIGMA = 3.0


def de_africa_threshold(fdi_values: np.ndarray, is_water: np.ndarray,
                        n_sigma: float = N_SIGMA) -> float:
    """
    mean + n*sigma of FDI over the water population -- the deployed rule.

    Digital Earth Africa's coastline/debris convention, and the same one
    `fdi_tool.detection.debris_binary_mask_de_africa` applies in production.
    """
    pop = fdi_values[is_water & np.isfinite(fdi_values)]
    if pop.size < 2:
        return float("nan")
    return float(pop.mean() + n_sigma * pop.std())


def predict(X: np.ndarray, y: np.ndarray,
            group: Optional[np.ndarray] = None,
            n_sigma: float = N_SIGMA,
            with_ndvi: bool = False) -> Dict:
    """
    Apply the rule and report what it caught.

    The threshold is derived PER SCENE when `group` is supplied, because that
    is how the pipeline works -- one AOI, one composite, one threshold. Pooling
    every scene into a single global threshold would be a different algorithm
    and would flatter or damage the rule depending on how scene brightness
    happens to distribute.

    Water pixels for the threshold population are MARIDA's water classes. Cloud
    and cloud-shadow pixels are excluded, matching the pipeline's cloud mask.

    `with_ndvi=True` adds the second arm of the production rule
    (`fdi_tool.detection.debris_binary_mask_de_africa`): NDVI must also exceed
    mean + n*sigma of NDVI over the same water population. The default, FDI
    alone, is the form section 2 of marida_workflow.ipynb evaluates.
    """
    fdi = F.fdi(X)
    ndvi = F.ndvi(X) if with_ndvi else None
    is_water = np.isin(y, WATER_CLASSES)
    scored = ~np.isin(y, NON_WATER)          # pixels the rule would ever judge

    pred = np.zeros(len(y), dtype=bool)
    thresholds: Dict[str, float] = {}

    groups = np.asarray(group) if group is not None else np.zeros(len(y), int)
    for g in np.unique(groups):
        m = groups == g
        thr = de_africa_threshold(fdi[m], is_water[m], n_sigma)
        thresholds[str(g)] = thr
        if np.isfinite(thr):
            pred[m] = scored[m] & (fdi[m] > thr)
            if with_ndvi:
                t_ndvi = de_africa_threshold(ndvi[m], is_water[m], n_sigma)
                pred[m] &= np.isfinite(t_ndvi) & (ndvi[m] > t_ndvi)

    return {"pred": pred, "fdi": fdi, "thresholds": thresholds,
            "is_water": is_water, "scored": scored}


def what_does_it_fire_on(y: np.ndarray, pred: np.ndarray):
    """
    The composition of the rule's positives -- the central table of this work.

    Reports, for every MARIDA class, how many of its pixels the rule flagged
    and what share of all flagged pixels that class represents. The second
    column is the one that matters: it is the answer to 'when Guinea reports
    an FDI patch, what is it most likely to be?'
    """
    import pandas as pd

    rows = []
    n_pos = int(pred.sum())
    for code, name in CLASSES.items():
        m = y == code
        n = int(m.sum())
        if n == 0:
            continue
        hit = int((m & pred).sum())
        rows.append({
            "code": code, "class": name, "pixels": n,
            "flagged": hit,
            "recall_within_class": hit / n,
            "share_of_all_detections": hit / n_pos if n_pos else np.nan,
            "role": ("TARGET" if code == TARGET
                     else "floating confuser" if code in FLOATING_CONFUSERS
                     else "water background" if code in WATER_CLASSES
                     else "other"),
        })
    df = pd.DataFrame(rows).sort_values("share_of_all_detections",
                                        ascending=False)
    return df


def binary_scores(y: np.ndarray, pred: np.ndarray,
                  scored: Optional[np.ndarray] = None) -> Dict:
    """
    Debris-vs-everything-else scores for the rule.

    `scored` restricts to pixels the rule would actually judge (cloud and
    shadow excluded), so the rule is not penalised for classes the production
    cloud mask removes before it ever runs.
    """
    m = np.ones(len(y), bool) if scored is None else scored
    t = (y[m] == TARGET)
    p = pred[m]
    tp = int((t & p).sum()); fp = int((~t & p).sum())
    fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and
                                           np.isfinite(prec) and
                                           np.isfinite(rec) and
                                           (prec + rec) > 0) else float("nan")
    base = t.mean()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "base_rate": float(base),
            "lift": (prec / base) if (base > 0 and np.isfinite(prec)) else np.nan,
            "n_scored": int(m.sum())}
