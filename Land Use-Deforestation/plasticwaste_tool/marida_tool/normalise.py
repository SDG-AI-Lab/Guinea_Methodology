"""
Scene-relative features: make the classifier work the way the rule already does.

THE INSIGHT THIS MODULE IS BUILT ON

The deployed FDI rule is already immune to the shift that broke the
classifier. It does not ask "is this pixel's FDI above 0.005?" -- it asks "is
this pixel's FDI more than three standard deviations above THIS SCENE'S OWN
water population?". A scene whose water is uniformly brighter in the
shortwave infrared simply gets a higher threshold, and the rule carries on
working.

The classifier had no such protection. It was trained on absolute
reflectance, so a Conakry pixel whose SWIR sits 2.4x above anything in MARIDA
falls outside every leaf the forest ever fitted, and the prediction it
returns is arbitrary rather than merely uncertain.

The fix is to give the model the same footing the rule has: express every
feature RELATIVE TO THE LOCAL WATER POPULATION before the model sees it.

    z = (value - median over this scene's water) / (IQR over the same water)

Median and interquartile range rather than mean and standard deviation,
because the reference population is contaminated by the very anomalies we are
trying to detect -- a large debris field would inflate a standard deviation
and depress its own z-score. The rule accepts that contamination because its
3-sigma cut is generous; a model keying on small differences should not.

WHAT THIS DOES AND DOES NOT FIX

It removes a LEVEL shift: water that is uniformly brighter or darker. It does
not remove a SHAPE shift -- if Guinean water differs in the *relationship*
between bands rather than in their levels, scene-relative features will
differ too, and `domain.feature_shift` will still say so. That is the correct
behaviour: the diagnostic must keep working after the remedy is applied, or
it is not a diagnostic.

It also does nothing about temporal compositing, which is a separate problem
with a separate remedy (single-date acquisitions), and nothing about MARIDA's
label meaning.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .data import WATER_CLASSES

# Features that are already differences or ratios anchored on the water
# background carry meaning at their absolute value too, but standardising them
# costs nothing and keeps one rule for every column.
EPS = 1e-8


def scene_reference(feats, water_mask: np.ndarray,
                    group: Optional[np.ndarray] = None) -> Dict:
    """
    Per-scene median and IQR of every feature over WATER pixels.

    `water_mask` selects the reference population. `group` names the scene each
    pixel belongs to; with no group the whole array is treated as one scene,
    which is the right behaviour when standardising a single deployment AOI.

    A scene with too few water pixels to characterise returns NaNs, which
    propagate to its z-scores and make the failure visible rather than
    producing confident nonsense.
    """
    import pandas as pd

    F = feats if hasattr(feats, "columns") else None
    if F is None:
        raise TypeError("scene_reference expects the DataFrame from features.build")

    g = np.zeros(len(F), dtype=int) if group is None else np.asarray(group)
    w = np.asarray(water_mask, bool)

    ref: Dict = {}
    for scene in np.unique(g):
        m = (g == scene) & w
        if m.sum() < 30:
            ref[scene] = {c: (np.nan, np.nan) for c in F.columns}
            continue
        sub = F.loc[m]
        q1 = sub.quantile(0.25)
        q3 = sub.quantile(0.75)
        med = sub.median()
        ref[scene] = {c: (float(med[c]), float(max(q3[c] - q1[c], EPS)))
                      for c in F.columns}
    ref["_n_scenes"] = len(np.unique(g))
    ref["_n_water"] = int(w.sum())
    return ref


def to_scene_relative(feats, ref: Dict, group: Optional[np.ndarray] = None):
    """
    Standardise each feature against its own scene's water reference.

    Returns a DataFrame with the same columns, suffixed `_z` so a mistake
    that mixes absolute and relative features is visible in a column list
    rather than silent in a fit.
    """
    import pandas as pd

    F = feats
    g = np.zeros(len(F), dtype=int) if group is None else np.asarray(group)
    out = np.full((len(F), len(F.columns)), np.nan, dtype="float32")
    cols = list(F.columns)
    vals = F.values

    for scene in np.unique(g):
        r = ref.get(scene)
        if r is None:
            continue                      # unseen scene -> stays NaN, visibly
        m = g == scene
        for j, c in enumerate(cols):
            med, iqr = r[c]
            out[m, j] = (vals[m, j] - med) / iqr

    return pd.DataFrame(out, columns=[f"{c}_z" for c in cols], index=F.index)


def build_relative(X: np.ndarray, water_mask: np.ndarray,
                   group: Optional[np.ndarray] = None,
                   columns: Optional[Sequence[str]] = None):
    """
    features.build -> scene-relative, in one call. Returns (Z, ref).

    `columns` restricts to a feature subset BEFORE standardising, so the
    reference is computed only on columns that will actually be used.
    """
    from . import features as F

    feats = F.build(X)
    if columns is not None:
        feats = feats[list(columns)]
    ref = scene_reference(feats, water_mask, group)
    return to_scene_relative(feats, ref, group), ref


def water_mask_from_labels(y: np.ndarray) -> np.ndarray:
    """MARIDA's water classes as the reference population."""
    return np.isin(y, WATER_CLASSES)
