"""
Does a model trained on MARIDA transfer to Guinea? Measure, do not assume.

WHY THIS MODULE DECIDES WHETHER ANY OF THE REST IS DEPLOYABLE

The MARIDA results are strong: on held-out locations the classifier separates
annotated debris from every other floating class at ROC-AUC ~0.997, against
~0.70 for the deployed FDI rule. It is tempting to stop there and ship it.

That would be premature, because the classifier would then be fed Guinean
pixels drawn from a possibly different distribution than the one it was
fitted on. What a tree ensemble does under such a shift is not graceful
degradation: outside the range of its training data a tree cannot
interpolate, so it returns whatever the nearest leaf held, which is arbitrary
rather than merely imprecise. Measuring the shift is a precondition, not a
refinement.

TWO MEASUREMENT MISTAKES THIS MODULE MAKES EASY TO AVOID, BECAUSE IT MADE
THEM FIRST

1. THE WRONG UNIT. A band ratio is scale-free, and scale is exactly what
   matters here. MARIDA's median shortwave-infrared reflectance over water is
   0.006; Conakry's is 0.014. The ratio is 2.4x and looks decisive. Expressed
   in the units the model was fitted in it is about half a training spread --
   comfortably inside the training range. The operative test is the
   standardised shift in FEATURE space plus the fraction of deployment pixels
   outside the training support; `compare_water_populations` is retained for
   context and its ratio flag is explicitly advisory.

2. THE WRONG POPULATION. Comparing the pixels a detector FLAGS against the
   training set's WATER pixels makes every flagged pixel look extreme on FDI
   -- necessarily, since flagged pixels are defined as the upper tail of the
   water FDI distribution. That is a tautology, not a shift. Use
   `flagged_shift` to compare like with like.

WHAT NO DISTRIBUTIONAL CHECK CAN ADDRESS

Two limitations survive any amount of measurement. MARIDA annotates single
acquisitions while Guinea composites monthly medians, and a median over a
month suppresses transient floating material -- which is what debris is --
while leaving turbid plumes, shallow bathymetry and moored vessels intact.
And MARIDA's debris label is expert visual interpretation of Sentinel-2
imagery at sites with reported debris events, not physical sampling. Both are
stated alongside every transfer verdict rather than folded into it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import features as F
from .data import BANDS, WATER_CLASSES

# The deployed pipeline downloads only these seven. Anything the model needs
# beyond them would require changing the fetch, so they bound what is
# deployable without extra cost. B06 joined the fetch when the FDI was
# corrected to Biermann et al.'s RE2 band (config.S2_CONFIG["lambda_nm"]).
PIPELINE_BANDS = ["B03", "B04", "B06", "B08", "B8A", "B11", "B12"]
PIPELINE_FEATURES = PIPELINE_BANDS + ["FDI", "NDVI", "NDWI", "FAI", "PI"]

# Composite band names -> MARIDA band names.
COMPOSITE_TO_MARIDA = {"B3": "B03", "B4": "B04", "B6": "B06", "B8": "B08",
                       "B8A": "B8A", "B11": "B11", "B12": "B12"}


def is_atmospherically_corrected(X: np.ndarray) -> Dict:
    """
    Evidence on whether reflectances have had an atmospheric correction.

    Top-of-atmosphere reflectance is a ratio of measured to incident radiance
    and cannot be negative. Surface-reflectance retrievals routinely go
    slightly negative over dark water, where the correction subtracts a path
    radiance estimate larger than the measured signal. A non-trivial negative
    fraction is therefore positive evidence of Level-2A-style processing.
    """
    neg_any = float((X < 0).any(axis=1).mean())
    return {"min_reflectance": float(X.min()),
            "fraction_with_negative_band": neg_any,
            "verdict": ("atmospherically corrected (L2A-like)" if neg_any > 1e-4
                        else "no evidence of correction (possibly L1C/TOA)")}


def compare_water_populations(X_marida: np.ndarray, y_marida: np.ndarray,
                              composite_bands: Dict[str, np.ndarray],
                              water_mask: np.ndarray,
                              max_px: int = 200_000,
                              seed: int = 20240119):
    """
    Band-by-band comparison of WATER pixels in training and deployment.

    Water is the right population to compare: it is the background the FDI
    threshold is estimated over, so a shift here moves the threshold itself,
    not merely the classifier's inputs.

    Returns a table with a ratio of medians per band. A ratio far from 1
    identifies a band the model cannot be trusted on.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = {b: i for i, b in enumerate(BANDS)}
    mw = X_marida[np.isin(y_marida, WATER_CLASSES)]

    sel = np.flatnonzero(np.asarray(water_mask).ravel())
    if sel.size > max_px:
        sel = rng.choice(sel, max_px, replace=False)

    rows = []
    for comp_name, mar_name in COMPOSITE_TO_MARIDA.items():
        if comp_name not in composite_bands:
            continue
        a = mw[:, idx[mar_name]]
        b = np.asarray(composite_bands[comp_name]).ravel()[sel]
        b = b[np.isfinite(b)]
        if b.size == 0:
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        rows.append({
            "band": mar_name,
            "marida_median": ma, "deploy_median": mb,
            "ratio": mb / ma if ma else np.nan,
            "marida_p05": float(np.quantile(a, 0.05)),
            "marida_p95": float(np.quantile(a, 0.95)),
            "deploy_p95": float(np.quantile(b, 0.95)),
            # Share of deployment pixels outside the training range entirely.
            # A tree cannot extrapolate: these get whatever the edge leaf held.
            "deploy_outside_training_range": float(
                np.mean((b < a.min()) | (b > a.max()))),
        })
    df = pd.DataFrame(rows)
    df["shift_flag"] = np.where(
        (df["ratio"] > 1.5) | (df["ratio"] < 0.67), "SHIFTED", "ok")
    return df


def feature_shift(X_marida: np.ndarray, y_marida: np.ndarray,
                  X_deploy: np.ndarray, features: Optional[List[str]] = None):
    """
    The same comparison in FEATURE space, which is what the model sees.

    Reported separately from the band comparison because the indices are
    ratios and differences: a shift that is modest in two bands can be large
    in an index built from them, or can cancel. FDI in particular is a
    difference against a SWIR-anchored baseline, so a SWIR shift propagates
    into it directly.
    """
    import pandas as pd

    cols = features or PIPELINE_FEATURES
    A = F.build(X_marida[np.isin(y_marida, WATER_CLASSES)])[cols]
    Bd = F.build(X_deploy)[cols]

    rows = []
    for c in cols:
        a, b = A[c].to_numpy(), Bd[c].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        # Standardised difference of medians, scaled by the TRAINING spread:
        # how far the deployment centre sits in units the model was fitted on.
        spread = np.quantile(a, 0.84) - np.quantile(a, 0.16)
        rows.append({
            "feature": c,
            "marida_median": float(np.median(a)),
            "deploy_median": float(np.median(b)),
            "shift_in_training_sd": float(
                (np.median(b) - np.median(a)) / spread) if spread else np.nan,
            "deploy_outside_training_range": float(
                np.mean((b < a.min()) | (b > a.max()))),
        })
    df = pd.DataFrame(rows)
    df["severity"] = np.where(df["shift_in_training_sd"].abs() > 2, "SEVERE",
                       np.where(df["shift_in_training_sd"].abs() > 1, "moderate", "ok"))
    return df.reindex(df["shift_in_training_sd"].abs().sort_values(
        ascending=False).index)


def flagged_shift(X_marida: np.ndarray, y_marida: np.ndarray,
                  group_marida: np.ndarray, X_deploy: np.ndarray,
                  features: Optional[List[str]] = None):
    """
    Compare the pixels the RULE FLAGS in training against those it flags in
    deployment -- the like-for-like population.

    `feature_shift` compares deployment against the training WATER
    population, which is the right reference for the threshold itself but the
    wrong one for the classifier's inputs: the classifier only ever sees
    flagged pixels, and flagged pixels are by construction in the tail of the
    water distribution. Measuring them against water therefore reports a
    100% out-of-range rate on FDI no matter how well the two settings agree.

    Returns the same columns as `feature_shift`, plus the share of deployment
    pixels lying inside the training population's 1st-99th percentile band on
    EVERY feature -- the quantity that says whether the model is
    interpolating.
    """
    import pandas as pd

    from . import baseline as _B

    cols = features or PIPELINE_FEATURES
    r = _B.predict(X_marida, y_marida, group=group_marida)
    A = F.build(X_marida)[cols][r["pred"] & r["scored"]]
    Bd = F.build(X_deploy)[cols]

    rows = []
    for c in cols:
        a, b = A[c].to_numpy(), Bd[c].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 10 or b.size == 0:
            continue
        spread = np.quantile(a, 0.84) - np.quantile(a, 0.16)
        lo, hi = np.quantile(a, [0.01, 0.99])
        rows.append({
            "feature": c,
            "marida_flagged_median": float(np.median(a)),
            "deploy_flagged_median": float(np.median(b)),
            "shift_in_training_sd": float(
                (np.median(b) - np.median(a)) / spread) if spread else np.nan,
            "deploy_outside_1_99pct": float(np.mean((b < lo) | (b > hi))),
        })
    df = pd.DataFrame(rows)
    df["severity"] = np.where(df["shift_in_training_sd"].abs() > 2, "SEVERE",
                       np.where(df["shift_in_training_sd"].abs() > 1,
                                "moderate", "ok"))

    lo = A[cols].quantile(0.01)
    hi = A[cols].quantile(0.99)
    outside_any = ((Bd[cols] < lo) | (Bd[cols] > hi)).any(axis=1)
    df.attrs["n_marida_flagged"] = int(len(A))
    df.attrs["n_deploy_flagged"] = int(len(Bd))
    df.attrs["frac_outside_any"] = float(outside_any.mean())
    return df.reindex(df["shift_in_training_sd"].abs().sort_values(
        ascending=False).index)


def verdict(band_df, feature_df, composite_note: str = "",
            frac_outside: float = None) -> str:
    """
    One paragraph a methodology section can quote.

    THE CRITERION, AND A CORRECTION TO AN EARLIER ONE

    An earlier version of this function keyed on the BAND RATIO in
    `band_df` -- flagging a shift whenever a band's median moved by more than
    50%. That test is wrong for this purpose and produced a false alarm.

    Shortwave infrared over water is a small number: MARIDA's median B11 is
    0.006 and Conakry's is 0.014. The ratio is 2.4x, which sounds decisive,
    but MARIDA's own B11 distribution over water is wide enough that 0.014
    sits about half a training spread from its centre -- comfortably inside
    the range the model was fitted on. A ratio is the wrong unit because it
    is scale-free in a setting where scale is exactly what matters.

    What matters for a tree ensemble is whether deployment pixels fall
    OUTSIDE the training range, because that is the only regime where a tree
    cannot interpolate. So the decisive quantity is the standardised shift in
    FEATURE space, measured in training spreads, plus the fraction of
    deployment pixels outside the training support. The band ratio is
    retained as context and explicitly demoted.

    One further subtlety, learned the same way: the comparison population
    must be LIKE FOR LIKE. Comparing the pixels a detector flags against the
    training set's WATER population makes every flagged pixel look extreme on
    FDI, because flagged pixels are defined as the tail of the water FDI
    distribution. That is a tautology, not a shift. Flagged pixels must be
    compared against flagged pixels.
    """
    severe = feature_df.loc[feature_df["severity"] == "SEVERE", "feature"].tolist()
    moderate = feature_df.loc[feature_df["severity"] == "moderate", "feature"].tolist()
    ratio_flagged = band_df.loc[band_df["shift_flag"] == "SHIFTED", "band"].tolist()
    worst = feature_df["shift_in_training_sd"].abs().max()

    ok = not severe
    head = "TRANSFER SUPPORTED. " if ok else "TRANSFER NOT SUPPORTED. "
    body = (
        f"In the feature space the model actually uses, the largest "
        f"standardised shift between training and deployment is "
        f"{worst:.2f} training spreads; {len(severe)} features exceed two "
        f"spreads and {len(moderate)} exceed one. ")
    if frac_outside is not None:
        body += (f"{1 - frac_outside:.0%} of deployment pixels fall inside the "
                 f"training support on every feature, so the model is "
                 f"interpolating rather than extrapolating for the great "
                 f"majority of what it scores. ")
    if ratio_flagged:
        body += (f"Raw band medians differ by more than 50% in "
                 f"{', '.join(ratio_flagged)}, but that ratio is not the "
                 f"operative test: shortwave infrared over water is a small "
                 f"number, so a large ratio can still be a small distance in "
                 f"the units the model was fitted in. ")
    if ok:
        body += ("Predictions for this deployment are therefore usable, "
                 "subject to the caveats on label meaning and geographic "
                 "representation that no distributional check can address. ")
    else:
        body += ("Scores produced under this shift must be reported as a "
                 "DIAGNOSTIC of distributional difference, not as evidence "
                 "about what the deployment scene contains. ")
    if composite_note:
        body += composite_note
    return head + body


COMPOSITE_CAVEAT = (
    "Independently of the measured shift, Guinea runs a monthly MEDIAN "
    "composite while MARIDA annotates single acquisitions. A median over a "
    "month suppresses transient floating material -- which is what debris is "
    "-- while preserving persistent features such as turbid plumes, shallow "
    "bathymetry and moored vessels. Deployment is therefore biased toward the "
    "confuser classes by construction, and no amount of retraining on "
    "single-date imagery removes that bias.")
