"""
Validation for the DPWMS hotspot-detection pipeline.

Scientific rationale (for the methodology document)
-----------------------------------------------------
Unlike the deforestation/land-use models, the FDI mesoscale pipeline has no
supervised training step — it's an unsupervised spectral-index + Otsu-
threshold classifier. Its "accuracy" claim therefore rests on three different
kinds of evidence, all provided here:

1. Reference-point validation: precision/recall/F1 of the hotspot binary mask
   against any known plastic-accumulation / dumpsite ground-truth points
   (field surveys, prior confirmed reports). This is the only true accuracy
   check and should be reported whenever any reference points exist, however
   few — report sample size honestly (see `hotspot_reference_validation`).
2. Threshold sensitivity: how much the mapped hotspot area changes when the
   Otsu threshold is perturbed, and how often Otsu vs the manual fallback is
   actually used across scenes. A method whose output changes wildly for
   small threshold shifts is not a method to report with false confidence.
3. `confusion_matrix_report` / `spatial_block_cv`: self-contained, standard
   implementations (sklearn-based), kept here for if/when a supervised
   classifier (e.g. a trained dumpsite CNN) replaces or supplements Otsu
   thresholding. NOT imported from landuse_tool -- this module has no
   dependency on landuse_tool's internals at all, deliberately, since
   assumptions about landuse_tool's exact module layout/signatures have
   repeatedly not matched the actual deployed files.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd

from . import config as pwcfg
from . import detection as D

logger = logging.getLogger(__name__)

__all__ = ["confusion_matrix_report", "spatial_block_cv",
          "hotspot_reference_validation", "photo_interpretation_validation",
          "threshold_sensitivity_analysis", "otsu_fallback_rate"]


# ============================================================================
# PHOTO-INTERPRETATION VALIDATION  (the figure to actually report)
# ============================================================================
def photo_interpretation_validation(labels, ambiguous_label: str = "ambiguous") -> Dict:
    """
    Precision by confidence band from a photo-interpreted review table.

    Why this exists instead of `hotspot_reference_validation`
    ---------------------------------------------------------
    The review set is generated FROM the detections: every flagged-patch row is
    the centroid of a patch the detector already produced. Feeding the confirmed
    positives back through `hotspot_reference_validation` therefore measures
    nothing -- every point lies inside a patch by construction, so recall is
    1.0 trivially and "precision" is just the positive rate re-labelled. That
    circularity is the reason the earlier `reference_validation` block in
    metrics.json (recall 1.0, precision 0.055) was misleading and must not be
    quoted. This function reports what the review can actually support.

    Expected columns: `sample_type` ('flagged_patch' | 'unflagged_random'),
    `confidence_level`, `label` ('positive' | 'negative' | ambiguous_label).

    Ambiguous rows are EXCLUDED from the precision denominator rather than
    counted as errors, and the excluded count is reported alongside. Both
    conventions are defensible; reporting only one number without the ambiguous
    share is not, because at 10 m a large fraction of patches genuinely cannot
    be adjudicated by eye. Treating every ambiguous row as a false positive
    gives the pessimistic bound, also returned here as `precision_worst_case`.

    Recall is NOT returned as a number. The unflagged sample establishes only
    whether obvious debris was missed; with no positives found in it there is no
    denominator, and a recall figure would imply a wall-to-wall reference map
    that does not exist.
    """
    import pandas as pd

    df = labels if hasattr(labels, "columns") else pd.read_csv(labels)
    flagged = df[df["sample_type"] == "flagged_patch"]
    unflagged = df[df["sample_type"] == "unflagged_random"]

    def _band(sub) -> Dict:
        decided = sub[sub["label"].isin(["positive", "negative"])]
        n_pos = int((sub["label"] == "positive").sum())
        n_amb = int((sub["label"] == ambiguous_label).sum())
        return {
            "n": int(len(sub)),
            "n_positive": n_pos,
            "n_negative": int((sub["label"] == "negative").sum()),
            "n_ambiguous": n_amb,
            "n_decided": int(len(decided)),
            "precision": round(float((decided["label"] == "positive").mean()), 3)
                         if len(decided) else None,
            "precision_worst_case": round(n_pos / len(sub), 3) if len(sub) else None,
        }

    by_band = {b: _band(flagged[flagged["confidence_level"] == b])
               for b in ("HIGH", "MEDIUM", "LOW")
               if (flagged["confidence_level"] == b).any()}

    overall = _band(flagged)
    n_miss = int((unflagged["label"] == "positive").sum())
    bands_present = [b for b in ("HIGH", "MEDIUM", "LOW") if b in by_band]
    ordered = [by_band[b]["precision"] for b in bands_present
               if by_band[b]["precision"] is not None]

    return {
        "n_reviewed": int(len(df)),
        "overall": overall,
        "by_confidence": by_band,
        "bands_monotonic": (ordered == sorted(ordered, reverse=True)) if ordered else None,
        "recall_check": {
            "n_unflagged_reviewed": int(len(unflagged)),
            "n_obvious_misses": n_miss,
            "n_ambiguous": int((unflagged[  "label"] == ambiguous_label).sum()),
            "note": (("No confirmed debris found in the unflagged sample, so no "
                      "evidence of systematic misses. " if n_miss == 0 else
                      f"{n_miss} unflagged location(s) show debris-like material the "
                      f"layer did not publish: evidence of misses. ") +
                     "This is NOT a recall figure: it has no positive denominator "
                     "and no wall-to-wall reference map behind it."),
        },
        "method": ("Photo-interpretation of Sentinel-2 chips by a single "
                   "interpreter. Single-interpreter review carries no "
                   "inter-rater agreement estimate; a second interpreter on a "
                   "subset would give Cohen's kappa and should precede any "
                   "publication-grade accuracy claim."),
    }


# ============================================================================
# SELF-CONTAINED SUPERVISED-VALIDATION HELPERS (not yet used by this
# pipeline -- kept ready for a future supervised dumpsite classifier)
# ============================================================================
def confusion_matrix_report(y_true: Sequence, y_pred: Sequence,
                            labels: Optional[Sequence] = None) -> Dict:
    """Confusion matrix + per-class precision/recall/F1, via scikit-learn.
    For a future supervised classifier (e.g. the macro dumpsite model) --
    nothing in the current unsupervised FDI pipeline calls this yet."""
    from sklearn.metrics import confusion_matrix, classification_report

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    label_list = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    report = classification_report(y_true, y_pred, labels=label_list,
                                   output_dict=True, zero_division=0)
    return {"confusion_matrix": cm.tolist(), "labels": label_list, "report": report}


def spatial_block_cv(X: np.ndarray, y: np.ndarray, coords: np.ndarray,
                     n_splits: int = 5, block_size: Optional[float] = None,
                     model=None, random_state: int = 42) -> Dict:
    """
    Spatial-block cross-validation (Roberts et al. 2017): groups samples into
    spatial blocks by coordinate binning rather than random folds, avoiding
    the optimistic bias random CV gives on spatially autocorrelated data.
    For a future supervised classifier -- nothing in the current unsupervised
    FDI pipeline calls this yet.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
    if block_size is None:
        span = max(coords[:, 0].max() - coords[:, 0].min(),
                   coords[:, 1].max() - coords[:, 1].min())
        block_size = (span / max(n_splits, 1)) or 1.0

    block_ids = (np.floor(coords[:, 0] / block_size).astype(int).astype(str) + "_" +
                np.floor(coords[:, 1] / block_size).astype(int).astype(str))

    base_model = model if model is not None else RandomForestClassifier(random_state=random_state)
    gkf = GroupKFold(n_splits=n_splits)
    fold_auc = []
    for train_idx, test_idx in gkf.split(X, y, groups=block_ids):
        m = clone(base_model)
        m.fit(X[train_idx], y[train_idx])
        scores = (m.predict_proba(X[test_idx])[:, 1] if hasattr(m, "predict_proba")
                 else m.predict(X[test_idx]))
        try:
            fold_auc.append(float(roc_auc_score(y[test_idx], scores)))
        except ValueError:
            continue  # a fold with only one class present; skip rather than crash

    return {
        "n_splits": n_splits, "n_blocks": len(set(block_ids)), "fold_auc": fold_auc,
        "mean_auc": float(np.mean(fold_auc)) if fold_auc else np.nan,
        "std_auc": float(np.std(fold_auc)) if fold_auc else np.nan,
    }


# ============================================================================
# REFERENCE-POINT VALIDATION
# ============================================================================
def hotspot_reference_validation(hotspot_gdf: gpd.GeoDataFrame,
                                 reference_points_gdf: gpd.GeoDataFrame,
                                 buffer_m: float = 100.0) -> Dict:
    """
    Precision / recall / F1 of detected hotspot patches against a set of
    reference points (confirmed plastic accumulations or dumpsites from field
    surveys). A reference point counts as a "hit" if it falls within
    `buffer_m` of any detected patch (patches, not pixels, since the ground
    truth here is point-based, not a wall-to-wall reference map).

    IMPORTANT — report `n_reference_points` alongside any score: with a
    small reference set (a handful of field-verified points, as is typical
    at this project stage) precision/recall estimates carry wide
    uncertainty and should be presented as provisional, exactly the way the
    land-use module flags its hold-out numbers rather than the inflated
    same-window back-test figure.

    NOT FOR PHOTO-INTERPRETED PATCH CENTROIDS. This function assumes the
    reference points were located INDEPENDENTLY of the detections (a field
    survey, a prior confirmed report). Passing back the confirmed positives from
    a review of the detector's own output makes recall 1.0 by construction and
    the result meaningless — use `photo_interpretation_validation` for that.
    The guard below catches the common case.
    """
    if reference_points_gdf.empty:
        return {"n_reference_points": 0, "precision": np.nan, "recall": np.nan,
               "f1": np.nan, "note": "No reference points supplied — cannot validate."}

    # REFUSE, don't warn. A warning still lets the circular numbers into the
    # results dict and from there into metrics.json, where nothing marks them as
    # unusable -- and with n >= 30 the "provisional" note below is suppressed too,
    # so they read as a solid recall of 1.0. Returning the refusal is what keeps
    # them out of the methodology document.
    cols = list(getattr(reference_points_gdf, "columns", []))
    if "sample_type" in cols or "interpreter_confidence" in cols:
        msg = ("Reference set came from the photo-interpretation review of this "
               "detector's own output (it carries review columns). Recall would be "
               "1.0 by construction and precision would divide by all detected "
               "patches rather than the reviewed ones. Refusing to compute. Use "
               "validation.photo_interpretation_validation() instead.")
        logger.warning(msg)
        return {"n_reference_points": int(len(reference_points_gdf)),
                "precision": None, "recall": None, "f1": None,
                "refused": True, "note": msg}
    if hotspot_gdf.empty:
        return {"n_reference_points": len(reference_points_gdf), "precision": np.nan,
               "recall": 0.0, "f1": 0.0, "note": "No patches detected in this window."}

    ref = reference_points_gdf.to_crs(hotspot_gdf.crs)
    patches_m = hotspot_gdf.to_crs(hotspot_gdf.estimate_utm_crs()) \
        if hotspot_gdf.crs and getattr(hotspot_gdf.crs, "is_geographic", False) else hotspot_gdf
    ref_m = ref.to_crs(patches_m.crs)
    buffered = patches_m.buffer(buffer_m)
    union = buffered.union_all() if hasattr(buffered, "union_all") else buffered.unary_union

    hits = ref_m.geometry.within(union)
    recall = float(hits.sum()) / len(ref_m)

    # Precision: fraction of detected patches that have >=1 reference point
    # nearby (a coarse proxy in the absence of a wall-to-wall absence map —
    # true negatives cannot be enumerated for a point-based reference set).
    matched_patches = patches_m.geometry.buffer(buffer_m).apply(
        lambda g: ref_m.geometry.within(g).any())
    precision = float(matched_patches.sum()) / len(patches_m) if len(patches_m) else np.nan

    f1 = (2 * precision * recall / (precision + recall)
         if (precision and recall and np.isfinite(precision + recall)) else np.nan)

    return {
        "n_reference_points": int(len(ref_m)),
        "n_patches": int(len(patches_m)),
        "recall": round(recall, 3),
        "precision": round(precision, 3) if np.isfinite(precision) else precision,
        "f1": round(f1, 3) if np.isfinite(f1) else f1,
        "buffer_m": buffer_m,
        "note": ("Provisional — based on a small point reference set; treat as "
                "indicative, not a publication-grade accuracy figure."
                if len(ref_m) < 30 else None),
    }


# ============================================================================
# THRESHOLD SENSITIVITY  (unsupervised-method robustness check)
# ============================================================================
def threshold_sensitivity_analysis(masked_fdi: np.ndarray, perturbation_frac: float = 0.10) -> Dict:
    """
    Report how much the mapped hotspot area changes if the Otsu threshold is
    perturbed by +/- `perturbation_frac`. A stable detector should show a
    roughly proportional (not wildly non-linear) change in area — large,
    non-monotonic swings indicate the FDI histogram is not cleanly bimodal
    for this scene and the Otsu threshold is unreliable there (favour the
    manual fallback and flag the scene for manual review).
    """
    threshold, method = D.threshold_with_fallback(masked_fdi)
    valid = ~np.isnan(masked_fdi)
    base_area_px = int(np.nansum(masked_fdi > threshold))

    results = {"base_threshold": threshold, "method": method, "base_area_px": base_area_px}
    for sign, label in [(-1, "lower"), (1, "upper")]:
        perturbed = threshold * (1 + sign * perturbation_frac)
        area_px = int(np.nansum(masked_fdi > perturbed))
        pct_change = (100.0 * (area_px - base_area_px) / base_area_px) if base_area_px else np.nan
        results[f"{label}_perturbed_threshold"] = perturbed
        results[f"{label}_perturbed_area_px"] = area_px
        results[f"{label}_area_change_pct"] = round(pct_change, 1) if np.isfinite(pct_change) else pct_change
    return results


def threshold_sensitivity_de_africa(fdi: np.ndarray, ndvi: np.ndarray,
                                    water: np.ndarray,
                                    perturbation_frac: float = 0.10,
                                    min_patch_px: Optional[int] = None,
                                    post_filter=None) -> Dict:
    """
    Sensitivity of the ACTUAL de_africa rule to a +/- `perturbation_frac` shift
    in both of its thresholds.

    Why this exists: `threshold_sensitivity_analysis` thresholds FDI alone. Under
    the de_africa rule that is not the deployed detector -- the conjunction with
    NDVI is what carries the selectivity. On Conakry 2024-01 the FDI-only
    population is 8,691 px against the rule's actual 261 px, a factor of 33, so
    the single-index figure describes a detector that is not being published.

    Also reports patch COUNT, not only area. For a hotspot map the count is the
    decision-relevant quantity: area can move substantially while the same
    places stay flagged, and a map whose patch identities are stable is usable
    even if its area estimate is soft.
    """
    import math

    from scipy import ndimage

    cfg = pwcfg.DETECTION_CONFIG
    if min_patch_px is None:
        # ceil, not int: a patch must REACH min_patch_area_m2 to be published, and
        # truncating admits patches below it. At 20 m, int(500/400) = 1 keeps
        # single-pixel (400 m2) patches that vectorize_hotspots drops, which is why
        # this reported 234 patches against the pipeline's actual 79.
        min_patch_px = max(1, math.ceil(cfg["min_patch_area_m2"] /
                                        (pwcfg.S2_CONFIG["download_scale_m"] ** 2)))

    in_water = np.nan_to_num(water, nan=0.0) > 0 if water is not None else np.ones_like(fdi, bool)
    valid = in_water & np.isfinite(fdi) & np.isfinite(ndvi)
    if not valid.any():
        return {"note": "No valid water pixels — sensitivity not computable."}

    f, v = fdi[valid], ndvi[valid]
    n_sigma = cfg["threshold_n_sigma"]
    t_fdi = float(f.mean() + n_sigma * f.std())
    t_ndvi = float(v.mean() + n_sigma * v.std())

    def _measure(tf, tv):
        """Area and count of the patches that would actually be PUBLISHED.

        Both figures exclude sub-minimum components, so the area reported here
        matches the pipeline's mapped area rather than counting pixels that
        vectorize_hotspots discards.
        """
        # 8-connected, matching detection.vectorize_hotspots' structure=ones((3,3)).
        # ndimage's DEFAULT is 4-connected, which splits diagonally-touching pixel
        # pairs into two single-pixel components that then fail the minimum-area
        # filter and vanish -- it reported 80 patches / 229 px against the
        # pipeline's 79 / 261 px, more patches from fewer pixels.
        m = valid & (fdi > tf) & (ndvi > tv)
        # The published detector, when a classifier follows the rule
        # (fdi_tool.debris_filter): perturb the rule, then filter as deployed.
        if post_filter is not None:
            m = post_filter(m)
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if not n:
            return 0, 0
        sizes = np.bincount(lab.ravel())[1:]
        keep = sizes >= min_patch_px
        return int(sizes[keep].sum()), int(keep.sum())

    base_px, base_n = _measure(t_fdi, t_ndvi)
    out = {
        "rule": "de_africa",
        "base_threshold_fdi": t_fdi,
        "base_threshold_ndvi": t_ndvi,
        "base_area_px": base_px,
        "base_n_patches": base_n,
        "perturbation_frac": perturbation_frac,
    }
    for sign, label in [(-1, "lower"), (1, "upper")]:
        k = 1 + sign * perturbation_frac
        px, n = _measure(t_fdi * k, t_ndvi * k)
        out[f"{label}_area_px"] = px
        out[f"{label}_n_patches"] = n
        out[f"{label}_area_change_pct"] = (
            round(100.0 * (px - base_px) / base_px, 1) if base_px else np.nan)
        out[f"{label}_patch_count_change_pct"] = (
            round(100.0 * (n - base_n) / base_n, 1) if base_n else np.nan)
    return out


def otsu_fallback_rate(methods_used: List[str]) -> Dict:
    """
    Across a batch of scenes, what fraction used Otsu vs the manual fallback
    (`FDI_MANUAL_THRESHOLD`)? A high fallback rate means the AOI/date window
    combination frequently lacks a clean bimodal FDI histogram (e.g. very
    small or very cloudy scenes) and the manual threshold is doing most of
    the real classification work — worth stating explicitly in the
    methodology document rather than implying Otsu is always in control.
    """
    n = len(methods_used)
    if n == 0:
        return {"n_scenes": 0, "otsu_rate_pct": np.nan, "fallback_rate_pct": np.nan}
    n_otsu = sum(1 for m in methods_used if m == "otsu")
    return {
        "n_scenes": n,
        "otsu_rate_pct": round(100.0 * n_otsu / n, 1),
        "fallback_rate_pct": round(100.0 * (n - n_otsu) / n, 1),
    }