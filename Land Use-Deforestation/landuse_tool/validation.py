"""
Validation & accuracy assessment.

Implements the metrics that are standard in the land-change and remote-sensing
literature, so reported accuracy is defensible rather than optimistic:

  * confusion_matrix_report   - OA, user's/producer's accuracy (Congalton/Foody)
  * quantity_allocation_disagreement - Pontius & Millones (2011) components,
    which replace Kappa (they show Kappa is misleading for map comparison)
  * figure_of_merit           - Pontius et al. (2008): the land-change model
    metric. Compares observed vs predicted CHANGE, not overall agreement, so it
    is not inflated by persistence.
  * spatial_block_cv          - Roberts et al. (2017): random CV on spatially
    autocorrelated data seriously underestimates predictive error. This runs
    BOTH random and spatially-blocked CV so the gap is visible and reportable.
  * risk_map_auc              - ROC-AUC of a continuous risk/suitability map
    against observed change (for the deforestation risk map).
  * olofsson_area_estimate    - Olofsson et al. (2014) good-practice stratified
    area estimate with a 95% confidence interval.

References are listed in ANALYSIS_REPORT_DRAFT.md.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Confusion matrix / standard accuracy
# ---------------------------------------------------------------------------
def confusion_matrix_report(y_true, y_pred, class_names=None):
    """
    Overall accuracy plus per-class user's and producer's accuracy.

    user's accuracy (precision)    = correct / mapped as that class
    producer's accuracy (recall)   = correct / actually that class
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    total = cm.sum()
    oa = float(np.trace(cm) / total) if total else 0.0
    per_class = {}
    for c, i in idx.items():
        mapped = cm[:, i].sum()
        actual = cm[i, :].sum()
        ua = float(cm[i, i] / mapped) if mapped else float("nan")
        pa = float(cm[i, i] / actual) if actual else float("nan")
        f1 = (2 * ua * pa / (ua + pa)) if (ua and pa and np.isfinite(ua + pa)) else float("nan")
        name = (class_names or {}).get(int(c), f"Class {int(c)}")
        per_class[name] = {"users_accuracy": round(ua, 4),
                           "producers_accuracy": round(pa, 4),
                           "f1": round(f1, 4),
                           "n_reference": int(actual)}
    return {"overall_accuracy": round(oa, 4),
            "confusion_matrix": cm,
            "classes": [int(c) for c in classes],
            "per_class": per_class,
            "n": int(total)}


# ---------------------------------------------------------------------------
# 2. Quantity & allocation disagreement (Pontius & Millones 2011)
# ---------------------------------------------------------------------------
def quantity_allocation_disagreement(y_true, y_pred):
    """
    Decompose total disagreement into:
      quantity disagreement  - wrong PROPORTIONS of classes
      allocation disagreement- right proportions, wrong PLACES
    Reported as proportions of all pixels; they sum to (1 - overall accuracy).
    Recommended in place of Kappa.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    q_total, a_total = 0.0, 0.0
    for c in classes:
        ref = np.count_nonzero(y_true == c) / n
        cmp_ = np.count_nonzero(y_pred == c) / n
        agree = np.count_nonzero((y_true == c) & (y_pred == c)) / n
        q_total += abs(ref - cmp_)
        a_total += 2 * min(ref - agree, cmp_ - agree)
    quantity = q_total / 2.0
    allocation = a_total / 2.0
    oa = float(np.count_nonzero(y_true == y_pred) / n)
    return {"overall_accuracy": round(oa, 4),
            "quantity_disagreement": round(float(quantity), 4),
            "allocation_disagreement": round(float(allocation), 4),
            "total_disagreement": round(float(quantity + allocation), 4)}


# ---------------------------------------------------------------------------
# 3. Figure of Merit (Pontius et al. 2008) - THE land-change model metric
# ---------------------------------------------------------------------------
def figure_of_merit(ref_t1, ref_t2, pred_t2, mask=None):
    """
    Three-map comparison for a land-change simulation.

      hits              observed change, predicted change (correct)
      misses            observed change, predicted persistence
      false_alarms      observed persistence, predicted change
      wrong_hits        observed change and predicted change, but to the
                        wrong destination class
      FoM = hits / (hits + misses + false_alarms + wrong_hits)

    FoM ranges 0-100%. Published land-change models frequently score well under
    30%, because error commonly exceeds correctly predicted change - so a modest
    FoM is normal and should be reported honestly.
    """
    ref_t1 = np.asarray(ref_t1)
    ref_t2 = np.asarray(ref_t2)
    pred_t2 = np.asarray(pred_t2)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        ref_t1, ref_t2, pred_t2 = ref_t1[m], ref_t2[m], pred_t2[m]

    observed_change = ref_t1 != ref_t2
    predicted_change = ref_t1 != pred_t2

    hits = int(np.count_nonzero(observed_change & predicted_change & (pred_t2 == ref_t2)))
    wrong_hits = int(np.count_nonzero(observed_change & predicted_change & (pred_t2 != ref_t2)))
    misses = int(np.count_nonzero(observed_change & ~predicted_change))
    false_alarms = int(np.count_nonzero(~observed_change & predicted_change))
    correct_rejections = int(np.count_nonzero(~observed_change & ~predicted_change))

    denom = hits + misses + false_alarms + wrong_hits
    fom = (100.0 * hits / denom) if denom else 0.0

    # Null model: predict persistence everywhere (the baseline any model must beat)
    null_accuracy = 100.0 * correct_rejections / max(len(ref_t1), 1) if np.ndim(ref_t1) else 0.0
    model_accuracy = 100.0 * np.count_nonzero(pred_t2 == ref_t2) / max(ref_t2.size, 1)

    return {"figure_of_merit_pct": round(float(fom), 2),
            "hits": hits, "misses": misses, "false_alarms": false_alarms,
            "wrong_hits": wrong_hits, "correct_rejections": correct_rejections,
            "observed_change_px": int(np.count_nonzero(observed_change)),
            "predicted_change_px": int(np.count_nonzero(predicted_change)),
            "model_accuracy_pct": round(float(model_accuracy), 2),
            "null_model_accuracy_pct": round(float(
                100.0 * np.count_nonzero(ref_t1 == ref_t2) / max(ref_t2.size, 1)), 2)}


# ---------------------------------------------------------------------------
# 4. Spatial block cross-validation (Roberts et al. 2017)
# ---------------------------------------------------------------------------
def spatial_block_cv(X, y, coords, model=None, n_blocks=5, block_size=None,
                     seed=42, compare_random=True):
    """
    Cross-validate with SPATIAL blocks so training and test data are separated in
    space. Random CV lets neighbouring (autocorrelated) pixels appear in both
    folds, which inflates the score. Returns both so the gap is reportable.

    coords: (n, 2) array of (row, col) pixel positions for each sample.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score, GroupKFold
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                       n_jobs=-1, random_state=seed)

    X = np.asarray(X)
    y = np.asarray(y)
    coords = np.asarray(coords)

    # assign each sample to a spatial block (a coarse grid over the study area)
    if block_size is None:
        span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        block_size = max(int(span / max(n_blocks, 1)), 1)
    block_r = (coords[:, 0] // block_size).astype(int)
    block_c = (coords[:, 1] // block_size).astype(int)
    groups = block_r * (block_c.max() + 1) + block_c
    n_groups = len(np.unique(groups))
    folds = min(n_blocks, n_groups)

    out = {"block_size_px": int(block_size), "n_spatial_blocks": int(n_groups)}
    if folds < 2:
        logger.warning("Not enough spatial blocks for blocked CV (%d)", n_groups)
        out["spatial_cv_mean"] = float("nan")
    else:
        scores = cross_val_score(clone(model), X, y, groups=groups,
                                 cv=GroupKFold(n_splits=folds), n_jobs=-1)
        out["spatial_cv_mean"] = round(float(scores.mean()), 4)
        out["spatial_cv_std"] = round(float(scores.std()), 4)
        out["spatial_cv_folds"] = folds

    if compare_random:
        rnd = cross_val_score(clone(model), X, y, cv=5, n_jobs=-1)
        out["random_cv_mean"] = round(float(rnd.mean()), 4)
        out["random_cv_std"] = round(float(rnd.std()), 4)
        if np.isfinite(out.get("spatial_cv_mean", np.nan)):
            out["optimism_gap"] = round(out["random_cv_mean"] - out["spatial_cv_mean"], 4)
    return out


# ---------------------------------------------------------------------------
# 5. ROC-AUC for a continuous risk / suitability map
# ---------------------------------------------------------------------------
def risk_map_auc(risk_map, observed_change_mask, eligible_mask=None):
    """
    How well does the risk/suitability map rank pixels that actually changed
    above those that did not? AUC 0.5 = no skill, 1.0 = perfect ranking.
    Evaluated only over eligible pixels (e.g. standing forest).
    """
    from sklearn.metrics import roc_auc_score
    risk = np.asarray(risk_map, dtype="float64")
    obs = np.asarray(observed_change_mask, dtype=bool)
    valid = np.isfinite(risk)
    if eligible_mask is not None:
        valid &= np.asarray(eligible_mask, dtype=bool)
    y = obs[valid].astype(int)
    s = risk[valid]
    if y.sum() == 0 or y.sum() == y.size:
        return {"auc": float("nan"), "n": int(y.size),
                "note": "only one class present in evaluation area"}
    return {"auc": round(float(roc_auc_score(y, s)), 4),
            "n": int(y.size), "n_positive": int(y.sum())}


# ---------------------------------------------------------------------------
# 6. Olofsson et al. (2014) good-practice area estimation
# ---------------------------------------------------------------------------
def olofsson_area_estimate(confusion_matrix, mapped_area_px, pixel_area_ha=1.0,
                           class_labels=None, z=1.96):
    """
    Stratified estimator of the TRUE area of each class with a confidence
    interval, correcting the mapped area for classification error.

    confusion_matrix : reference (rows) x map (cols) sample counts
    mapped_area_px   : array of mapped pixel counts per class (same order)

    Returns per class: mapped area, adjusted (unbiased) area, and 95% CI. This
    is the recommended way to report change area, because simply counting mapped
    pixels is biased whenever the map has error.
    """
    cm = np.asarray(confusion_matrix, dtype="float64")
    n_map = cm.sum(axis=0)                       # samples per MAP class (columns)
    total_px = float(np.sum(mapped_area_px))
    w = np.asarray(mapped_area_px, dtype="float64") / total_px   # stratum weights

    k = cm.shape[0]
    # p_ij: estimated population proportions
    p = np.zeros_like(cm)
    for j in range(k):
        if n_map[j] > 0:
            p[:, j] = w[j] * cm[:, j] / n_map[j]

    results = {}
    for i in range(k):
        p_i = p[i, :].sum()                      # estimated proportion of class i
        # variance of the stratified estimator (Olofsson et al. 2014, eq. 10)
        var = 0.0
        for j in range(k):
            if n_map[j] > 1:
                p_ij = cm[i, j] / n_map[j]
                var += (w[j] ** 2) * p_ij * (1 - p_ij) / (n_map[j] - 1)
        se = float(np.sqrt(var))
        label = (class_labels or {}).get(i, f"Class {i}")
        results[label] = {
            "mapped_area_ha": round(float(mapped_area_px[i] * pixel_area_ha), 2),
            "adjusted_area_ha": round(float(p_i * total_px * pixel_area_ha), 2),
            "ci95_ha": round(float(z * se * total_px * pixel_area_ha), 2),
            "estimated_proportion": round(float(p_i), 5),
        }
    return results


# ---------------------------------------------------------------------------
# Convenience: full report for a land-change simulation
# ---------------------------------------------------------------------------
def validate_simulation(ref_t1, ref_t2, pred_t2, mask=None, class_names=None):
    """Run the standard battery on a simulated land-cover map and return a dict."""
    ref_t1 = np.asarray(ref_t1); ref_t2 = np.asarray(ref_t2); pred_t2 = np.asarray(pred_t2)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a, b, c = ref_t1[m], ref_t2[m], pred_t2[m]
    else:
        a, b, c = ref_t1.ravel(), ref_t2.ravel(), pred_t2.ravel()
    return {
        "accuracy": confusion_matrix_report(b, c, class_names),
        "disagreement": quantity_allocation_disagreement(b, c),
        "figure_of_merit": figure_of_merit(ref_t1, ref_t2, pred_t2, mask),
    }
