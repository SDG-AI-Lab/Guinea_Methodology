"""
Deforestation driver analysis.

Answers "what is driving forest loss, and where is at risk?" by:
  1. assembling a driver stack (accessibility, terrain, population, climate, fire,
     and deforestation-specific proximity layers) aligned to the analysis grid;
  2. sampling deforested pixels (label 1) vs stable-forest pixels (label 0);
  3. fitting a risk model (RandomForest or logistic regression) whose feature
     importances ARE the driver attribution;
  4. predicting a continuous risk map P(deforestation) over remaining forest.

Reuses the Land Use platform: data_loader.assemble_predictor_stack for the shared
predictors, utils/scipy for the derived proximity layers, sklearn for the model.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derived proximity drivers
# ---------------------------------------------------------------------------
def distance_to(mask, profile=None, resolution_m=100.0):
    """
    Euclidean distance (metres) from every pixel to the nearest True pixel in
    `mask`. Used for distance-to-roads, distance-to-prior-loss (frontier), and
    distance-to-forest-edge.
    """
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype="float32")
    if profile is not None:
        t = profile["transform"]
        px = (abs(t.a) + abs(t.e)) / 2.0
        # geographic grid -> convert degrees to ~metres at the scene centre
        if getattr(profile.get("crs"), "is_geographic", False):
            px = px * 111_320.0
    else:
        px = resolution_m
    return (distance_transform_edt(~mask) * px).astype("float32")


def forest_edge(forest_mask):
    """Boolean forest-edge pixels (forest adjacent to non-forest)."""
    from scipy.ndimage import binary_erosion
    forest_mask = np.asarray(forest_mask, dtype=bool)
    interior = binary_erosion(forest_mask, iterations=1, border_value=0)
    return forest_mask & ~interior


def build_driver_stack(shared_predictors, forest_mask, prior_loss_mask,
                       roads_mask=None, fire_mask=None, profile=None,
                       include_forest_edge=False, commodity_price_surface=None):
    """
    Combine shared predictors (from data_loader.assemble_predictor_stack) with
    deforestation-specific proximity/fire drivers into one {name: array} stack.

    shared_predictors: {name: 2D array} already aligned (slope, elevation,
                       population, precipitation, distance-to-roads, ...).
    prior_loss_mask:   cumulative loss BEFORE the modelled period (frontier).
    include_forest_edge: if True, adds dist_forest_edge to the stack. Default
        False -- confirmed via true temporal hold-out (calibrate 2015-2018,
        test 2019-2022, Guinea pilot) that this feature encodes each period's
        specific frontier geometry rather than a transferable risk pattern:
        including it collapses hold-out AUC to ~0.49 (chance level); excluding
        it gives 0.686, stable across three independent historical splits.
        A sweep of reformulations (capped/decayed/log/binned distance) was
        also tested and none improved on simple exclusion. See
        Methodology_Deforestation_Guinea.docx Section 9.1 for the full
        diagnostic. Only set True for descriptive/associational driver
        attribution (e.g. Table 4-style importance ranking), never for
        forward risk prediction or scenario projection.
    commodity_price_surface: optional 2D array (IDW-interpolated WFP market
        prices, see data_loader.rasterize_market_points), added as
        "commodity_price" if given. ADOPTED per config.dcfg.WFP_PRICES_SOURCE
        (commodity="Oil (palm)", reference_year=2018) -- confirmed via
        ablation + rolling-origin hold-out to improve AUC on 3/3 historical
        splits (+0.0081, +0.0063, +0.0021), the same bar dist_forest_edge's
        exclusion had to clear. Caller is responsible for fetching/rasterizing
        this (pipeline.run_from_arrays does this automatically using the
        config's commodity/reference_year) -- this function just adds
        whatever is passed in.
    """
    stack = dict(shared_predictors)
    stack["dist_prior_loss"] = distance_to(prior_loss_mask, profile)
    if include_forest_edge:
        stack["dist_forest_edge"] = distance_to(forest_edge(forest_mask), profile)
    if roads_mask is not None and "ROADS_distance" not in stack:
        stack["dist_roads"] = distance_to(roads_mask, profile)
    if fire_mask is not None:
        stack["fire"] = np.asarray(fire_mask, dtype="float32")
    if commodity_price_surface is not None:
        stack["commodity_price"] = np.asarray(commodity_price_surface, dtype="float32")
    return stack


# ---------------------------------------------------------------------------
# Sampling: deforested (1) vs stable forest (0)
# ---------------------------------------------------------------------------
def sample_training(deforested_mask, stable_forest_mask, driver_stack,
                    samples_per_class=5000, seed=42):
    """
    Build a balanced (X, y, names) training set. Only pixels with finite values
    across all drivers are used. Positive = deforested; negative = stayed forest.
    """
    names = list(driver_stack.keys())
    cube = np.stack([np.asarray(driver_stack[n], dtype="float32") for n in names], axis=-1)
    finite = np.all(np.isfinite(cube), axis=-1)

    pos = np.argwhere(np.asarray(deforested_mask, dtype=bool) & finite)
    neg = np.argwhere(np.asarray(stable_forest_mask, dtype=bool) & finite)
    if len(pos) < 10 or len(neg) < 10:
        logger.warning("Driver sampling: too few samples (pos=%d neg=%d)", len(pos), len(neg))
        return None, None, names

    rng = np.random.default_rng(seed)
    n = min(samples_per_class, len(pos), len(neg))
    pos = pos[rng.choice(len(pos), n, replace=False)]
    neg = neg[rng.choice(len(neg), n, replace=False)]
    coords = np.vstack([pos, neg])
    X = cube[coords[:, 0], coords[:, 1], :]
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return X, y, names


# ---------------------------------------------------------------------------
# Model + importances + risk map
# ---------------------------------------------------------------------------
def fit_risk_model(X, y, model_type="random_forest", seed=42):
    """Fit the risk model. Returns (model, metrics_dict)."""
    from sklearn.model_selection import cross_val_score
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=18,
                                       min_samples_leaf=5, n_jobs=-1,
                                       random_state=seed, class_weight="balanced")
    try:
        cv = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        cv_mean, cv_std = float(cv.mean()), float(cv.std())
    except Exception:
        cv_mean, cv_std = float("nan"), float("nan")
    model.fit(X, y)
    return model, {"cv_accuracy_mean": round(cv_mean, 4), "cv_accuracy_std": round(cv_std, 4),
                   "n_samples": int(len(y))}


def driver_importance(model, names):
    """Return {driver: importance} sorted desc (RF importances or |logit coef|)."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:  # logistic pipeline
        try:
            imp = np.abs(model[-1].coef_[0])
            imp = imp / imp.sum()
        except Exception:
            return {}
    order = np.argsort(imp)[::-1]
    return {names[i]: round(float(imp[i]), 4) for i in order}


def predict_risk_map(model, driver_stack, names, forest_mask):
    """
    P(deforestation) in [0,1] over remaining-forest pixels (NaN elsewhere).
    Only forest pixels are scored (that's where future loss can occur).
    """
    cube = np.stack([np.asarray(driver_stack[n], dtype="float32") for n in names], axis=-1)
    h, w, _ = cube.shape
    forest = np.asarray(forest_mask, dtype=bool)
    finite = np.all(np.isfinite(cube), axis=-1)
    score_mask = forest & finite
    risk = np.full((h, w), np.nan, dtype="float32")
    idx = np.argwhere(score_mask)
    if idx.size == 0:
        return risk
    proba = model.predict_proba(cube[score_mask])[:, 1]
    risk[score_mask] = proba.astype("float32")
    return risk