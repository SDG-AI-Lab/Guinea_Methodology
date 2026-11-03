"""Estimate covariate-shift weights from dated features, without outcome labels."""

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def domain_feature_columns(names):
    return np.array([i for i, name in enumerate(names) if
        name.startswith(("modis_current_", "landsat_current_mean_", "current_class_",
                         "fraction_class_", "current_log_", "annual_lag_0_"))
        and not any(token in name for token in ("missing", "support", "valid_periods"))], dtype=int)


def estimate_density_weights(historical, starting, feature_names, groups, jobs=4, seed=42,
                             max_domain_rows=30000, trees=100):
    """Three spatial folds; balanced domain sampling, clipped likelihood ratios.

    Each pair of rows describes the same training location at two starting
    dates. A domain classifier never sees that location's block when estimating
    its weight. It receives no future land-cover outcomes.
    """
    if historical.shape != starting.shape or len(groups) != len(historical):
        raise ValueError("Domain feature rows/groups must align")
    columns = domain_feature_columns(feature_names)
    if not len(columns):
        raise ValueError("No eligible starting-condition features for adaptation")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Domain adaptation needs at least three training blocks")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(unique), 3)
    h, s = historical[:, columns], starting[:, columns]
    historical_probability = np.zeros(len(h), dtype=np.float32)
    starting_probability = np.zeros(len(s), dtype=np.float32)
    for i, blocks in enumerate(folds):
        heldout = np.isin(groups, blocks)
        eligible = np.flatnonzero(~heldout)
        sampled = rng.choice(eligible, min(max_domain_rows, len(eligible)), replace=False)
        X = np.concatenate([h[sampled], s[sampled]])
        y = np.repeat([0, 1], len(sampled))
        model = XGBClassifier(n_estimators=trees, max_depth=3, learning_rate=.05,
            min_child_weight=20, reg_lambda=10, subsample=.8, colsample_bytree=.8,
            tree_method="hist", objective="binary:logistic", eval_metric="logloss",
            n_jobs=jobs, random_state=seed+i)
        model.fit(X, y)
        historical_probability[heldout] = model.predict_proba(h[heldout])[:, 1]
        starting_probability[heldout] = model.predict_proba(s[heldout])[:, 1]
    ratio = historical_probability / np.maximum(1-historical_probability, 1e-6)
    weights = np.clip(ratio, .2, 5.)
    weights /= weights.mean()
    report = dict(domain_features=[feature_names[i] for i in columns], spatial_folds=3,
        outcome_labels_used=False, ratio_clip_before_normalizing=[.2, 5.],
        oof_domain_auc=float(roc_auc_score(np.repeat([0, 1], len(h)),
            np.concatenate([historical_probability, starting_probability]))),
        weight_min=float(weights.min()), weight_max=float(weights.max()),
        effective_sample_size=float(weights.sum()**2 / np.sum(weights**2)),
        original_sample_size=len(weights),
        assumption="Conditional destination relationship is sufficiently stable; this cannot correct arbitrary concept shift.")
    return weights.astype(np.float32), report
