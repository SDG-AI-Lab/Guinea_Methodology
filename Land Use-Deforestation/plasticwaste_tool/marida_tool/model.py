"""
The learned classifier, and the honest framing of what it can be asked for.

THE CLASS IMBALANCE IS THE WHOLE PROBLEM

Marine Debris is a fraction of a per cent of MARIDA's labelled pixels. A
classifier that predicts "not debris" everywhere scores over 99% accuracy, so
accuracy is not reported anywhere in this module. Precision, recall and the
composition of the positives are.

TWO MODELS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

  binary   debris vs everything else. Directly comparable to the FDI rule,
           which is also binary, so this is the comparison that decides
           whether machine learning earns its place in the pipeline.

  multiclass  all MARIDA classes at once. Not deployed, but it is the model
           that can say what a false positive actually IS -- and telling an
           analyst "this patch looks like Sargassum" is more useful than
           "this patch is not debris."

WHY A RANDOM FOREST

It is the baseline MARIDA's own authors report, which makes our numbers
comparable to a published reference rather than to nothing. It needs no
feature scaling, tolerates the mixed units in the feature table, and yields
importances that can be read against the spectral reasoning in the
methodology. Gradient boosting was tried and is reported in the notebook; it
does not change the conclusion, which is itself worth recording.

WHAT THIS MODEL MUST NOT BE ASKED

It cannot be asked whether a Guinean patch is plastic. It was trained on
eleven sites, none in West Africa, on a target class defined by MARIDA's
annotators from visual interpretation of Sentinel-2 imagery plus reports of
debris events -- not from anyone picking material out of the water. The
label means "annotated as marine debris", which is the best available and is
not the same thing as "plastic". `evaluate.transfer_warning()` prints this
alongside any Guinea prediction so the caveat travels with the number.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import features as F
from .data import TARGET

SEED = 20240119


def _rf(n_estimators: int = 400, **kw):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        # Debris is rarer than 1%, so without this the forest optimises the
        # majority class and predicts nothing. "balanced_subsample"
        # reweights within each bootstrap rather than globally, which suits
        # the very skewed draws this data produces.
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED, **kw)


def train_binary(X_tr: np.ndarray, y_tr: np.ndarray, **kw):
    """Debris vs rest. Returns (model, feature_names)."""
    feats = F.build(X_tr)
    m = _rf(**kw)
    m.fit(feats.values, (y_tr == TARGET).astype(int))
    return m, list(feats.columns)


def train_multiclass(X_tr: np.ndarray, y_tr: np.ndarray, **kw):
    """All MARIDA classes. Returns (model, feature_names)."""
    feats = F.build(X_tr)
    m = _rf(**kw)
    m.fit(feats.values, y_tr)
    return m, list(feats.columns)


def predict_proba(model, X: np.ndarray, positive: int = 1) -> np.ndarray:
    feats = F.build(X)
    p = model.predict_proba(feats.values)
    classes = list(model.classes_)
    if positive not in classes:
        raise KeyError(f"class {positive} not in model classes {classes}")
    return p[:, classes.index(positive)]


def importances(model, names) -> "object":
    """
    Feature importances, with the caution they require.

    Impurity-based importances are biased toward high-cardinality features and
    are computed on the training set, so they describe what the forest USED,
    not what is causally informative. They are reported because a spectral
    index appearing at the top is a check that the model is keying on the
    physics it should be -- and a band index at the top, on brightness, would
    be a warning. They are not offered as evidence of mechanism.
    """
    import pandas as pd
    return (pd.DataFrame({"feature": names,
                          "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def threshold_for_recall(y_true: np.ndarray, score: np.ndarray,
                         target_recall: float) -> float:
    """
    The probability cut that achieves a given recall.

    Provided because comparing a probabilistic model to a binary rule at the
    model's default 0.5 cut is not a fair comparison: the FDI rule sits at
    whatever recall mean+3sigma happens to give, and the two must be compared
    at MATCHED recall or matched precision, never at arbitrary cuts.
    """
    t = np.asarray(y_true, bool)
    if not t.any():
        return float("nan")
    order = np.argsort(-score)
    hits = np.cumsum(t[order])
    need = int(np.ceil(target_recall * t.sum()))
    idx = int(np.searchsorted(hits, need))
    idx = min(idx, len(order) - 1)
    return float(score[order][idx])
